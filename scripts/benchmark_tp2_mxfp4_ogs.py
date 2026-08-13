#!/usr/bin/env python3
"""Microbenchmark the TP2 DeepSeek-V4 MXFP4 expert kernels on one MI300X.

Run this only while the inference service is stopped.  The shapes mirror one
TP rank: hidden=1024, intermediate=16384, 256 experts, top-k=6.
"""

import argparse
import json
import statistics
import time

import torch
import triton

from vllm.utils.import_utils import import_triton_kernels

import_triton_kernels()

from triton_kernels.matmul_ogs import (  # noqa: E402
    FlexCtx,
    FnSpecs,
    FusedActivation,
    GatherIndx,
    PrecisionConfig,
    RoutingData,
    ScatterIndx,
    matmul_ogs,
)
from triton_kernels.matmul_ogs_details.opt_flags import (  # noqa: E402
    OptFlags,
    reset_opt_flags,
    set_opt_flags,
)
from triton_kernels.numerics import InFlexData  # noqa: E402
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp  # noqa: E402
from triton_kernels.target_info import get_cdna_version  # noqa: E402
from triton_kernels.tensor import make_ragged_tensor_metadata  # noqa: E402
from triton_kernels.topk import topk  # noqa: E402


@triton.jit
def silu_mul(inp):
    gate, up = triton.language.split(
        triton.language.reshape(inp, (inp.shape[0], inp.shape[1] // 2, 2))
    )
    gate = gate.to(triton.language.float32)
    up = up.to(triton.language.float32)
    return gate / (1.0 + triton.language.exp(-gate)) * up


def make_routing(batch: int, num_experts: int, top_k: int, device: str):
    logits = torch.randn((batch, num_experts), device=device, dtype=torch.float32)
    result = topk(logits, top_k, apply_softmax=True)
    dispatch = result.mask_metadata.row_sorted_indx
    combine = result.mask_metadata.col_sorted_indx
    metadata = make_ragged_tensor_metadata(
        result.mask_metadata.col_sum, dispatch.shape[0]
    )
    gates = result.vals.flatten()[combine]
    routing = RoutingData(
        gates, metadata.slice_sizes, num_experts, top_k, metadata
    )
    return routing, GatherIndx(combine, dispatch), ScatterIndx(dispatch, combine)


def quantize_mxfp4(weight: torch.Tensor):
    quantized, scales = downcast_to_mxfp(
        weight, torch.uint8, axis=1
    )
    return quantized, PrecisionConfig(
        flex_ctx=FlexCtx(rhs_data=InFlexData()), weight_scale=scales
    )


def flags(block_m, block_n, block_k, num_warps, waves_per_eu):
    return OptFlags(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=2,
        group_m=4,
        xcd_swizzle=8,
        w_cache_modifier=None,
        split_k=1,
        is_persistent=False,
        idle_sms=0,
        epilogue_subtile=1,
        arch=None,
        target_kernel_kwargs={
            "waves_per_eu": waves_per_eu,
            "matrix_instr_nonkdim": 16,
            "kpack": 1,
        },
    )


CONFIGS = {
    "production": flags(16, 128, 256, 4, 2),
    "bn64": flags(16, 64, 256, 4, 2),
    "bn256": flags(16, 256, 256, 4, 2),
    "bk128": flags(16, 128, 128, 4, 2),
    "bm32": flags(32, 128, 256, 4, 2),
    "warps8": flags(16, 128, 256, 8, 2),
    "waves0": flags(16, 128, 256, 4, 0),
    "waves1": flags(16, 128, 256, 4, 1),
    "waves3": flags(16, 128, 256, 4, 3),
}


def timed(fn, warmup: int, repetitions: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples), min(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    args = parser.parse_args()

    if get_cdna_version() != 3:
        raise SystemExit("This benchmark is intended for CDNA3/MI300X")

    device = "cuda"
    torch.manual_seed(7)
    num_experts = 256
    top_k = 6
    hidden = 1024
    local_intermediate = 8192

    print(json.dumps({"phase": "allocate_and_quantize", "batch": args.batch}))
    started = time.perf_counter()
    raw_w1 = torch.randn(
        (num_experts, hidden, local_intermediate * 2),
        device=device,
        dtype=torch.bfloat16,
    )
    w1, pc1 = quantize_mxfp4(raw_w1)
    del raw_w1
    torch.cuda.empty_cache()
    raw_w2 = torch.randn(
        (num_experts, local_intermediate, hidden),
        device=device,
        dtype=torch.bfloat16,
    )
    w2, pc2 = quantize_mxfp4(raw_w2)
    del raw_w2
    torch.cuda.empty_cache()

    x = torch.randn((args.batch, hidden), device=device, dtype=torch.bfloat16)
    routing, gather, scatter = make_routing(
        args.batch, num_experts, top_k, device
    )
    y1 = torch.empty(
        (1, args.batch * top_k, local_intermediate),
        device=device,
        dtype=torch.bfloat16,
    )
    y2 = torch.empty(
        (1, args.batch, hidden), device=device, dtype=torch.bfloat16
    )
    activation = FusedActivation(
        FnSpecs("silu_mul", silu_mul, (), reduction_n=2), ()
    )
    print(
        json.dumps(
            {
                "phase": "ready",
                "seconds": round(time.perf_counter() - started, 3),
                "free_bytes": torch.cuda.mem_get_info()[0],
            }
        ),
        flush=True,
    )

    reference = None
    for name, config in CONFIGS.items():
        try:
            reset_opt_flags()
            set_opt_flags(config)

            def run_w1():
                matmul_ogs(
                    x,
                    w1,
                    None,
                    routing,
                    gather_indx=gather,
                    precision_config=pc1,
                    fused_activation=activation,
                    y=y1,
                )

            w1_median, w1_min = timed(
                run_w1, args.warmup, args.repetitions
            )

            def run_w2():
                matmul_ogs(
                    y1.view(args.batch * top_k, local_intermediate),
                    w2,
                    None,
                    routing,
                    scatter_indx=scatter,
                    precision_config=pc2,
                    y=y2,
                )

            w2_median, w2_min = timed(
                run_w2, args.warmup, args.repetitions
            )
            output = y2.clone()
            if reference is None:
                reference = output
            diff = (output.float() - reference.float()).abs()
            print(
                json.dumps(
                    {
                        "config": name,
                        "flags": config.__dict__,
                        "w1_median_ms": round(w1_median, 5),
                        "w1_min_ms": round(w1_min, 5),
                        "w2_median_ms": round(w2_median, 5),
                        "w2_min_ms": round(w2_min, 5),
                        "total_median_ms": round(w1_median + w2_median, 5),
                        "max_abs_diff": float(diff.max()),
                        "mean_abs_diff": float(diff.mean()),
                    }
                ),
                flush=True,
            )
        except Exception as error:
            print(
                json.dumps(
                    {"config": name, "error": repr(error)}
                ),
                flush=True,
            )
        finally:
            reset_opt_flags()


if __name__ == "__main__":
    main()
