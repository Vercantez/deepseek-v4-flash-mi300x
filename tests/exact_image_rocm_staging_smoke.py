"""Run inside the pinned vLLM image with the GPU-worker overlay mounted."""

import ctypes
import os

import numpy as np
import torch

from vllm.v1.kv_offload.cpu.gpu_worker import (
    SingleDirectionOffloadingHandler,
    _swap_blocks_rocm_stream_ordered,
    copy_host_pages,
)


def main() -> None:
    first_source = (ctypes.c_uint8 * 4)(1, 2, 3, 4)
    second_source = (ctypes.c_uint8 * 3)(5, 6, 7)
    first_destination = (ctypes.c_uint8 * 4)()
    second_destination = (ctypes.c_uint8 * 3)()

    elapsed = copy_host_pages(
        np.array(
            [ctypes.addressof(first_source), ctypes.addressof(second_source)],
            dtype=np.uint64,
        ),
        np.array(
            [ctypes.addressof(first_destination), ctypes.addressof(second_destination)],
            dtype=np.uint64,
        ),
        np.array([4, 3], dtype=np.int64),
    )

    assert elapsed >= 0
    assert list(first_destination) == [1, 2, 3, 4]
    assert list(second_destination) == [5, 6, 7]

    # Descriptor pools retain their largest allocation. A smaller later store
    # must copy only its active destination count, not walk stale tail sizes.
    staged = torch.tensor([8, 9, 10, 11], dtype=torch.uint8)
    pooled_sizes = torch.tensor([2, 2, 99], dtype=torch.int64)
    first_active_destination = (ctypes.c_uint8 * 2)()
    second_active_destination = (ctypes.c_uint8 * 2)()
    transfer = type("Transfer", (), {
        "staging": staged,
        "host_copy_destinations": np.array([
            ctypes.addressof(first_active_destination),
            ctypes.addressof(second_active_destination),
        ], dtype=np.uint64),
        "batch_sizes": pooled_sizes,
    })()
    SingleDirectionOffloadingHandler._complete_host_copy(transfer)
    assert list(first_active_destination) == [8, 9]
    assert list(second_active_destination) == [10, 11]

    # Exercise the exact ROCm path used for staged KV stores and restores.
    # The production crash used 273 descriptors and about 28 MB per rank.
    if torch.cuda.is_available() and torch.version.hip is not None:
        descriptor_count = 273
        bytes_per_descriptor = 128 * 1024
        total_bytes = descriptor_count * bytes_per_descriptor
        gpu = torch.empty(total_bytes, device="cuda", dtype=torch.uint8)
        host = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=True)
        offsets = torch.arange(descriptor_count, dtype=torch.int64)
        offsets *= bytes_per_descriptor
        src_ptrs = offsets + gpu.data_ptr()
        dst_ptrs = offsets + host.data_ptr()
        sizes = torch.full(
            (descriptor_count,), bytes_per_descriptor, dtype=torch.int64
        )
        iterations = int(os.environ.get("ROCM_STAGING_STRESS_ITERS", "1"))
        for iteration in range(iterations):
            expected = (iteration % 251) + 1
            gpu.fill_(expected)
            host.zero_()
            _swap_blocks_rocm_stream_ordered(src_ptrs, dst_ptrs, sizes)
            torch.cuda.synchronize()
            assert int(host[0]) == expected
            assert int(host[-1]) == expected

            gpu.zero_()
            _swap_blocks_rocm_stream_ordered(dst_ptrs, src_ptrs, sizes)
            torch.cuda.synchronize()
            assert int(gpu[0]) == expected
            assert int(gpu[-1]) == expected
    print("exact-image ROCm staged host-copy smoke: PASS")


if __name__ == "__main__":
    main()
