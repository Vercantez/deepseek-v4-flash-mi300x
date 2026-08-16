"""Run inside the pinned vLLM image with the GPU-worker overlay mounted."""

import ctypes

import numpy as np
import torch

from vllm.v1.kv_offload.cpu.gpu_worker import (
    SingleDirectionOffloadingHandler,
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
    print("exact-image ROCm staged host-copy smoke: PASS")


if __name__ == "__main__":
    main()
