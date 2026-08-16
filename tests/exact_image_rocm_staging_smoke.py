"""Run inside the pinned vLLM image with the GPU-worker overlay mounted."""

import ctypes

import numpy as np

from vllm.v1.kv_offload.cpu.gpu_worker import copy_host_pages


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
    print("exact-image ROCm staged host-copy smoke: PASS")


if __name__ == "__main__":
    main()
