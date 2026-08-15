"""Run inside the pinned vLLM image with the scheduler overlay mounted."""

from types import SimpleNamespace

from vllm.v1.kv_offload.base import LookupResult
from vllm.v1.request import RequestStatus

import vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler as scheduler


def group(group_idx: int, *, skip_external_reuse: bool):
    return scheduler.GroupOffloadConfig(
        group_idx=group_idx,
        tokens_per_block=256,
        tokens_per_chunk=256,
        hashes_per_chunk=1,
        kv_event_group_spec=None,
        sliding_window_size_in_chunks=None,
        is_eagle_group=skip_external_reuse,
        eagle_group_is_veto_exempt=skip_external_reuse,
        skip_external_reuse=skip_external_reuse,
    )


class Manager:
    def __init__(self) -> None:
        self.lookups = []
        self.touches = []
        self.store_keys = None

    def lookup(self, key, _context):
        self.lookups.append(key)
        if str(key).startswith("draft"):
            raise AssertionError("DSpark draft key reached external lookup")
        return LookupResult.HIT

    def touch(self, keys, _context):
        keys = list(keys)
        assert not any(str(key).startswith("draft") for key in keys)
        self.touches.extend(keys)

    def prepare_store(self, keys, _context):
        self.store_keys = list(keys)
        return None


class RequestState(SimpleNamespace):
    def storable_chunks(self, _group_config, group_state, _num_tokens):
        return len(group_state.offload_keys)

    def advance_stored_idx(self, _num_tokens):
        pass


def main() -> None:
    target_group = group(0, skip_external_reuse=False)
    draft_group = group(1, skip_external_reuse=True)
    target_state = scheduler.RequestGroupState(offload_keys=["target-0", "target-1"])
    draft_state = scheduler.RequestGroupState(offload_keys=["draft-0", "draft-1"])
    manager = Manager()

    connector = scheduler.OffloadingConnectorScheduler.__new__(
        scheduler.OffloadingConnectorScheduler
    )
    connector.config = SimpleNamespace(
        kv_group_configs=(target_group, draft_group),
        blocks_per_chunk=1,
        offload_prompt_only=False,
    )
    connector.manager = manager
    connector._lookup_groups = (0, 1)
    connector._sliding_window_groups = ()
    connector._mamba_align_size = None
    connector._chunks_being_loaded = None
    connector._events_tracker = SimpleNamespace(record_lookup=lambda *args: None)
    connector._stores_enabled = True

    req = SimpleNamespace(
        request_id="req",
        num_tokens=512,
        num_prompt_tokens=512,
        num_computed_tokens=0,
        status=RequestStatus.RUNNING,
        is_finished=lambda: False,
    )
    state = RequestState(
        num_locally_computed_tokens=0,
        req=req,
        req_context=SimpleNamespace(req_id="req"),
        group_states=(target_state, draft_state),
        lookup_excluded_groups=frozenset(),
        max_offload_tokens=None,
    )

    # Even if the draft tier contains a tempting partial hit, it must not
    # participate in prefix selection at all.
    assert connector._lookup(state) == 512
    assert state.lookup_excluded_groups == frozenset({1})
    assert manager.lookups == ["target-0", "target-1"]

    connector._touch(state)
    assert manager.touches == ["target-0", "target-1"]

    # Store scheduling must likewise keep draft keys out of CPU/filesystem
    # tiers. Returning None stops after prepare_store, which is enough to
    # inspect the exact proposed key set without creating transfer metadata.
    target_state.block_ids = [10, 11]
    draft_state.block_ids = [20, 21]
    connector._req_status = {"req": state}
    connector._connector_stats = SimpleNamespace(increase_counter=lambda *args: None)
    output = SimpleNamespace(
        num_scheduled_tokens={"req": 512},
        finished_req_ids=(),
    )
    assert connector._build_store_jobs(output) == {}
    assert manager.store_keys == ["target-0", "target-1"]

    # Read-only cache mode must leave lookup/load behavior intact while never
    # constructing a GPU->host transfer or touching the manager's store path.
    manager.store_keys = None
    connector._stores_enabled = False
    assert connector._build_store_jobs(output) == {}
    assert manager.store_keys is None
    print("exact-image DSpark external-reuse exclusion smoke: PASS")


if __name__ == "__main__":
    main()
