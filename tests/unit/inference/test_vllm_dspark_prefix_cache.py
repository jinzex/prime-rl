from types import SimpleNamespace

import torch
from vllm.v1.core.kv_cache_coordinator import (
    HybridKVCacheCoordinator,
    SpecGroup,
)
from vllm.v1.core.kv_cache_utils import _annotate_eagle_groups
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, SlidingWindowSpec


def test_dspark_noncausal_draft_group_is_veto_exempt():
    spec = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float16,
        sliding_window=64,
        non_causal_multi_token_decode=True,
    )
    group = KVCacheGroupSpec(["draft.attn"], spec)
    speculative_config = SimpleNamespace(
        use_eagle=lambda: True,
        use_dspark=lambda: True,
    )

    _annotate_eagle_groups(
        SimpleNamespace(speculative_config=speculative_config),
        {"draft.attn": spec},
        [group],
    )

    assert group.is_eagle_group
    assert group.eagle_group_is_veto_exempt


class _FakeManager:
    block_size = 576
    hits: dict[int, int] = {}

    @classmethod
    def find_longest_cache_hit(
        cls,
        *,
        max_length,
        kv_cache_group_ids,
        drop_eagle_block,
        **_kwargs,
    ):
        hit_length = min(cls.hits[kv_cache_group_ids[0]], max_length)
        if drop_eagle_block:
            hit_length = max(hit_length - cls.block_size, 0)
        blocks = [[object()]] if hit_length else [[]]
        return blocks, hit_length


def _coordinator(group_hits: dict[int, int]) -> HybridKVCacheCoordinator:
    _FakeManager.hits = group_hits
    coordinator = object.__new__(HybridKVCacheCoordinator)
    coordinator.kv_cache_config = SimpleNamespace(kv_cache_groups=[None] * 4)
    coordinator.attention_groups = [
        SpecGroup(object(), [0], _FakeManager, False),
        SpecGroup(object(), [1], _FakeManager, False),
        SpecGroup(object(), [2], _FakeManager, True, True),
        SpecGroup(object(), [3], _FakeManager, False),
    ]
    coordinator.single_type_managers = [_FakeManager()] * 4
    coordinator.block_pool = object()
    coordinator.veto_exempt_eagle_group_ids = {2}
    coordinator.enable_partial_hash_hits = False
    coordinator.hash_block_size = 1
    coordinator.scheduler_block_size = 1
    coordinator.dcp_world_size = 1
    return coordinator


def test_dspark_prefix_hit_waits_for_shared_target_checkpoint():
    coordinator = _coordinator({0: 1728, 1: 1728, 2: 1728, 3: 0})

    _, hit_length, checkpoint_length = coordinator.find_longest_cache_hit([], 1728)

    assert hit_length == 0
    assert checkpoint_length == 1152

    _FakeManager.hits = {0: 1152, 1: 1152, 2: 0, 3: 1152}
    _, hit_length, checkpoint_length = coordinator.find_longest_cache_hit([], 1728)

    assert hit_length == 1152
    assert checkpoint_length == 0
