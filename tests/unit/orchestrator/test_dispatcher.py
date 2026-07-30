import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock

from prime_rl.orchestrator.dispatcher import RolloutDispatcher
from prime_rl.orchestrator.types import GroupState, Policy


def test_dispatcher_defers_new_groups_when_no_client_is_eligible():
    async def run() -> None:
        pool = MagicMock()
        pool.get_eval_client = AsyncMock(return_value=None)
        pool.select_train_client = AsyncMock(return_value=None)
        train_envs = MagicMock()
        train_envs.get.return_value.sampler.pool = pool
        train_envs.get.return_value.sampler.samples_from_live_policy = True
        dispatcher = RolloutDispatcher(
            train_envs=train_envs,
            eval_envs=MagicMock(),
            train_source=MagicMock(),
            eval_source=MagicMock(),
            policy_pool=pool,
            policy=Policy(model_name="test-model"),
            max_inflight_rollouts=32,
            tasks_per_minute=None,
            max_off_policy_steps=2,
            client_eligibility=lambda: set(),
        )

        for kind in ("train", "eval"):
            group_id = uuid.uuid4()
            group = GroupState(
                kind=kind,
                env_name="test-env",
                task_idx=0,
                rollouts_to_schedule=16,
                target_rollouts=16,
            )
            dispatcher.groups[group_id] = group
            assert await dispatcher.schedule_group_rollout(group_id, group) is False
            assert group.pinned_client is None

        pool.select_train_client.assert_awaited_once_with({}, set())
        pool.get_eval_client.assert_awaited_once_with(set())

    asyncio.run(run())


def test_dispatcher_gates_only_unpinned_policy_groups():
    async def run() -> None:
        policy_pool = MagicMock()
        frozen_pool = MagicMock()
        frozen_pool.select_train_client = AsyncMock(return_value=None)
        train_envs = MagicMock()
        train_envs.get.return_value.sampler.pool = frozen_pool
        train_envs.get.return_value.sampler.samples_from_live_policy = False
        eligibility = MagicMock(return_value=set())
        dispatcher = RolloutDispatcher(
            train_envs=train_envs,
            eval_envs=None,
            train_source=MagicMock(),
            eval_source=None,
            policy_pool=policy_pool,
            policy=Policy(model_name="test-model"),
            max_inflight_rollouts=32,
            tasks_per_minute=None,
            max_off_policy_steps=2,
            client_eligibility=eligibility,
        )

        pinned_id = uuid.uuid4()
        pinned = GroupState(
            kind="eval",
            env_name="test-env",
            task_idx=0,
            rollouts_to_schedule=15,
            target_rollouts=16,
            pinned_client=MagicMock(),
        )
        dispatcher.groups[pinned_id] = pinned
        assert await dispatcher.schedule_group_rollout(pinned_id, pinned) is False

        frozen_id = uuid.uuid4()
        frozen = GroupState(
            kind="train",
            env_name="test-env",
            task_idx=0,
            rollouts_to_schedule=16,
            target_rollouts=16,
        )
        dispatcher.groups[frozen_id] = frozen
        assert await dispatcher.schedule_group_rollout(frozen_id, frozen) is False

        eligibility.assert_not_called()
        policy_pool.get_eval_client.assert_not_called()
        frozen_pool.select_train_client.assert_awaited_once_with({}, None)

    asyncio.run(run())
