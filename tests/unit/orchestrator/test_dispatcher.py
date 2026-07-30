import asyncio
import uuid
from collections import Counter

import verifiers.v1 as vf
from verifiers.v1.clients import EvalClientConfig

from prime_rl.orchestrator.dispatcher import RolloutDispatcher
from prime_rl.orchestrator.types import InflightRollout, Policy, RolloutKind
from prime_rl.utils.client import ClientIdentity, client_identity


class FakeInferencePool:
    def __init__(self, clients: list[vf.ClientConfig]) -> None:
        self.train_clients = clients
        self._eval_index = 0

    async def select_train_client(self, load: Counter[ClientIdentity]) -> vf.ClientConfig:
        return min(self.train_clients, key=lambda client: load[client_identity(client)])

    async def get_eval_client(self) -> vf.ClientConfig:
        client = self.train_clients[self._eval_index % len(self.train_clients)]
        self._eval_index += 1
        return client


def make_dispatcher(loads: list[tuple[vf.ClientConfig, int]], cap: int | None = 40) -> RolloutDispatcher:
    dispatcher = RolloutDispatcher(
        train_envs=object(),
        eval_envs=None,
        train_source=object(),
        eval_source=None,
        policy_pool=object(),
        policy=Policy(),
        max_inflight_rollouts=256,
        max_inflight_rollouts_per_client=cap,
        tasks_per_minute=None,
        max_off_policy_steps=2,
    )
    dispatcher.inflight = {
        object(): InflightRollout(
            kind="train",
            env_name="env",
            group_id=uuid.uuid4(),
            policy_version=0,
            rollout_count=count,
            client_config=client,
        )
        for client, count in loads
    }
    return dispatcher


def select(dispatcher: RolloutDispatcher, pool: FakeInferencePool, kind: RolloutKind) -> vf.ClientConfig | None:
    return asyncio.run(dispatcher._select_client_for_group(pool, kind, 16))


def test_train_client_respects_projected_group_size() -> None:
    clients = [EvalClientConfig(base_url=f"http://{name}/v1") for name in ("a", "b")]
    pool = FakeInferencePool(clients)

    # 24 + 16 fits the cap; 25 + 16 does not.
    assert select(make_dispatcher([(clients[0], 24), (clients[1], 25)]), pool, "train") is clients[0]
    assert select(make_dispatcher([(clients[0], 25), (clients[1], 25)]), pool, "train") is None


def test_eval_round_robin_skips_clients_over_cap() -> None:
    clients = [EvalClientConfig(base_url=f"http://{name}/v1") for name in ("a", "b")]
    pool = FakeInferencePool(clients)

    # Skip overloaded A; defer when both are full; then resume round-robin at A.
    assert select(make_dispatcher([(clients[0], 25), (clients[1], 24)]), pool, "eval") is clients[1]
    assert select(make_dispatcher([(clients[0], 25), (clients[1], 25)]), pool, "eval") is None
    assert select(make_dispatcher([(clients[0], 24), (clients[1], 25)]), pool, "eval") is clients[0]


def test_client_cap_is_optional() -> None:
    client = EvalClientConfig(base_url="http://a/v1")
    pool = FakeInferencePool([client])
    dispatcher = make_dispatcher([(client, 100)], cap=None)

    assert select(dispatcher, pool, "train") is client
    assert select(dispatcher, pool, "eval") is client
