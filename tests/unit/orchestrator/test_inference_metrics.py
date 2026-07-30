import asyncio

import httpx
import pytest

from prime_rl.orchestrator.inference_metrics import InferenceMetricsCollector


def test_waiting_request_eligibility_uses_complete_raw_metrics():
    async def run() -> None:
        metrics: dict[str, str | None] = {
            "worker-a": 'vllm:num_requests_waiting{engine="0"} 0\n',
            "worker-b": 'vllm:num_requests_waiting{engine="0"} 3\n',
        }

        def handle(request: httpx.Request) -> httpx.Response:
            text = metrics[request.url.host]
            if text is None:
                return httpx.Response(500)
            return httpx.Response(200, text=text)

        transport = httpx.MockTransport(handle)
        clients = [
            httpx.AsyncClient(base_url=f"http://{host}:8100", transport=transport) for host in ("worker-a", "worker-b")
        ]
        identities = [("http://worker-a:8000/v1", None), ("http://worker-b:8000/v1", None)]
        collector = InferenceMetricsCollector(
            clients,
            client_identities=identities,
            log_to_wandb=False,
        )

        await collector.collect_and_log()
        assert collector.eligible_clients(0) == {identities[0]}

        metrics["worker-b"] = "vllm:num_requests_running 1\n"
        await collector.collect_and_log()
        assert collector.eligible_clients(0) is None

        metrics["worker-b"] = "{"
        with pytest.raises(ValueError):
            await collector.collect_and_log()
        assert collector.eligible_clients(0) is None

        metrics["worker-b"] = None
        await collector.collect_and_log()
        assert collector.eligible_clients(0) is None

        await asyncio.gather(*(client.aclose() for client in clients))

    asyncio.run(run())
