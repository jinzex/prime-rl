import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from prime_rl.inference.vllm.server import pause


def test_pause_forwards_mode_and_clear_cache():
    engine = MagicMock()
    engine.pause_generation = AsyncMock()

    with patch("prime_rl.inference.vllm.server.engine_client", return_value=engine):
        response = asyncio.run(pause(MagicMock(), mode="wait", clear_cache=False))

    assert response == {"status": "paused"}
    engine.pause_generation.assert_awaited_once_with(mode="wait", clear_cache=False)
