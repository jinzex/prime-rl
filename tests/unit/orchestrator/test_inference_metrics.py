from prime_rl.orchestrator.inference_metrics import (
    EndpointSample,
    EngineRollup,
    MetricsEndpoint,
    NodeRollup,
    TimedRollup,
    build_scope_metrics,
    parse_prometheus_text,
)


def test_parse_spec_decode_counters():
    rollup = parse_prometheus_text(
        """
# TYPE vllm:spec_decode_num_drafts counter
vllm:spec_decode_num_drafts_total{engine="0"} 10
# TYPE vllm:spec_decode_num_draft_tokens counter
vllm:spec_decode_num_draft_tokens_total{engine="0"} 30
# TYPE vllm:spec_decode_num_accepted_tokens counter
vllm:spec_decode_num_accepted_tokens_total{engine="0"} 24
"""
    )

    engine = rollup.engines["0"]
    assert engine.spec_decode_num_drafts_total == 10
    assert engine.spec_decode_num_draft_tokens_total == 30
    assert engine.spec_decode_num_accepted_tokens_total == 24


def test_spec_decode_metrics_use_aggregate_counter_deltas():
    endpoints = [
        MetricsEndpoint(client=None, role=None, key="a", index=0),  # type: ignore[arg-type]
        MetricsEndpoint(client=None, role=None, key="b", index=1),  # type: ignore[arg-type]
    ]
    previous = {
        "a": TimedRollup(
            timestamp=0,
            rollup=NodeRollup(
                engines={
                    "0": EngineRollup(
                        spec_decode_num_drafts_total=100,
                        spec_decode_num_draft_tokens_total=300,
                        spec_decode_num_accepted_tokens_total=200,
                    )
                }
            ),
        ),
        "b": TimedRollup(
            timestamp=0,
            rollup=NodeRollup(
                engines={
                    "0": EngineRollup(
                        spec_decode_num_drafts_total=200,
                        spec_decode_num_draft_tokens_total=600,
                        spec_decode_num_accepted_tokens_total=300,
                    )
                }
            ),
        ),
    }
    samples = [
        EndpointSample(
            endpoint=endpoints[0],
            timestamp=10,
            rollup=NodeRollup(
                engines={
                    "0": EngineRollup(
                        spec_decode_num_drafts_total=110,
                        spec_decode_num_draft_tokens_total=325,
                        spec_decode_num_accepted_tokens_total=220,
                    )
                }
            ),
        ),
        EndpointSample(
            endpoint=endpoints[1],
            timestamp=10,
            rollup=NodeRollup(
                engines={
                    "0": EngineRollup(
                        spec_decode_num_drafts_total=290,
                        spec_decode_num_draft_tokens_total=870,
                        spec_decode_num_accepted_tokens_total=390,
                    )
                }
            ),
        ),
    ]

    metrics = build_scope_metrics("agg", samples, previous)

    assert metrics["inference/agg/spec_decode_mean_acceptance_length"] == 2.1
    assert metrics["inference/agg/spec_decode_draft_acceptance_rate"] == 110 / 295
