from __future__ import annotations

from safeloop.adapters.base import BenchmarkSample


def make_agentic_rag_samples() -> list[BenchmarkSample]:
    return [
        BenchmarkSample(
            request_id="agentic_rag_calc_001",
            prompt=(
                "Docs: The invoice has 18 boxes and each box has 7 units. "
                "Use the calculator tool if needed. JSON:"
            ),
            task="agentic_rag",
            stage="tool_call_json",
            expected='{"tool":"calculator","arguments":{"expression":"18*7"}}',
        ),
        BenchmarkSample(
            request_id="agentic_rag_entity_001",
            prompt=(
                "Docs: Ada Lovelace wrote notes on the Analytical Engine. "
                "Answer with citation:"
            ),
            task="agentic_rag",
            stage="final_answer",
            expected="Ada Lovelace",
        ),
    ]

