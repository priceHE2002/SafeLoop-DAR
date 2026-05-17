from __future__ import annotations

from safeloop.adapters.base import BenchmarkSample


def make_khop_samples(num_samples: int = 16, max_hops: int = 8) -> list[BenchmarkSample]:
    samples: list[BenchmarkSample] = []
    for idx in range(num_samples):
        hops = 1 + (idx % max_hops)
        chain = " -> ".join(f"n{j}" for j in range(hops + 1))
        samples.append(
            BenchmarkSample(
                request_id=f"khop_{idx:04d}",
                prompt=f"Follow the chain: {chain}. Final node:",
                task="synthetic_khop",
                stage="final_answer",
                expected=f"n{hops}",
                metadata={"hops": hops},
            )
        )
    return samples


def make_mixed_control_samples(num_repeats: int = 12) -> list[BenchmarkSample]:
    """Build a small mixed benchmark for controlled halt-aware experiments."""

    samples: list[BenchmarkSample] = []
    for idx in range(num_repeats):
        left = 20 + idx
        right = 7 + (idx % 5)
        samples.append(
            BenchmarkSample(
                request_id=f"mix_math_{idx:04d}",
                prompt=f"Question: {left} + {right} = ? Answer:",
                task="gsm8k",
                stage="final_answer",
                expected=str(left + right),
                metadata={"difficulty": "math_final_number"},
            )
        )
        samples.append(
            BenchmarkSample(
                request_id=f"mix_code_{idx:04d}",
                prompt=f"Write a Python function named add_{idx} that returns x + {idx}.",
                task="humaneval",
                stage="code_generation",
                expected=f"def add_{idx}",
                metadata={"entry_point": f"add_{idx}", "expected_contains": f"def add_{idx}"},
            )
        )
        samples.append(
            BenchmarkSample(
                request_id=f"mix_tool_{idx:04d}",
                prompt=f"Call calculator for {left}*{right}. JSON:",
                task="tool",
                stage="tool_call_json",
                expected=f'{{"tool":"calculator","arguments":{{"expression":"{left}*{right}"}}}}',
                metadata={"expected_tool": "calculator", "expected_expression": f"{left}*{right}"},
            )
        )
        samples.append(
            BenchmarkSample(
                request_id=f"mix_rag_{idx:04d}",
                prompt=(
                    "Context: Ada Lovelace wrote notes on the Analytical Engine. "
                    "Question: Who wrote the notes? Answer:"
                ),
                task="rag",
                stage="rag_entity",
                expected="Ada Lovelace",
                metadata={"gold_entities": ["Ada Lovelace"]},
            )
        )
        samples.append(
            BenchmarkSample(
                request_id=f"mix_text_{idx:04d}",
                prompt="Answer yes or no: is water wet?",
                task="rag",
                stage="final_answer",
                expected="yes",
                metadata={"difficulty": "low_risk_text", "gold_entities": ["yes"]},
            )
        )
    return samples
