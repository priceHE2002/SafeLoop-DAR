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

