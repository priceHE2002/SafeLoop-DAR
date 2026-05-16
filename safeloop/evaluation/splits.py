from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import TypeVar

from safeloop.types import FeatureRow

T = TypeVar("T", bound=FeatureRow)


def _stable_key(request_id: str) -> str:
    return hashlib.sha256(request_id.encode("utf-8")).hexdigest()


def split_by_request(rows: Sequence[T], train_fraction: float = 0.7) -> tuple[list[T], list[T]]:
    """Split rows by request_id without leaking depths from the same request.

    If the input has only one request, both splits receive the same rows. That
    fallback keeps smoke tests usable, but real experiments should use enough
    requests for a disjoint split.
    """

    if not rows:
        return [], []
    request_ids = sorted({row.request_id for row in rows}, key=_stable_key)
    if len(request_ids) == 1:
        one = list(rows)
        return one, one
    cutoff = round(len(request_ids) * train_fraction)
    cutoff = max(1, min(cutoff, len(request_ids) - 1))
    train_ids = set(request_ids[:cutoff])
    train = [row for row in rows if row.request_id in train_ids]
    test = [row for row in rows if row.request_id not in train_ids]
    return train, test

