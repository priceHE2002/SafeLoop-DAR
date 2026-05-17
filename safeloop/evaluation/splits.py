from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

from safeloop.types import FeatureRow

T = TypeVar("T", bound=FeatureRow)


@dataclass(slots=True)
class RequestSplitIds:
    train: list[str]
    calibration: list[str]
    test: list[str]

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "train": self.train,
            "calibration": self.calibration,
            "test": self.test,
        }


def _stable_key(request_id: str) -> str:
    return hashlib.sha256(request_id.encode("utf-8")).hexdigest()


def _stable_key_with_salt(request_id: str, salt: str = "") -> str:
    return hashlib.sha256(f"{salt}:{request_id}".encode("utf-8")).hexdigest()


def split_request_ids_three_way(
    request_ids: Sequence[str],
    train_fraction: float = 0.5,
    calibration_fraction: float = 0.25,
    salt: str = "",
) -> RequestSplitIds:
    """Split request ids into disjoint train/calibration/test partitions."""

    unique_ids = sorted({str(request_id) for request_id in request_ids}, key=lambda item: _stable_key_with_salt(item, salt))
    if not unique_ids:
        return RequestSplitIds([], [], [])
    if len(unique_ids) == 1:
        return RequestSplitIds(unique_ids, unique_ids, unique_ids)
    if len(unique_ids) == 2:
        return RequestSplitIds([unique_ids[0]], [unique_ids[0]], [unique_ids[1]])

    train_cutoff = round(len(unique_ids) * train_fraction)
    calibration_count = round(len(unique_ids) * calibration_fraction)
    train_cutoff = max(1, min(train_cutoff, len(unique_ids) - 2))
    calibration_count = max(1, min(calibration_count, len(unique_ids) - train_cutoff - 1))
    calibration_cutoff = train_cutoff + calibration_count
    return RequestSplitIds(
        train=unique_ids[:train_cutoff],
        calibration=unique_ids[train_cutoff:calibration_cutoff],
        test=unique_ids[calibration_cutoff:],
    )


def split_by_request_three_way(
    rows: Sequence[T],
    train_fraction: float = 0.5,
    calibration_fraction: float = 0.25,
    salt: str = "",
) -> tuple[list[T], list[T], list[T], RequestSplitIds]:
    """Split rows by request_id into train/calibration/test.

    Small smoke tests with fewer than three request ids intentionally reuse the
    train partition for calibration so scripts remain runnable. Formal
    experiments must use enough requests for disjoint splits.
    """

    split_ids = split_request_ids_three_way(
        [row.request_id for row in rows],
        train_fraction=train_fraction,
        calibration_fraction=calibration_fraction,
        salt=salt,
    )
    train_ids = set(split_ids.train)
    calibration_ids = set(split_ids.calibration)
    test_ids = set(split_ids.test)
    return (
        [row for row in rows if row.request_id in train_ids],
        [row for row in rows if row.request_id in calibration_ids],
        [row for row in rows if row.request_id in test_ids],
        split_ids,
    )


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
