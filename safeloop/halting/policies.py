from __future__ import annotations

from dataclasses import dataclass

from safeloop.risk.calibration import GroupCalibrator
from safeloop.risk.predictor import OnlineLogisticRiskPredictor
from safeloop.types import FeatureRow, HaltingDecision


@dataclass
class FixedDepthPolicy:
    depth: int

    def decide(self, row: FeatureRow) -> HaltingDecision:
        should_exit = row.depth >= self.depth
        return HaltingDecision(
            request_id=row.request_id,
            position=row.position,
            depth=row.depth,
            should_exit=should_exit,
            risk_score=0.0,
            threshold=0.0,
            group=row.group,
        )


@dataclass
class ScoreThresholdPolicy:
    predictor: OnlineLogisticRiskPredictor
    threshold: float

    def decide(self, row: FeatureRow) -> HaltingDecision:
        score = self.predictor.predict_features(row.features)
        return HaltingDecision(
            request_id=row.request_id,
            position=row.position,
            depth=row.depth,
            should_exit=score <= self.threshold,
            risk_score=score,
            threshold=self.threshold,
            group=row.group,
        )


@dataclass
class GroupCalibratedPolicy:
    predictor: OnlineLogisticRiskPredictor
    calibrator: GroupCalibrator

    def decide(self, row: FeatureRow) -> HaltingDecision:
        score = self.predictor.predict_features(row.features)
        threshold = self.calibrator.threshold_for(row.group)
        return HaltingDecision(
            request_id=row.request_id,
            position=row.position,
            depth=row.depth,
            should_exit=score <= threshold,
            risk_score=score,
            threshold=threshold,
            group=row.group,
        )

