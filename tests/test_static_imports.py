def test_imports():
    import safeloop  # noqa: F401
    from safeloop.features.builder import FeatureBuilder  # noqa: F401
    from safeloop.risk.calibration import GroupCalibrator  # noqa: F401
    from safeloop.risk.predictor import OnlineLogisticRiskPredictor  # noqa: F401
    from safeloop.risk.ucb_calibration import UCBGroupCalibrator  # noqa: F401
    from safeloop.evaluation.splits import split_by_request_three_way  # noqa: F401
    from safeloop.evaluation.task_degradation import annotate_task_degradation  # noqa: F401
    from safeloop.controlled.halt_aware import train_halt_aware_state  # noqa: F401
    from safeloop.controlled.tiny_loop_lm import train_tiny_loop_lm  # noqa: F401
