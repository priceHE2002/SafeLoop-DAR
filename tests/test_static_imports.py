def test_imports():
    import safeloop  # noqa: F401
    from safeloop.features.builder import FeatureBuilder  # noqa: F401
    from safeloop.risk.calibration import GroupCalibrator  # noqa: F401
    from safeloop.risk.predictor import OnlineLogisticRiskPredictor  # noqa: F401

