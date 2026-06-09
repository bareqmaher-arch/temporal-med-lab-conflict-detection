from src.config import SCENARIOS
from src.rules.baseline_rules import static_fires
from src.rules.temporal_rules import temporal_fires

ACE = SCENARIOS["ace_potassium"]


def test_static_fires_above_threshold():
    assert static_fires(ACE, 5.5) is True
    assert static_fires(ACE, 5.0) is False
    assert static_fires(ACE, None) is False


def test_temporal_fires_on_sustained_rise():
    feats = {"current_value": 5.1, "delta_value": 0.7,
             "consecutive_abnormal_trend": 3, "slope_30d": 0.05}
    assert temporal_fires(ACE, feats) is True


def test_temporal_does_not_fire_on_transient_spike():
    # a single spike (consecutive == 1) past the gate must NOT fire the temporal rule
    feats = {"current_value": 5.6, "delta_value": 1.2,
             "consecutive_abnormal_trend": 1, "slope_30d": 0.0}
    assert temporal_fires(ACE, feats) is False


def test_temporal_requires_both_change_and_current_gate():
    # large delta but current below gate -> no fire
    feats = {"current_value": 4.8, "delta_value": 0.9,
             "consecutive_abnormal_trend": 3, "slope_30d": 0.05}
    assert temporal_fires(ACE, feats) is False
