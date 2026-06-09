from src.risk.risk_score import band_for, compute_risk_score


def test_band_mapping():
    assert band_for(10) == "Low"
    assert band_for(45) == "Moderate"
    assert band_for(70) == "High"
    assert band_for(95) == "Critical"


def test_score_in_range_and_monotonic():
    low = {"current_value": 4.3, "delta_value": 0.1, "percent_change": 0.02,
           "slope_7d": 0.0, "renal_impairment": 0, "hepatic_impairment": 0,
           "age": 50, "polypharmacy_count": 1, "consecutive_abnormal_trend": 0}
    high = {"current_value": 6.0, "delta_value": 1.5, "percent_change": 0.35,
            "slope_7d": 0.08, "renal_impairment": 1, "hepatic_impairment": 0,
            "age": 80, "polypharmacy_count": 8, "consecutive_abnormal_trend": 4}
    lo = compute_risk_score(low, 0.9, +1, "potassium")
    hi = compute_risk_score(high, 0.9, +1, "potassium")
    assert 0 <= lo["risk_score"] <= 100
    assert 0 <= hi["risk_score"] <= 100
    assert hi["risk_score"] > lo["risk_score"]


def test_resolving_trend_does_not_inflate_delta():
    feats = {"current_value": 4.0, "delta_value": -0.5, "percent_change": -0.1,
             "slope_7d": 0.0, "renal_impairment": 0, "hepatic_impairment": 0,
             "age": 40, "polypharmacy_count": 0, "consecutive_abnormal_trend": 0}
    out = compute_risk_score(feats, 0.9, +1, "potassium")
    assert out["components"]["delta_from_baseline"] == 0.0
