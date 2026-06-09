import pytest

from src.models.dataset import build_scenario_dataset
from src.models.evaluate import binary_metrics
from src.models.train_model import train_scenario_models
from src.preprocessing.clean_labs import clean_labs
from src.preprocessing.clean_medications import clean_medications
from src.ingestion.synthetic_generator import SyntheticLoader


@pytest.fixture(scope="module")
def cohort():
    t = SyntheticLoader(n_patients=180, seed=7).load()
    return (t.patients, clean_medications(t.medications),
            clean_labs(t.labs), t.knowledge)


def test_dataset_has_labels_and_features(cohort):
    patients, meds, labs, knowledge = cohort
    df = build_scenario_dataset("ace_potassium", patients, meds, labs, knowledge)
    assert len(df) > 0
    assert set(df["label"].unique()).issubset({0, 1})
    assert df["label"].sum() > 0          # some positives present
    assert "slope_14d" in df.columns


def test_training_produces_valid_predictions(cohort):
    patients, meds, labs, knowledge = cohort
    df = build_scenario_dataset("ace_potassium", patients, meds, labs, knowledge)
    out = train_scenario_models(df)
    for name, r in out["results"].items():
        assert len(r["proba"]) == len(out["y_test"])
        assert all(0.0 <= p <= 1.0 for p in r["proba"])
        assert "auroc" in r["metrics"]


def test_binary_metrics_perfect_case():
    m = binary_metrics([0, 1, 1, 0], [0, 1, 1, 0])
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 1.0
    assert m["f1"] == 1.0
