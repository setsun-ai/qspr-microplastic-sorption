"""Tests for run_analysis.py: data shape, split integrity, no train/test leakage, regression values."""
import numpy as np
import pytest

import run_analysis as ra


@pytest.fixture(scope="module")
def df():
    return ra.load_data()


def test_data_shape(df):
    assert len(df) == 35
    assert df[ra.LSER + [ra.TARGET]].notna().all().all()
    assert df[ra.NAME].is_unique


def test_split_sizes_and_no_overlap(df):
    tr, te = ra.split(len(df))
    assert (len(tr), len(te)) == (24, 11)
    assert set(tr).isdisjoint(te)
    assert sorted(np.concatenate([tr, te])) == list(range(len(df)))


@pytest.mark.parametrize("seed", [0, 7, 42, 199])
def test_stability_splits_are_disjoint(df, seed):
    tr, te = ra.split(len(df), seed=seed)
    assert set(tr).isdisjoint(te) and len(tr) + len(te) == len(df)


def test_scaler_is_fitted_on_training_rows_only(df):
    X = df[ra.FINAL].to_numpy(float)
    y = df[ra.TARGET].to_numpy(float)
    tr, _ = ra.split(len(df))
    model = ra.make_model("MLR").fit(X[tr], y[tr])
    scaler = model.named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_, X[tr].mean(axis=0))
    assert not np.allclose(scaler.mean_, X.mean(axis=0))


def test_test_set_targets_do_not_influence_training_side(df):
    """Corrupting the validation-set targets must not change anything computed from the training set:
    descriptor-subset search, k selection, SVR tuning, fitted parameters and CV scores."""
    tr, te = ra.split(len(df))
    corrupted = df.copy()
    corrupted.loc[te, ra.TARGET] += 10.0

    clean, dirty = ra.main_split(df), ra.main_split(corrupted)
    assert clean["subset_search_on_TS"] == dirty["subset_search_on_TS"]
    for a, b in zip(clean["table"], dirty["table"], strict=True):
        assert a["model"] == b["model"]  # same tuned k and SVR hyper-parameters
        assert a["R2_TS"] == pytest.approx(b["R2_TS"])
        assert a["Q2_CV"] == pytest.approx(b["Q2_CV"])
        assert a["Q2_ext"] != pytest.approx(b["Q2_ext"])  # only the external metric may move


def test_subset_search_on_training_set_recovers_final_pair(df):
    tr, _ = ra.split(len(df))
    best, _ = ra.select_subset(df.loc[tr, ra.LSER].to_numpy(float), df.loc[tr, ra.TARGET].to_numpy(float), ra.LSER)
    assert best == ra.FINAL


def test_final_model_matches_report(df):
    final = ra.final_model(df)
    assert final["R2"] == pytest.approx(0.939, abs=1e-3)
    assert final["Q2_LOO"] == pytest.approx(0.913, abs=1e-3)
    assert final["equation_raw"]["log D"] == pytest.approx(0.751, abs=1e-3)
    assert final["equation_raw"]["εβ"] == pytest.approx(-19.32, abs=1e-2)
    assert max(final["VIF"].values()) < 1.1


def test_quick_run_writes_outputs(tmp_path):
    ra.main(["--quick", "--out", str(tmp_path)])
    for name in ["main_split_metrics.csv", "stability_q2ext.csv", "external_predictions.csv",
                 "reported_vs_reproduced.csv", "summary.json"]:
        assert (tmp_path / name).stat().st_size > 0
