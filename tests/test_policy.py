import numpy as np
import pandas as pd
import pytest

from cwfc.policy import evaluate_policy, margin_by_band, margins_for_steps, tradeoff_table


def preds():
    # one model, 2 folds x 6 steps; shortfall (actual - forecast) = step-dependent
    rows = []
    for fold in range(2):
        for step in range(1, 7):
            rows.append({"fold": fold, "model": "m", "step": step, "y_pred": 10.0,
                         "y_true": 10.0 + (0.1 * step if fold == 0 else -0.5)})
    return pd.DataFrame(rows)


def test_margin_is_a_floored_quantile_of_shortfall_per_band():
    m = margin_by_band(preds(), "m", quantile=1 - 1e-9, bands=((1, 3), (4, 6)))
    assert m[(1, 3)] == pytest.approx(0.3)  # largest positive shortfall in steps 1-3
    assert m[(4, 6)] == pytest.approx(0.6)
    neg = preds().assign(y_true=9.0)  # actual always below forecast -> margin floored at 0
    assert margin_by_band(neg, "m", 0.9, ((1, 6),))[(1, 6)] == 0.0


def test_margins_are_looked_up_by_step_and_the_last_band_extends():
    margins = {(1, 3): 0.1, (4, 6): 0.2}
    assert margins_for_steps([1, 3, 4, 6, 9], margins).tolist() == [0.1, 0.1, 0.2, 0.2, 0.2]


def test_evaluate_policy_counts_under_and_over_provisioning_exactly():
    y_true = np.array([10.0, 11.0, 9.0, 10.5])
    y_pred = np.array([10.0, 10.0, 10.0, 10.0])
    r = evaluate_policy(y_true, y_pred, np.full(4, 0.5))
    # capacity 10.5: step 2 is short by 0.5 (strictly above), step 4 exactly meets it
    assert r["under_rate"] == pytest.approx(0.25)
    assert r["mean_shortfall_when_under"] == pytest.approx(0.5)
    assert r["mean_surplus"] == pytest.approx((0.5 + 0 + 1.5 + 0) / 4)
    assert evaluate_policy(y_true, y_pred, np.zeros(4))["under_rate"] == pytest.approx(0.5)


def test_bigger_quantile_means_bigger_margin_fewer_shortfalls_more_surplus():
    rng = np.random.default_rng(0)
    p = pd.DataFrame({"fold": 0, "model": "m", "step": np.tile(np.arange(1, 7), 50),
                      "y_pred": 10.0, "y_true": 10.0 + rng.normal(0, 1, 300)})
    t = tradeoff_table(p, "m", quantiles=(0.5, 0.8, 0.95), bands=((1, 6),))
    assert t["mean_margin"].is_monotonic_increasing
    assert t["under_rate"].is_monotonic_decreasing
    assert t["mean_surplus"].is_monotonic_increasing


def test_bad_inputs_raise():
    with pytest.raises(ValueError, match="quantile"):
        margin_by_band(preds(), "m", quantile=1.0)
    with pytest.raises(ValueError, match="no scored predictions"):
        margin_by_band(preds(), "missing")
