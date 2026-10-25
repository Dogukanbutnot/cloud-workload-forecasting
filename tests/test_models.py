import numpy as np
import pandas as pd
import pytest

from cwfc.baselines import mean_forecast, persistence_forecast
from cwfc.evaluation import run_rolling_origin
from cwfc.models import (
    ALPHAS,
    PHIS,
    MeanRevertingLevelFit,
    damped_persistence_forecast,
    fit_mean_reverting_level,
    mean_reverting_level_forecast,
    model_suite,
    ses_levels,
)
from cwfc.rolling import rolling_origin_folds


def frame(values, step=300):
    idx = pd.Index(np.arange(len(values), dtype="int64") * step, name="timestamp")
    return pd.DataFrame({"y": np.asarray(values, dtype=float)}, index=idx)


def template(n, start=10**6):
    return pd.DataFrame(index=pd.Index(start + np.arange(n) * 300, name="timestamp"))


def ar1(phi, n, rng, mean=10.0):
    y, e = np.zeros(n), rng.normal(0, 1, n)
    for i in range(1, n):
        y[i] = phi * y[i - 1] + e[i]
    return y + mean


# ------------------------------ building blocks ------------------------------ #


def test_ses_levels_matches_the_recursion_and_the_limits():
    y = np.array([1.0, 3.0, 2.0, 6.0])
    assert np.allclose(ses_levels(y, 1.0), y)  # alpha=1 is the series itself
    lvl = ses_levels(y, 0.5)
    assert lvl[0] == 1.0 and lvl[1] == pytest.approx(2.0) and lvl[2] == pytest.approx(2.0)
    assert lvl[3] == pytest.approx(4.0)


def test_forecast_formula_decays_geometrically_toward_the_mean():
    fit = MeanRevertingLevelFit(alpha=0.5, phi=0.8, level=12.0, mean=10.0, inner_loss=0.0, n_obs=50)
    assert fit.forecast(3).tolist() == pytest.approx([11.6, 11.28, 11.024])
    far = fit.forecast(60)
    assert (np.diff(far[:20]) < 0).all()  # strictly closer to the mean at every step...
    assert abs(far[-1] - 10.0) < 1e-5  # ...and practically there after 60 steps


def test_grids_are_sane():
    assert max(ALPHAS) == 1.0 and min(ALPHAS) > 0
    assert max(PHIS) < 1.0 and min(PHIS) > 0  # phi=1 would make the mean irrelevant


# ------------------------------ estimation ------------------------------ #


def test_pure_ar1_is_recovered():
    rng = np.random.default_rng(1)
    fit = fit_mean_reverting_level(ar1(0.8, 3000, rng), horizon=24)
    assert fit.alpha == 1.0  # no measurement noise: no reason to smooth
    assert fit.phi == pytest.approx(0.8)


def test_slow_level_with_noise_prefers_smoothing_and_slow_reversion():
    rng = np.random.default_rng(2)
    level = 10 + np.cumsum(rng.normal(0, 0.05, 400))
    fit = fit_mean_reverting_level(level + rng.normal(0, 1, 400), horizon=24)
    assert fit.alpha <= 0.4 and fit.phi >= 0.85


def test_white_noise_forecast_stays_at_the_mean():
    rng = np.random.default_rng(3)
    y = 10 + rng.normal(0, 1, 400)
    fit = fit_mean_reverting_level(y, horizon=24)
    assert np.abs(fit.forecast(24) - y.mean()).max() < 0.2


def test_constant_history_gives_the_constant_without_warnings():
    with np.errstate(all="raise"):
        fit = fit_mean_reverting_level(np.full(60, 7.0), horizon=24)
    assert fit.forecast(5).tolist() == [7.0] * 5 and fit.inner_loss == 0.0


def test_fit_is_deterministic():
    y = ar1(0.7, 120, np.random.default_rng(4))
    a, b = fit_mean_reverting_level(y, 24), fit_mean_reverting_level(y, 24)
    assert a == b


@pytest.mark.parametrize(
    "y,h,msg",
    [
        (np.r_[np.arange(30.0), np.nan], 5, "NaN"),
        (np.arange(8.0), 5, "too short"),
        (np.arange(40.0), 0, "positive"),
    ],
)
def test_bad_input_is_rejected(y, h, msg):
    with pytest.raises(ValueError, match=msg):
        fit_mean_reverting_level(y, h)


def test_inner_loss_equals_a_brute_force_computation_from_prefixes_only():
    # Independent re-implementation: at every inner origin s the forecast is built
    # from y[:s+1] alone (running mean and smoothed level of that prefix). The
    # vectorised fit must agree, which proves its inner loss never peeks ahead.
    rng = np.random.default_rng(5)
    y, n, horizon = ar1(0.7, 60, rng), 60, 6
    fit = fit_mean_reverting_level(y, horizon)
    start = max(10, n // 4)
    errors = []
    for s in range(start, n - 1):
        m = y[: s + 1].mean()
        level = ses_levels(y[: s + 1], fit.alpha)[-1]
        for k in range(1, min(horizon, n - 1 - start) + 1):
            if s + k <= n - 1:
                errors.append(abs(y[s + k] - (m + fit.phi**k * (level - m))))
    assert fit.inner_loss == pytest.approx(np.mean(errors))


# ------------------------------ forecaster adapters ------------------------------ #


def test_forecaster_uses_template_length_and_index_and_ignores_its_contents():
    y = ar1(0.7, 80, np.random.default_rng(6))
    hist = frame(y)
    plain = template(12)
    loaded = template(12).assign(y=999.0)  # a leaked target column must change nothing
    a = mean_reverting_level_forecast(hist, plain, "y")
    b = mean_reverting_level_forecast(hist, loaded, "y")
    assert list(a.index) == list(plain.index) and len(a) == 12
    assert a.tolist() == b.tolist() and np.isfinite(a).all()


def test_damped_persistence_is_the_alpha_equals_one_ablation():
    y = ar1(0.7, 90, np.random.default_rng(7))
    out = damped_persistence_forecast(frame(y), template(10), "y")
    fit = fit_mean_reverting_level(y, 10, alphas=(1.0,))
    assert fit.alpha == 1.0
    assert out.tolist() == pytest.approx(fit.forecast(10).tolist())


def test_model_suite_names_are_stable():
    assert set(model_suite()) == {"mean_reverting_level", "damped_persistence"}


# ------------------------------ nested, leakage-free evaluation ------------------------------ #


def test_fold_predictions_do_not_depend_on_values_after_the_origin():
    rng = np.random.default_rng(8)
    base = ar1(0.7, 80, rng)
    changed = base.copy()
    changed[62:] += rng.normal(0, 5, len(changed) - 62)  # different future from index 62 on
    folds = rolling_origin_folds(80, horizon=6, min_train_size=50, step=6)  # histories end at 49, 55, 61, 67, 73
    suite = {"model": mean_reverting_level_forecast, "damped": damped_persistence_forecast}
    pa = run_rolling_origin(frame(base), "y", suite, folds)
    pb = run_rolling_origin(frame(changed), "y", suite, folds)
    for fold in (0, 1, 2):  # histories end at 49, 55 and 61: identical in both series
        a = pa[pa.fold == fold].y_pred.to_numpy()
        b = pb[pb.fold == fold].y_pred.to_numpy()
        assert np.array_equal(a, b)
    # sensitivity check: once the history contains the changed values, forecasts differ
    assert not np.array_equal(pa[pa.fold == 3].y_pred.to_numpy(), pb[pb.fold == 3].y_pred.to_numpy())


def test_on_a_mean_reverting_process_the_model_beats_both_extreme_baselines():
    rng = np.random.default_rng(9)
    y = ar1(0.7, 400, rng)
    folds = rolling_origin_folds(400, horizon=24, min_train_size=60, step=24)
    suite = {"persistence": persistence_forecast, "damped": damped_persistence_forecast}
    suite["history_mean"] = mean_forecast
    preds = run_rolling_origin(frame(y), "y", suite, folds)
    mae = (preds.y_pred - preds.y_true).abs().groupby(preds.model).mean()
    assert mae["damped"] < mae["persistence"]
    assert mae["damped"] <= mae["history_mean"] * 1.02
