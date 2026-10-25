import numpy as np
import pandas as pd
import pytest

from cwfc import eda
from cwfc.targets import STEP_SECONDS

RNG = np.random.default_rng(42)


def matrix(values, step=STEP_SECONDS):
    idx = pd.Index(np.arange(values.shape[0], dtype="int64") * step, name="timestamp")
    return pd.DataFrame(values, index=idx, columns=[f"vm{i}" for i in range(values.shape[1])])


def long_rows(x, max_extra=0.0):
    df = x.stack().rename("avg_cpu").reset_index()
    df.columns = ["timestamp", "vm_id", "avg_cpu"]
    df["min_cpu"] = df["avg_cpu"] / 2
    df["max_cpu"] = df["avg_cpu"] + max_extra
    return df


# ------------------------------ variance structure ------------------------------ #


def test_independent_vms_are_fully_diversified():
    n, N = 400, 300
    levels = RNG.uniform(5, 60, N)
    x = matrix(levels + RNG.normal(0, 2, (n, N)))
    v = eda.variance_structure(x)
    assert v["variance_ratio_vs_independent"] == pytest.approx(1.0, abs=0.25)
    assert abs(v["mean_pairwise_corr"]) < 0.01
    assert v["effective_independent_vms"] == pytest.approx(N, rel=0.2)
    assert v["share_between_vm"] + v["share_within_vm"] == pytest.approx(1.0)
    assert v["share_between_vm"] > 0.9  # levels differ far more than the noise


def test_common_factor_collapses_effective_independent_vms():
    n, N = 300, 200
    common = RNG.normal(0, 3, n)[:, None]
    x = matrix(20 + common + RNG.normal(0, 0.1, (n, N)))
    v = eda.variance_structure(x)
    assert v["mean_pairwise_corr"] > 0.95
    assert v["effective_independent_vms"] < 2
    assert v["variance_ratio_vs_independent"] > 0.5 * N


def test_one_dominant_volatile_vm_takes_the_variance_share():
    n, N = 300, 200
    base = 20 + RNG.normal(0, 0.1, (n, N))
    base[:, 7] += RNG.normal(0, 30, n)  # a single wild VM
    v = eda.variance_structure(matrix(base))
    assert v["top_share_in_sample"]["top_0.1pct"]["n_vms"] == 1
    assert v["top_share_in_sample"]["top_0.1pct"]["variance_share"] > 0.8


# ------------------------------ temporal summary ------------------------------ #


def test_white_noise_has_no_autocorrelation_or_mean_reversion_signal():
    t = eda.temporal_summary(RNG.normal(0, 1, 2000))
    assert abs(t["acf"][1]) < 0.06 and abs(t["ar1_phi"]) < 0.06
    assert t["first_lag_acf_below_0_2"] == 1
    assert t["trend_r2"] < 0.01


def test_ar1_parameter_and_half_life_are_recovered():
    phi, n = 0.8, 5000
    y = np.zeros(n)
    e = RNG.normal(0, 1, n)
    for i in range(1, n):
        y[i] = phi * y[i - 1] + e[i]
    t = eda.temporal_summary(y)
    assert t["ar1_phi"] == pytest.approx(phi, abs=0.04)
    assert t["ar1_half_life_steps"] == pytest.approx(np.log(0.5) / np.log(phi), rel=0.25)
    assert t["acf"][1] == pytest.approx(phi, abs=0.04)


def test_linear_trend_and_sinusoid_period_are_detected():
    n = 200
    t = eda.temporal_summary(0.01 * np.arange(n))
    assert t["trend_per_step"] == pytest.approx(0.01) and t["trend_r2"] == pytest.approx(1.0)
    assert t["power_share_two_slowest_frequencies"] == 0.0  # no residual power, and no 0/0
    assert t["trend_over_window"] == pytest.approx(0.01 * n)

    s = eda.temporal_summary(np.sin(2 * np.pi * np.arange(n) / 20))
    assert s["periodogram_top_periods_steps"][0] == pytest.approx(20, abs=1.0)
    assert s["periodogram_top_power_share"][0] > 0.9


def test_acf_length_is_capped_by_series_length():
    t = eda.temporal_summary(RNG.normal(0, 1, 30), max_lag=48)
    assert len(t["acf"]) == 30  # lags 0..29


# ------------------------------ peaks and lows ------------------------------ #


def test_peak_episode_and_runs_are_located():
    y = np.full(40, 10.0)
    y[10:15] = 20.0  # one contiguous high episode
    y[30] = 0.0  # one isolated low point
    p = eda.peak_low_summary(y, hi_q=0.95, lo_q=0.01)
    assert p["peak_runs"] == [{"start": 10, "end": 14, "length": 5}]
    assert p["peak_episode"] == [10, 14]
    assert p["low_runs"] == [{"start": 30, "end": 30, "length": 1}]
    assert p["n_low_runs_of_length_1"] == 1
    assert p["argmax"] in range(10, 15) and p["argmin"] == 30
    assert p["range_pct_of_mean"] == pytest.approx(100 * 20 / y.mean())


# ------------------------------ plumbing, coverage, lifetimes ------------------------------ #


def test_restrict_to_window_is_inclusive_and_does_not_mutate():
    x = matrix(RNG.uniform(0, 50, (10, 3)))
    rows = long_rows(x)
    cut = eda.restrict_to_window(rows, 4 * STEP_SECONDS)
    assert cut["timestamp"].max() == 4 * STEP_SECONDS and len(cut) == 5 * 3
    assert len(rows) == 30


def test_to_matrix_roundtrips_the_long_format():
    x = matrix(RNG.uniform(0, 50, (6, 4)))
    pd.testing.assert_frame_equal(eda.to_matrix(long_rows(x)), x, check_names=False)


def test_coverage_summary_counts_full_and_partial_vms():
    x = matrix(RNG.uniform(0, 50, (8, 5)))
    rows = long_rows(x)
    rows = rows[~((rows["vm_id"] == "vm0") & (rows["timestamp"] >= 6 * STEP_SECONDS))]  # vm0 is partial
    series = pd.DataFrame({"is_gap": False}, index=x.index)
    c = eda.coverage_summary(series, rows)
    assert c["observed_step_seconds"] == [300]
    assert c["n_vms"] == 5 and c["n_vms_full_coverage"] == 4
    assert c["readings_per_vm_min"] == 6 and c["readings_per_vm_max"] == 8
    assert c["window_hours"] == pytest.approx(8 * 300 / 3600)


def test_lifetime_summary_describes_censoring():
    meta = pd.DataFrame(
        {
            "vm_id": list("abcde"),
            "timestamp_vm_created": 0,
            "timestamp_vm_deleted": [1000, 1000, 1000, 400, 700],
        }
    )
    lt = eda.lifetime_summary(meta, window_end=300)
    assert lt["share_created_at_trace_start"] == 1.0
    assert lt["share_deleted_at_trace_end"] == pytest.approx(0.6)
    assert lt["n_deleted_before_trace_end"] == 2 and lt["n_deleted_within_window"] == 0
    assert lt["earliest_deletion"] == 400
    assert lt["seconds_from_window_end_to_earliest_deletion"] == 100
    assert len(lt["early_deleted_lifetime_days_p10_p50_p90"]) == 3


# ------------------------------ distributions and load ------------------------------ #


def test_distribution_summary_on_known_values():
    rows = pd.DataFrame({"avg_cpu": np.arange(100, dtype=float)})
    d = eda.distribution_summary(rows)
    assert d["mean"] == pytest.approx(49.5)
    assert d["quantiles"]["p50"] == pytest.approx(49.5)
    assert d["share_ge_50"] == pytest.approx(0.5) and d["share_lt_5"] == pytest.approx(0.05)
    assert d["share_ge_80"] == pytest.approx(0.2) and d["share_lt_1"] == pytest.approx(0.01)


def test_load_concentration_of_one_dominant_vm():
    vals = np.ones((5, 100))
    vals[:, 0] = 100.0  # one VM carries 100 of 199 units
    c = eda.load_concentration(matrix(vals), fracs=(0.01, 0.5))
    assert c["top_1pct"]["n_vms"] == 1 and c["top_1pct"]["load_share"] == pytest.approx(100 / 199)
    assert c["top_50pct"]["load_share"] == pytest.approx((100 + 49) / 199)


def test_burstiness_counts_hidden_spikes_exactly():
    rows = pd.DataFrame(
        {
            "avg_cpu": [10.0, 10.0, 50.0, 95.0],
            "max_cpu": [95.0, 20.0, 60.0, 99.0],
        }
    )
    b = eda.burstiness(rows)
    assert b["n_intervals_hidden_spike"] == 1  # avg<20 and max>=90: only the first row
    assert b["share_intervals_max_ge_95"] == pytest.approx(0.5)
    assert b["mean_max_minus_avg_pp"] == pytest.approx((85 + 10 + 10 + 4) / 4)


# ------------------------------ segments ------------------------------ #


def segment_fixture():
    n, per = 60, 30
    a = 10 + RNG.normal(0, 0.05, (n, per))
    b = 30 + RNG.normal(0, 0.05, (n, per))
    a[20:30] += 5.0  # segment A has an episode
    b[:, :] -= np.linspace(0, 3, n)[:, None]  # segment B drifts down by 3 pp
    x = matrix(np.hstack([a, b]))
    meta = pd.DataFrame({"vm_id": x.columns, "grp": ["A"] * per + ["B"] * per})
    return x, meta


def test_segment_table_levels_shares_episode_and_trend():
    x, meta = segment_fixture()
    t = eda.segment_table(x, meta, "grp", episode=(20, 29)).set_index("segment")
    assert t["share_vms_pct"].sum() == pytest.approx(100)
    assert t.loc["A", "mean_cpu"] < t.loc["B", "mean_cpu"]
    assert t["share_of_level_pct"].sum() == pytest.approx(100, abs=0.5)
    assert t.loc["A", "episode_amp_pp"] == pytest.approx(5.0, abs=0.2)
    assert t.loc["B", "trend_pp_over_window"] == pytest.approx(-3.0 * 60 / 59, abs=0.15)
    # weighted contribution = trend * share of VMs
    assert t.loc["B", "trend_contribution_to_aggregate_pp"] == pytest.approx(
        t.loc["B", "trend_pp_over_window"] * 0.5
    )


def test_segment_series_returns_one_column_per_level():
    x, meta = segment_fixture()
    s = eda.segment_series(x, meta, "grp")
    assert list(s.columns) == ["A", "B"] and len(s) == len(x)


def test_tenant_diversity_counts_and_shares():
    meta = pd.DataFrame(
        {
            "vm_id": [f"v{i}" for i in range(10)],
            "subscription_id": ["s1"] * 5 + ["s2"] * 3 + ["s3"] * 2,
            "deployment_id": [f"d{i}" for i in range(10)],
        }
    )
    t = eda.tenant_diversity(meta, meta["vm_id"])
    assert t["subscriptions"]["n"] == 3 and t["subscriptions"]["largest_share_pct"] == pytest.approx(50)
    assert t["deployments"]["n"] == 10 and t["deployments"]["largest_share_pct"] == pytest.approx(10)


# ------------------------------ predictability of structure ------------------------------ #


def test_persistent_volatility_is_recognised_and_random_volatility_is_not():
    n, N = 120, 400
    sig = RNG.uniform(0.1, 5, N)
    persistent = matrix(RNG.normal(0, 1, (n, N)) * sig)
    vp = eda.volatility_persistence(persistent)
    assert vp["spearman_volatility"] > 0.8
    assert vp["cohort_top_overlap_share"] > 5 * vp["cohort_overlap_if_random"]

    # volatility drawn afresh in each half: nothing to learn from history
    a = RNG.normal(0, 1, (n // 2, N)) * RNG.uniform(0.1, 5, N)
    b = RNG.normal(0, 1, (n // 2, N)) * RNG.uniform(0.1, 5, N)
    vq = eda.volatility_persistence(matrix(np.vstack([a, b])))
    assert abs(vq["spearman_volatility"]) < 0.15


def test_volatility_persistence_level_and_parts_add_up():
    n, N = 80, 100
    lev = RNG.uniform(5, 50, N)
    x = matrix(lev + RNG.normal(0, 1, (n, N)))
    vp = eda.volatility_persistence(x, split_at=40, cohort_frac=0.1)
    assert vp["spearman_level"] > 0.95 and vp["cohort_n_vms"] == 10
    assert 0 < vp["cohort_share_of_level_pct"] < 100


def test_volatility_persistence_needs_enough_observations():
    with pytest.raises(ValueError, match="at least 3"):
        eda.volatility_persistence(matrix(RNG.normal(0, 1, (4, 10))))


# ------------------------------ operational framing ------------------------------ #


def test_candidate_targets_cv_matches_definition():
    avg = matrix(RNG.uniform(0, 100, (50, 40)))
    mx = avg.clip(lower=0) * 0 + 99.0
    t = eda.candidate_targets(avg, mx).set_index("target")
    assert len(t) == 7
    s = avg.mean(axis=1)
    assert t.loc["mean_avg_cpu (required)", "cv_pct"] == pytest.approx(100 * s.std() / s.mean())
    assert t.loc["share_vms_max_cpu_ge_95_pct", "mean"] == pytest.approx(100.0)


def test_metadata_representativeness_is_perfect_when_lifetime_equals_window():
    x = matrix(RNG.uniform(1, 60, (30, 25)))
    meta = pd.DataFrame({"vm_id": x.columns, "avg_cpu": x.mean().to_numpy(), "max_cpu": 99.0})
    r = eda.metadata_representativeness(x, meta)
    assert r["pearson_window_vs_lifetime"] == pytest.approx(1.0)
    assert r["spearman_window_vs_lifetime"] == pytest.approx(1.0)
    assert r["lifetime_columns_present"] == ["avg_cpu", "max_cpu"]


def test_constant_target_has_undefined_not_misleading_dynamics():
    avg = matrix(RNG.uniform(1, 50, (30, 20)))
    mx = avg * 0 + 50.0  # max_cpu is constant: the share >= 95 is a constant 0
    t = eda.candidate_targets(avg, mx).set_index("target")
    assert t.loc["share_vms_max_cpu_ge_95_pct", "std"] == 0.0
    assert np.isnan(t.loc["share_vms_max_cpu_ge_95_pct", "acf1"])
    assert np.isnan(t.loc["share_vms_max_cpu_ge_95_pct", "cv_pct"])  # mean is 0: CV undefined


def test_constant_series_does_not_produce_nan_shares():
    t = eda.temporal_summary(np.full(50, 7.0))
    assert t["power_share_two_slowest_frequencies"] == 0.0 and t["acf1_of_first_difference"] is None


def test_constant_lifetime_column_gives_nan_correlation_not_a_warning():
    x = matrix(RNG.uniform(1, 60, (30, 25)))
    meta = pd.DataFrame({"vm_id": x.columns, "avg_cpu": 20.0})
    r = eda.metadata_representativeness(x, meta)
    assert np.isnan(r["pearson_window_vs_lifetime"]) and np.isnan(r["spearman_window_vs_lifetime"])


def test_spearman_is_pearson_of_ranks_and_needs_no_scipy():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    b = np.array([1.0, 4.0, 9.0, 16.0, 25.0, 36.0])  # monotone but not linear
    assert eda._corr(a, b, "spearman") == pytest.approx(1.0)
    assert eda._corr(a, b, "pearson") < 1.0
    ties = eda._corr([1, 1, 2, 3], [4, 4, 5, 6], "spearman")  # average ranks handle ties
    assert ties == pytest.approx(1.0)
    with pytest.raises(ValueError, match="unsupported"):
        eda._corr(a, b, "kendall")
