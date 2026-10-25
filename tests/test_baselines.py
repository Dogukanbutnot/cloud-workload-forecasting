import numpy as np
import pandas as pd
import pytest

from cwfc.baselines import (
    mean_forecast,
    moving_average_forecast,
    persistence_forecast,
    seasonal_naive_forecast,
)


def frame(values):
    return pd.DataFrame({"y": values}, index=pd.Index(np.arange(len(values)) * 300, name="timestamp"))


def template(n, start=10_000):
    return pd.DataFrame(index=pd.Index(start + np.arange(n) * 300, name="timestamp"))


def test_moving_average_uses_only_the_last_window_values():
    out = moving_average_forecast(frame([1.0, 2.0, 3.0, 4.0, 10.0]), template(3), "y", window=3)
    assert out.tolist() == pytest.approx([17 / 3] * 3)
    assert list(out.index) == list(template(3).index)


def test_moving_average_window_larger_than_history_uses_everything():
    out = moving_average_forecast(frame([2.0, 4.0]), template(2), "y", window=50)
    assert out.tolist() == [3.0, 3.0]


def test_moving_average_ignores_nan_gaps_but_needs_some_observation():
    out = moving_average_forecast(frame([1.0, np.nan, 3.0]), template(1), "y", window=3)
    assert out.iloc[0] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="no observed values"):
        moving_average_forecast(frame([np.nan, np.nan]), template(1), "y", window=2)


def test_history_mean_is_the_mean_of_all_history():
    out = mean_forecast(frame([1.0, 2.0, 6.0]), template(2), "y")
    assert out.tolist() == [3.0, 3.0]


@pytest.mark.parametrize("fn,kw", [(moving_average_forecast, {"window": 2}), (mean_forecast, {})])
def test_empty_history_raises(fn, kw):
    with pytest.raises(ValueError, match="empty"):
        fn(frame([]), template(1), "y", **kw)


def test_invalid_window_raises():
    with pytest.raises(ValueError, match="positive"):
        moving_average_forecast(frame([1.0]), template(1), "y", window=0)


def test_forecast_depends_only_on_history_not_on_template_contents():
    hist = frame([1.0, 2.0, 3.0])
    plain = template(2)
    loaded = template(2).assign(y=[999.0, -999.0])  # even if a caller leaked target values here
    for fn, kw in [(moving_average_forecast, {"window": 2}), (mean_forecast, {}), (persistence_forecast, {})]:
        a = fn(hist, plain, "y", **kw)
        b = fn(hist, loaded, "y", **kw)
        assert a.tolist() == b.tolist()


def test_starter_baselines_behave_as_relied_upon():
    hist = frame([1.0, 2.0, 3.0, 4.0])
    assert persistence_forecast(hist, template(3), "y").tolist() == [4.0, 4.0, 4.0]
    # season of 2 repeats the last full season: [3, 4, 3, 4, 3]
    assert seasonal_naive_forecast(hist, template(5), "y", season_length=2).tolist() == [3.0, 4.0, 3.0, 4.0, 3.0]
    with pytest.raises(ValueError, match="not enough history"):
        seasonal_naive_forecast(hist, template(1), "y", season_length=5)
