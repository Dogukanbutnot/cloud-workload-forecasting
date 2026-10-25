import pytest

from cwfc.splits import TrainHoldoutSplit


@pytest.fixture
def never_reveal(monkeypatch):
    """Make any call to ``reveal_holdout`` fail the test."""

    def boom(self):
        raise AssertionError("holdout values were revealed during modelling/validation")

    monkeypatch.setattr(TrainHoldoutSplit, "reveal_holdout", boom)
