# Cloud workload forecasting: submission

Forecasts the next two hours (24 five-minute steps) of two aggregate series built
from the Azure VM CPU readings in the `data-v1` release:

| target | method | holdout MAE |
|---|---|---|
| `active_vm_count` | constant 10,000, *proven* fixed by construction (report §2) | 0 |
| `aggregate_cpu_utilization` | damped persistence toward the history mean, one fitted parameter | 0.106 pp (persistence 0.147, history mean 0.175) |

The challenge statement is in [`ASSIGNMENT.md`](ASSIGNMENT.md), the file contract in
[`SUBMISSION.md`](SUBMISSION.md). The write-up is [`reports/final_report.md`](reports/final_report.md).

## Quick start: from a clean checkout to `predictions.csv`

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                       # numpy + pandas are the only runtime dependencies
```

Get the data (not committed to Git): download `data-v1.tar.gz` from
<https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1>, then

```bash
tar -xzf data-v1.tar.gz -C challenge_data/
(cd challenge_data && sha256sum -c SHA256SUMS)     # integrity check
```

Generate the submission (**the one command**):

```bash
python -m cwfc.predict --data challenge_data --output predictions.csv
```

It takes about 7 seconds, is deterministic (two runs give byte-identical files) and
validates the file against the `SUBMISSION.md` contract before writing it: exact
columns and order, 24 rows, holdout timestamps, finite values, `active_vm_count >= 0`,
CPU in `[0, 100]`, no index column. `--data` accepts the release directory or the
readings file itself. The pipeline derives everything (including the holdout
timestamps) from the data it is given, so it also runs on a small replacement
dataset with the same schema; where its assumptions cannot be verified it falls back
to persistence and says so in the printed summary (`active_method`, `cpu_method`).

## Tests

```bash
pip install -e ".[dev,eda]"            # pytest (+ matplotlib for the figure tests)
python -m pytest                       # 170+ tests, under 10 seconds
```

The tests include leakage guards: modelling code is run with `reveal_holdout()`
patched to raise, and a source scan enforces that only `evaluate_holdout.py` may call it.

## Reproducing the analysis behind the report

Steps 1-4 read only the training window (the first 110 of 134 observations).

```bash
python -m cwfc.build_targets    --data challenge_data --output-dir outputs/targets    # 1. targets, 110/24 split, population check
python -m cwfc.run_baselines    --data challenge_data --output-dir outputs/baselines  # 2. baselines, rolling origin
python -m cwfc.run_eda          --data challenge_data --output-dir outputs/eda        # 3. exploratory analysis, figures
python -m cwfc.run_model        --data challenge_data --output-dir outputs/model      # 4. model vs baselines, rolling origin
python -m cwfc.evaluate_holdout --data challenge_data --predictions predictions.csv \
        --output-dir outputs/holdout --confirm                                         # 5. ONE-SHOT: reveals the holdout
```

`outputs/` is git-ignored. Run step 5 only once the model is frozen: the `--confirm`
flag exists to make that deliberate.

## Layout

```
src/cwfc/
  targets.py        clean readings -> the two target series on the 5-minute grid (+ audit report)
  splits.py         110/24 chronological split; holdout values hidden behind reveal_holdout()
  population.py     is active_vm_count a constant? (training window + metadata, never holdout values)
  rolling.py        rolling-origin folds (positional)
  baselines.py      persistence, seasonal naive (starter) + moving average, history mean
  models.py         mean-reverting level model; damped persistence is its alpha = 1 case
  evaluation.py     rolling-origin harness and metrics (MAE, RMSE, sMAPE, bias, MASE, regret, ...)
  availability.py   columns that must not be features (lifetime statistics, future events)
  eda.py, run_eda.py, eda_plots.py     exploratory analysis
  policy.py         headroom margin from validation residuals
  predict.py        final fit + predictions.csv          <- the submission command
  submission.py     the predictions.csv contract as executable checks
  evaluate_holdout.py   one-shot holdout evaluation (the only place holdout values are read)
tests/              unit, integration and leakage tests (synthetic "smoke" dataset in tests/smoke.py)
reports/            final report and its figures
```

## Main modelling decisions (details and evidence in the report)

1. **Build the full target series first, then split.** The last 24 observations are the
   holdout; fitting, feature choice, tuning and model selection use the first 110 only.
2. **`active_vm_count` is not a forecasting problem here.** Only VMs with complete
   traces were released, so no VM can appear or disappear in the window. The code
   forecasts the constant only when constancy is proven from the training window plus
   VM metadata (creation before the window, deletion after the horizon); otherwise it
   carries the last count forward.
3. **A deliberately simple CPU model.** The aggregate has no trend, no identifiable
   cycle in 9 hours and a regime change inside the window, so with 110 points anything
   richer mostly fits noise. Damped persistence has one parameter (`phi`, estimated on
   the history) and is never far behind the best baseline in any validation fold.
4. **Validation mirrors production.** Rolling origin, expanding window, 24-step horizon,
   7 folds inside the training window, every fit nested (a fold's parameters see only
   its own history).
5. **No lifetime statistics as features.** `avg_cpu`, `max_cpu`, `p95_max_cpu` in the
   metadata cover the VM's whole 30-day life, i.e. they contain future information.
6. **Optional extensions were skipped** (extra targets, segment-level forecasting,
   notebooks) for time; see the report's limitations.
