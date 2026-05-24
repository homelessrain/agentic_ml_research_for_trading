---
name: evaluate
description: Evaluate ML models.
---

## Methodology

### Evaluation Mode

#### Quick One-Cycle Evaluation

Use `scripts/train_and_test.py` to run only one cycle of training and testing.

#### Rolling Evaluation

Use `scripts/rolling_eval.py` to train/test a model using various time windows that represent different regimes.

### Metrics

#### Threshold-Agnostic Metrics

For binary classification, some metrics (e.g. accuracy, precision and recall) can be affected by the decision threshold we use to decide if a prediction is positive. Therefore, it is helpful to look at threshold-agnostic metrics, as eventually we will likely use the probability output in trading algos

* ROC AUC and PR AUC
* **Always report calibration curves + correlation calculated using calibration points**
* Log loss

#### Interpreting Training, Validation, Testing Metrics

* Validation by default is just random split, which means randomly sampled from the same period of training. Having parity between training and validation performance means the model is able to generalize to the holdout data that was not seen during training

* Testing by default is walk-forward testing, which means testing uses future data w.r.t training/validation time period. Financial ML problems usually see gaps between training and testing performance, due to regime shift and the inability to generalize to future data. Therefore, understanding where training and testing performance differ and how they differ is critical to improve ML models for trading purpose.


## Execution
Run using the venv Python (`env/bin/python3`). Always run from the project root so imports resolve correctly.

**Standard train + test** (reuse the existing script):
```bash
env/bin/python3 scripts/train_and_test.py \
  --model MyModel \
  --train-start 2026-01-01 --train-end 2026-03-31 \
  --test-start  2026-04-01 --test-end  2026-04-30 \
  --param key=value
```

**Custom script**:
```bash
env/bin/python3 scripts/agent/<experiment_name>.py
```

Capture stdout/stderr. If the run fails, diagnose and fix before writing the report — do not report on a failed experiment.

Suppress `PerformanceWarning` noise with `-W ignore` if needed.
