---
name: research
description: Conduct a ML research iteration given a hypothesis or direction. Implements new model variants or feature changes, trains and tests them, and writes a structured report. The main entry point for agentic ML research.
---

# research

Run a focused ML research iteration: read past context, implement the idea, train + test, and write a structured report.

## Inputs

Parse these from the user's prompt:

| Parameter | Description | Default |
|---|---|---|
| `hypothesis` | The research direction or idea to explore (required) | — |
| `symbol` | Ticker (e.g. `QQQ`) | `QQQ` |
| `train_start` / `train_end` | Training date range | `2026-01-01` / `2026-03-31` |
| `test_start` / `test_end` | Test date range | `2026-04-01` / `2026-04-30` |

If `hypothesis` is missing or ambiguous, ask before proceeding.

---

## Step 1 — Read past context

Before implementing anything, read what has already been tried and learned.

1. Read all files in `recaps/` (research reflections and next-step suggestions).
2. Read recent reports in `reports/` and `reports/local/` — focus on the Findings & Next Steps sections.
3. Identify: what was tried, what worked, what didn't, and what was explicitly recommended next.

Use this context to:
- Avoid re-running experiments that are already documented.
- Ground the implementation in known findings (e.g., overfitting issues, leaky features, feature importance patterns).
- Prioritise the user's hypothesis and reconcile it with past learnings.

---

## Step 2 — Formulate a plan

Write a short plan (in your response, not to a file) covering:
- **Hypothesis**: what you expect and why, grounded in past findings.
- **What changes**: model class, features, label definition, hyperparameters, or data selection — be specific.
- **What stays the same**: reuse `MinuteAndDailyMLDataProcessor`, `OLHCVFeatureTransformer`, `BinaryClassificationEvaluator`, and `scripts/train_and_test.py` unless the hypothesis explicitly requires changing them.
- **Success criteria**: which metric(s) you will use to judge the result (typically test ROC-AUC and PR-AUC vs the baseline in the most recent report).

---

## Step 3 — Implement

### New model classes → `models/agent/`

All agent-generated model classes go under `models/agent/`. Name the file after the class in snake_case.

```
models/agent/<snake_case_model_name>.py
```

New models **must**:
- Inherit from `SingleAssetModel` (for single-symbol models) or `BaseModel` directly.
- Implement all abstract methods: `train`, `predict`, `test`, `_evaluate`.
- Accept `symbol: str` as the first `__init__` argument (required by `resolve_model_class` in `scripts/train_and_test.py`).
- Use `MinuteAndDailyMLDataProcessor` for data — do not bypass the data pipeline.
- Use `BinaryClassificationEvaluator` for metrics — do not invent new evaluation logic.

Minimal skeleton:

```python
from models.single_asset_model import SingleAssetModel
from utils.ml_data_processor import MinuteAndDailyMLDataProcessor
from utils.evaluator import BinaryClassificationEvaluator
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import cast

class MyModel(SingleAssetModel):
    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol)
        self._ml_data_processor = MinuteAndDailyMLDataProcessor(symbol)
        self._evaluator = BinaryClassificationEvaluator()
        self._model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=42,
            enable_categorical=True,
            **kwargs,
        )

    def train(self, start_datestr: str, end_datestr: str):
        full_df = self._ml_data_processor.run(start_datestr, end_datestr, drop_label_na=True)
        feature_cols = self._ml_data_processor.feature_columns
        label_col = self._ml_data_processor.label_column
        df = full_df.loc[:, feature_cols + [label_col]]

        train_df, val_df = cast(
            tuple[pd.DataFrame, pd.DataFrame],
            train_test_split(df, test_size=0.1, random_state=42),
        )
        X_train, y_train = train_df[feature_cols], train_df[label_col]
        X_val,   y_val   = val_df[feature_cols],   val_df[label_col]
        self._model.fit(X_train, y_train)

        train_metrics = self._evaluate(X_train, y_train)
        val_metrics   = self._evaluate(X_val,   y_val)
        return train_metrics, val_metrics

    def predict(self, start_datestr: str, end_datestr: str, drop_label_na: bool = True):
        full_df = self._ml_data_processor.run(start_datestr, end_datestr, drop_label_na=drop_label_na)
        feature_cols = self._ml_data_processor.feature_columns
        label_col    = self._ml_data_processor.label_column
        self._prediction_full_df    = full_df
        self._prediction_feature_cols = feature_cols
        self._prediction_label_col  = label_col
        X = full_df[feature_cols]
        y_pred    = self._model.predict(X)
        prob_pred = self._model.predict_proba(X)[:, 1]
        return y_pred, prob_pred

    def test(self, start_datestr: str, end_datestr: str) -> dict:
        y_pred, prob_pred = self.predict(start_datestr, end_datestr, drop_label_na=True)
        y_true = self._prediction_full_df[self._prediction_label_col]
        return self._evaluator.evaluate(y_true, y_pred, prob_pred)

    def _evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        y_pred    = self._model.predict(X)
        prob_pred = self._model.predict_proba(X)[:, 1]
        return self._evaluator.evaluate(y, y_pred, prob_pred)
```

### New scripts → `scripts/agent/`

If the experiment needs logic beyond what `scripts/train_and_test.py` covers (e.g. learning curves, feature ablation, walk-forward validation), add a script under `scripts/agent/`.

```
scripts/agent/<experiment_name>.py
```

Reuse `scripts/train_and_test.py` for the standard train + test flow — call it via `resolve_model_class`:

```python
from utils.script_utils import resolve_model_class, build_param_overrides
```

---

## Step 4 — Run the experiment

Delegate evaluation tasks to `/evaluate` skill whenever possible.

---

## Step 5 — Write the report

Create a new folder `reports/local/<experiment_name>_<YYYY-MM-DD>` (Use today's date):
    * Save the main report to `report.md`. 
    * Save other files/plots to the same folder.

```markdown
# <Experiment title>

**Date:** <YYYY-MM-DD>
**Hypothesis:** <one sentence>
**Data:** <symbol>, <train range> (train) / <test range> (test), <N rows after dropna>
**Label:** <transformer + params + positive rate>
**Features:** <feature set description, any additions/removals vs baseline>
**Model:** <class name + key hyperparameters>

---

## 1. Results

| Split      | ROC-AUC | PR-AUC | Positive-label rate |
|------------|---------|--------|---------------------|
| Train      |         |        |                     |
| Validation |         |        |                     |
| Test       |         |        |                     |
| Baseline*  | 0.5000  | <pos_rate> | <pos_rate>      |

*Random baseline: ROC-AUC = 0.5, PR-AUC = positive label rate.

---

## 2. Analysis

<Compare to the most recent prior result. Explain what changed and why metrics moved the way they did.>

---

## 3. Findings & Next Steps

<Bullet list of concrete findings and actionable suggestions for the next iteration.>
```

---

## Conventions and constraints

- **Do not** modify files under `utils/` or `models/` (non-agent). Only create new files under `models/agent/` and `scripts/agent/`.
- **Do not** change `DirectionModel` or `BaseModel` — extend them instead.
- **Reuse** `BinaryClassificationEvaluator` — do not add new metric computations inline.
- **Reuse** `MinuteAndDailyMLDataProcessor` — do not bypass the data pipeline.
- **Use chronological splits**, not random, for train/test — random splits inflate test metrics due to temporal autocorrelation in price data.
- **Compare against the most recent documented baseline** from `reports/` — always report delta.
- **Report positive label rate** alongside AUC — a model that predicts all-negative can achieve high accuracy on a 9% positive-rate label; ROC-AUC and PR-AUC are the primary metrics.
- `models/agent/` and `scripts/agent/` must have an `__init__.py` if not already present (required for `resolve_model_class` imports).
