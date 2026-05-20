---
name: prep-ml-data
description: Transform bar data into ML-ready features and labels and summarize basic insights. Use when any skill or user request needs a feature/label dataset as input to model training or evaluation.
---

# prep-ml-data

Transform OHLCV bar data into an ML-ready dataset with features and labels, cache the result to `data/ml/`, and summarize basic insights. This skill is a building block — other skills (research, evaluate) delegate here when they need a prepared dataset.

## Inputs

Parse these from the user's request:

| Parameter | Description |
|---|---|
| `symbol` | Ticker string or list |
| `start` / `end` | Date range `YYYY-MM-DD` |
| `features` | Natural language description of features to compute (e.g. "returns, volume ratio, rolling volatility") |
| `label` | Natural language description of the label (e.g. "direction up with 1% threshold over next 120 bars") |

If the user doesn't specify features or label, ask before proceeding.

## Preferred approach: ML data processor classes

Prefer these over assembling the pipeline manually. They handle bar fetching, flattening, feature/label transforming, and joining minute + daily features in one call. Live in `utils/ml_data_processor.py`.

**`MinuteAndDailyMLDataProcessor`** — single symbol. Fetches minute bars (for labels + short-term features) and daily bars (for longer-horizon features joined on the previous trading day).

```python
from utils.ml_data_processor import MinuteAndDailyMLDataProcessor

proc = MinuteAndDailyMLDataProcessor(
    symbol='QQQ',
    bar_fetcher=None,                  # defaults to AlpacaBarFetcher
    label_transformer=None,            # defaults to DirectionLabelTransformer()
    minute_feature_transformer=None,   # defaults to OLHCVFeatureTransformer()
    daily_feature_transformer=None,    # defaults to OLHCVFeatureTransformer()
    generate_labels=True,              # set False for inference
)
df = proc.run('2026-01-01', '2026-05-01')

# Access column names via properties
label_col   = proc.label_column     # e.g. 'label'
feature_cols = proc.feature_columns  # list of all minute + daily feature col names
```

Output columns use `_minute` / `_daily` suffixes to distinguish granularity. The `datetime` column is always present and is the join key.

**`MultipleSymbolMinuteAndDailyMLDataProcessor`** — multiple symbols, joined on `datetime`.

```python
from utils.ml_data_processor import MinuteAndDailyMLDataProcessor, MultipleSymbolMinuteAndDailyMLDataProcessor

multi = MultipleSymbolMinuteAndDailyMLDataProcessor(
    symbols=['QQQ', 'SQQQ'],
    list_of_ml_data_processors=[
        MinuteAndDailyMLDataProcessor('QQQ'),
        MinuteAndDailyMLDataProcessor('SQQQ'),
    ],
)
df = multi.run('2026-01-01', '2026-05-01')

label_col    = multi.label_column
feature_cols = multi.feature_columns
```

## Available label transformers

Use these directly when building a custom processor or working outside the processor classes. Both live in `utils/label_transformer.py`.

**`DirectionLabelTransformer`** — binary label: within the next `lookforward_period` bars, did the max gain reach `up_delta` AND the max loss stay below `down_delta`? This is a simultaneous max/min check, not path-dependent.
```python
from utils.label_transformer import DirectionLabelTransformer
transformer = DirectionLabelTransformer(
    price_column='close',
    label_column='label',
    positive_label='up',       # 'up' or 'down'
    up_delta=0.01,             # 1% max-gain threshold
    down_delta=0.01,           # 1% max-loss exclusion threshold
    lookforward_period=120,    # bars ahead to look
)
df = transformer.transform(df)
```
Recommended baseline: `up_delta=down_delta=0.01`, `lookforward_period=120` → ~9% positive rate on QQQ 1-min bars. Avoid `lookforward_period=30` at 1% (positive rate ~1.4%, too sparse).

**`VolatilityLabelTransformer`** — binary label: within the next `lookforward_period` bars, did the max gain OR max loss reach `change_delta`? Fires ~2× more often than direction at the same parameters.
```python
from utils.label_transformer import VolatilityLabelTransformer
transformer = VolatilityLabelTransformer(
    price_column='close',
    label_column='label',
    change_delta=0.01,
    lookforward_period=120,
)
df = transformer.transform(df)
```
At ≥1% threshold, volatility ≈ direction-up OR direction-down with negligible whipsaw contamination. Use volatility when you care about any large move; use direction when you need a directional signal.

## Available feature transformers

Use these directly when building a custom processor. Lives in `utils/feature_transformer.py`.

**`OLHCVFeatureTransformer`** — derives a rich feature set from OHLCV columns. Requires a `datetime` column (not index); the processor classes handle this automatically via `_flatten()`.

Features produced:
- **Lag features** — raw OHLCV values at lags 0–9
- **Bar-over-bar change** — diff ratio and log-ratio for each OHLCV column
- **Moving-average ratios** — price/volume relative to MA windows {5, 10, 20, 50, 100, 200}, both ratio and log-ratio
- **Technical indicators** — RSI (lengths 14/20/30/50/100/200), MACD line/histogram/signal
- **Time features** — `hour_of_day` and `day_of_week`

```python
from utils.feature_transformer import OLHCVFeatureTransformer

# Flatten MultiIndex bar df first
df = df.reset_index().rename(columns={'timestamp': 'datetime'})

transformer = OLHCVFeatureTransformer()
df = transformer.transform(df)
feature_cols = transformer.feature_columns  # list of added feature column names
```

## Steps (when using the processor classes)

1. **Instantiate and run** the appropriate processor class above.

2. **Drop rows with NaNs** introduced by rolling windows or the lookforward shift:
   ```python
   df = df.dropna(subset=proc.feature_columns + [proc.label_column])
   ```

3. **Save three files to `data/ml/`**:
   ```python
   import os
   base = f'data/ml/{symbol}_{start}_{end}_minute'
   os.makedirs('data/ml', exist_ok=True)

   features_path = f'{base}_features.parquet'
   labels_path   = f'{base}_{label_description}_labels.parquet'
   joined_path   = f'{base}_{label_description}_joined.parquet'

   df[proc.feature_columns].to_parquet(features_path)
   df[[proc.label_column]].to_parquet(labels_path)
   df[proc.feature_columns + [proc.label_column]].to_parquet(joined_path)
   ```

4. **Summarize and report**:
   - Paths written: features, labels, joined
   - Shape of each file
   - Label distribution (value counts + positive rate)
   - NaN rows dropped (before vs after dropna)
   - When called by another skill, return all three paths: `features_path`, `labels_path`, `joined_path`

## Notes

- `data/ml/` is gitignored — cache files are local only.
- The processor extends the fetch range automatically: ±1 day for minute bars (covers rolling warmup and lookforward tail), −365 days for daily bars (covers MA-200 warmup). The output is filtered back to the requested `start`/`end`.
- Daily features are joined on the *previous* trading day to avoid same-day data leakage.
- Output columns: minute features end in `_minute`, daily features in `_daily`. Use `proc.feature_columns` to get the full list rather than inferring from column names.
- For label column name, use `proc.label_column` rather than hardcoding `'label'`.
