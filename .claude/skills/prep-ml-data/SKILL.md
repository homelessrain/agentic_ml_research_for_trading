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
| `symbol` | Ticker string or list (passed through to `fetch-bars` if bar data isn't already loaded) |
| `start` / `end` | Date range `YYYY-MM-DD` (passed through to `fetch-bars` if needed) |
| `timeframe` | Bar timeframe: `minute` / `hour` / `day` (default: `day`) |
| `source` | Bar data source: `alpaca` or `yahoo` (default: `alpaca`) |
| `features` | Natural language description of features to compute (e.g. "returns, volume ratio, rolling volatility") |
| `label` | Natural language description of the label (e.g. "direction up with 1% threshold over next 5 bars") |

If the user doesn't specify features or label, ask before proceeding.

## Available label transformers

Prefer these over writing label logic from scratch. Both live in `utils/label_transformer.py`.

**`DirectionLabelTransformer`** — binary label: did price move up (or down) by at least `up_delta` within `lookforward_period` bars without first hitting `down_delta` in the opposite direction?
```python
from utils.label_transformer import DirectionLabelTransformer
transformer = DirectionLabelTransformer(
    price_column='close',
    label_column='label',
    positive_label='up',      # 'up' or 'down'
    up_delta=0.01,            # 1% move threshold
    down_delta=0.01,
    lookforward_period=5,     # bars ahead to look
)
df = transformer.transform(df)
```

**`VolatilityLabelTransformer`** — binary label: did price move at least `change_delta` in either direction within `lookforward_period` bars?
```python
from utils.label_transformer import VolatilityLabelTransformer
transformer = VolatilityLabelTransformer(
    price_column='close',
    label_column='label',
    change_delta=0.01,
    lookforward_period=5,
)
df = transformer.transform(df)
```

## Available feature transformers

Prefer this over writing feature logic from scratch. Lives in `utils/feature_transformer.py`.

**`OLHCVFeatureTransformer`** — derives a rich feature set from OHLCV columns. Requires a `datetime` column (not index), so reset the index first if the bar data uses a DatetimeIndex.

Features produced:
- **Lag features** — raw OHLCV values at lags 0–9 (10 bars back)
- **Bar-over-bar change** — diff ratio and log-ratio for each OHLCV column
- **Moving-average ratios** — price/volume relative to MA windows {5, 10, 20, 50, 100, 200}, both ratio and log-ratio
- **Technical indicators** — RSI, MACD, MACD signal, MACD histogram (on `close`)
- **Time features** — `hour_of_day` and `day_of_week` as categorical columns

```python
from utils.feature_transformer import OLHCVFeatureTransformer

# Reset index so 'timestamp' becomes a column, then rename to 'datetime'
df = df.reset_index().rename(columns={'timestamp': 'datetime'})

transformer = OLHCVFeatureTransformer()
df = transformer.transform(df)
```

`transform()` appends feature columns to the DataFrame in-place and returns it. It does not return the feature column list — track feature columns explicitly:
```python
feature_cols = [c for c in df.columns if c not in ['datetime', 'open', 'high', 'low', 'close', 'volume', 'label']]
```

If neither fits, write feature logic inline.

## Steps

1. **Load bar data** — check if the relevant parquet already exists in `data/bars/`, otherwise delegate to `/fetch-bars`:
   ```python
   import sys; sys.path.insert(0, '.')
   from utils.bar_fetcher import AlpacaBarFetcher, YahooBarFetcher

   fetcher = AlpacaBarFetcher() if source == 'alpaca' else YahooBarFetcher()
   df = fetcher.fetch_bars(symbol, start, end, timeframe=timeframe)
   # Work on a single symbol — drop the symbol level if needed
   df = df.droplevel('symbol') if isinstance(df.index, pd.MultiIndex) else df
   ```

2. **Apply label transformer** using the closest matching transformer above, or write inline.

3. **Apply feature transformer** using the closest matching transformer above, or write inline.

4. **Drop rows with NaNs** introduced by rolling windows or the lookforward shift:
   ```python
   df = df.dropna(subset=feature_cols + ['label'])
   ```

5. **Save three files to `data/ml/`**:
   ```python
   import os
   base = f'data/ml/{symbol}_{start}_{end}_{timeframe}'
   os.makedirs('data/ml', exist_ok=True)

   features_path = f'{base}_features.parquet'
   labels_path   = f'{base}_{label_description}_labels.parquet'
   joined_path   = f'{base}_{label_description}_joined.parquet'

   df[feature_cols].to_parquet(features_path)
   df[['label']].to_parquet(labels_path)
   df[feature_cols + ['label']].to_parquet(joined_path)
   ```
   All three share the same index so features and labels are always aligned.

6. **Summarize and report**:
   - Paths written: features, labels, joined
   - Shape of each file
   - Label distribution (value counts + positive rate)
   - Feature summary stats (mean, std, min, max per feature column)
   - NaN rows dropped (before vs after dropna)
   - When called by another skill, return all three paths: `features_path`, `labels_path`, `joined_path`

## Notes

- `data/ml/` is gitignored — cache files are local only.
- Keep features and labels on the same DataFrame so row alignment is preserved.
- For multi-symbol inputs, run the pipeline per symbol and concatenate with a symbol index level.
- The label transformers in `utils/label_transformer.py` currently have a string formatting bug in column names (f-string braces missing). Work around it by computing label logic inline if the transformer output looks wrong.
</thinking>
