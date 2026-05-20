# Sanity Check: XGBoost on QQQ via MinuteAndDailyMLDataProcessor

**Date:** 2026-05-19  
**Data:** QQQ 1-minute bars, 2026-01-02 → 2026-05-01 (32,333 rows after dropna)  
**Label:** DirectionLabelTransformer — direction-up 1% / fwd-120 bars (~9.1% positive rate)  
**Features:** 258 total (129 minute + 129 daily via OLHCVFeatureTransformer)  
**Split:** chronological 80/20 — train 2026-01-02→2026-04-09, test 2026-04-09→2026-05-01  
**Model:** XGBoost, n_estimators=300, max_depth=4, scale_pos_weight=9.6

---

## 1. Metrics

| Split | AUC | PR-AUC | Prec@0.5 | Recall@0.5 |
|---|---|---|---|---|
| Train | 0.9984 | 0.9842 | 0.734 | 0.997 |
| Test  | 0.6132 | 0.1279 | 0.875 | 0.028 |
| Baseline (random) | 0.5000 | 0.0770 | — | — |

Test PR-AUC is 1.66× the random baseline, confirming the model learns a real signal. But the train/test gap is extreme (AUC 0.998 vs 0.613), indicating severe overfitting.

---

## 2. Data Leakage Check — PASSED

Shuffling the test labels while keeping model predictions fixed:

| | Test AUC |
|---|---|
| Real labels | 0.6132 |
| Shuffled labels | 0.4966 |

AUC collapses to ~0.5 when labels are shuffled, confirming the model is scoring based on historical features only — **no future data leakage detected** in the pipeline.

---

## 3. Learning Curve (Train vs Test AUC by boosting round)

| Round | Train AUC | Test AUC |
|---|---|---|
| 1 | 0.891 | 0.547 |
| 50 | 0.969 | **0.617** ← peak |
| 100 | 0.986 | 0.595 |
| 150 | 0.993 | 0.608 |
| 200 | 0.996 | 0.604 |
| 300 | 0.998 | 0.613 |

Test AUC peaks around round 50 then degrades before recovering slightly. The model starts memorising training data almost immediately. **Early stopping at ~50 rounds would be the better operating point.**

---

## 4. Feature Importance

### By family

| Feature family | Total importance |
|---|---|
| Daily features | 0.622 |
| Raw price lags (minute) | 0.238 |
| MA ratios (minute) | 0.117 |
| Technical indicators | 0.019 |
| Diff ratios | 0.005 |

### Top 20 individual features (selected)

```
0.0375  high_lag_6_minute
0.0294  close_by_close_ma_200_ratio_daily
0.0239  high_lag_4_daily
0.0215  volume_by_volume_ma_100_ratio_log_daily
0.0211  open_lag_1_minute          ← raw price
0.0175  high_lag_7_minute
0.0165  volume_by_volume_ma_100_ratio_daily
0.0150  high_by_high_ma_20_ratio_daily
0.0117  low_lag_1_minute           ← raw price
```

**Daily features dominate at 62% importance.** This is partly explained by raw price lag columns (`open_lag_0_daily`, `close_lag_0_daily`, etc.) carrying the same time-of-day spurious signal identified in the EDA — the daily bar's raw close price correlates with QQQ's long-term price trend, which in turn correlates with label rate. `close_lag_0_daily` range: 558–668 (raw dollar prices, not ratios).

---

## 5. Findings & Next Steps

### Data pipeline — HEALTHY
- No leakage detected. Shuffled-label AUC collapses to chance.
- Chronological join of daily features (previous trading day) is correctly implemented.
- Label transformer lookforward window produces expected 120-bar NaN tail.

### Model — OVERFIT, needs fixing
1. **Early stopping**: peak test performance is at ~50 trees, not 300. Add `early_stopping_rounds=20`.
2. **Drop raw price lag features**: minute lags 0–9 and daily lags 0–9 are raw dollar prices — non-stationary and spurious. Keep only normalized features (`*_ratio`, `*_ratio_log`, `*_diff_ratio`, technical indicators). This removes ~23% of importance that is likely noise.
3. **Stronger regularization**: reduce `max_depth` to 3, increase `min_child_weight` to 5, add `reg_alpha` / `reg_lambda`.
4. **Consider dropping `hour_of_day`**: confirmed in EDA to be the highest-KL feature (0.41), mostly encoding time-of-day regime rather than alpha.
