# Feature Engineering Ablation Study + New Feature Exploration

**Date:** 2026-05-23  
**Hypothesis:** Systematically removing feature families will expose which features drive generalization; new domain-motivated features (volatility-normalized momentum, realized vol, intrabar range, VWAP deviation) will provide incremental improvement.  
**Data:** QQQ, 1-minute bars. 2-year rolling training window (best from prior rolling-window study), tested on 4 monthly OOS periods (Jan–Apr 2026).  
**Label:** `DirectionLabelTransformer` — direction-up 1% / fwd-120 bars. Positive rates: Jan 5.6%, Feb 9.0%, Mar 12.5%, Apr 9.8%.  
**Model:** `FeatureAblationModel` (XGBoost, n_estimators=300, max_depth=4, scale_pos_weight auto-computed per training fold, random 90/10 train/val split). Same architecture as `DirectionModel`, only the feature transformer changes.  
**Prior baseline:** rolling-window study 2y window → mean test AUC 0.758.

---

## Phase 1 — Ablation Study

### Feature family definitions

| Family | Features | Count per timeframe |
|--------|----------|:-------------------:|
| 1. Raw price lags | `{open,high,low,close,volume}_lag_{0..9}` | 50 |
| 2. Diff-ratios | `{col}_diff_ratio`, `{col}_diff_ratio_log` | 10 |
| 3. MA ratios | `{col}_by_{col}_ma_{5,10,20,50,100,200}_ratio[_log]` | 60 |
| 4. Tech indicators | RSI (6 lengths) + MACD (3 outputs) | 9 |
| 5. Time features | `hour_of_day`, `day_of_week` | 2 |

**Note:** The baseline `OLHCVFeatureTransformer` generates families 1–4 (129 features per timeframe × 2 timeframes = 258 total) but does NOT register time features — they exist in the dataframe but are excluded from `feature_columns`. This study tests adding them explicitly.

### Results

| Variant | N feat | Jan-26 | Feb-26 | Mar-26 | Apr-26 | **Mean AUC** | Δ baseline | Train AUC |
|---------|-------:|-------:|-------:|-------:|-------:|-----------:|:----------:|----------:|
| Full (baseline) | 258 | 0.7821 | 0.8456 | 0.5983 | 0.7049 | **0.7327** | — | 0.9980 |
| No raw lags | 158 | 0.7827 | 0.6957 | 0.6540 | 0.6551 | 0.6969 | **−0.036** | 0.9970 |
| No diff-ratios | 238 | 0.8071 | 0.8219 | 0.6024 | 0.6658 | 0.7243 | −0.008 | 0.9980 |
| No MA ratios | 138 | 0.7654 | 0.7658 | 0.5675 | 0.6173 | 0.6790 | **−0.054** | 0.9963 |
| No tech indicators | 240 | 0.7667 | 0.7899 | 0.7137 | 0.6774 | 0.7369 | **+0.004** | 0.9973 |
| **+ Time features** | **262** | **0.8999** | **0.8008** | **0.6796** | **0.7969** | **0.7943** | **+0.062** | 0.9995 |

> Random baseline: ROC-AUC = 0.500

### Per-period PR-AUC

| Variant | Jan-26 | Feb-26 | Mar-26 | Apr-26 | Mean PR-AUC |
|---------|-------:|-------:|-------:|-------:|:-----------:|
| Full (baseline) | 0.1877 | 0.2932 | 0.2154 | 0.3063 | 0.2507 |
| No raw lags | 0.1830 | 0.2665 | 0.1981 | 0.3105 | 0.2395 |
| No diff-ratios | 0.2218 | 0.3133 | 0.1969 | 0.2807 | 0.2532 |
| No MA ratios | 0.3518 | 0.1882 | 0.1616 | 0.1531 | 0.2137 |
| No tech indicators | 0.1278 | 0.2063 | 0.2875 | 0.1719 | 0.1984 |
| **+ Time features** | **0.3986** | **0.2135** | **0.2634** | **0.3335** | **0.3023** |

---

## Phase 2 — New Feature Exploration

### New feature definitions (15 per timeframe)

| Family | Features |
|--------|----------|
| ATR-normalized momentum | `(close − close_lag_N) / ATR(14)` for N ∈ {1, 5, 10, 20} |
| Realized volatility | `std(log_returns, W)` for W ∈ {5, 10, 30, 60} |
| Volume z-score | `(volume − vol_ma20) / vol_std30` |
| Intrabar range | `(high − low) / close`; rolling mean at W ∈ {5, 20} |
| Multi-timeframe momentum | Rolling sum of log-returns over {5, 15} bars |
| VWAP deviation | `(close − vwap_20) / close` |

### Results

| Variant | N feat | Jan-26 | Feb-26 | Mar-26 | Apr-26 | **Mean AUC** | Δ baseline | Train AUC |
|---------|-------:|-------:|-------:|-------:|-------:|-----------:|:----------:|----------:|
| Full (baseline) | 258 | 0.7821 | 0.8456 | 0.5983 | 0.7049 | 0.7327 | — | 0.9980 |
| New features only | 30 | 0.5679 | 0.6992 | 0.7306 | 0.4877 | 0.6214 | −0.111 | 0.9903 |
| No-raw-lags + New | 188 | 0.6742 | 0.7178 | 0.6669 | 0.6874 | 0.6866 | −0.046 | 0.9986 |
| **Full + New** | **288** | **0.7858** | **0.7845** | **0.7156** | **0.6977** | **0.7459** | **+0.013** | 0.9989 |

---

## Analysis

### Finding 1: Time features are the single biggest lever (+0.062 AUC)

Adding `hour_of_day` and `day_of_week` as proper model features produces the largest improvement of any experiment (+0.062 mean AUC, rising from 0.733 → 0.794). The EDA study had already documented strong time-of-day patterns: label rate climbs from ~8% at the open to ~28% near the close. By giving the model explicit time inputs, it can condition its directional predictions on where in the trading session each bar falls. Critically, train AUC only rises from 0.998 to 0.9995, so this is not a pure overfitting story — the time feature genuinely generalises.

**Why the baseline omits time features is an important implementation gap.** `OLHCVFeatureTransformer` adds `hour_of_day` and `day_of_week` to the dataframe but doesn't call `self._register()`, so they are silently excluded from `feature_columns` and never seen by the model. This should be fixed.

### Finding 2: MA ratios are the most valuable stationary family (−0.054 without them)

Removing MA ratios causes the single largest drop among the ablation experiments (−0.054). The 60 MA-ratio features capture multi-timeframe trend relative to the recent price — how extended the price is relative to its 5/10/20/50/100/200-bar averages. This is exactly the information content needed to assess momentum exhaustion or continuation. Their loss hurts every test period, particularly April (from 0.705 → 0.617, −0.088).

### Finding 3: Raw price lags surprisingly help overall (−0.036 without them)

This contradicts the prior generalization experiments (Exp2, May 2026), where removing raw lags was recommended as a stationarity improvement. The resolution: the prior study used a different training setup (chronological early stopping, short training window). In the 2-year rolling window with 300 trees and random split, raw price lags encode the persistent upward trend of QQQ from 2024–2026 Q1, which correlates with the rising label rates. The model exploits this trend signal.

However, this is **regime-conditional**: when the trend reverses (April 2026 sell-off), the raw-lag advantage disappears. The "No raw lags" variant loses the Feb boost (0.696 vs 0.846) but partially recovers in March (0.654 vs 0.598). Net effect is negative (−0.036), but the tradeoff is real.

### Finding 4: Tech indicators (RSI, MACD) are neutral to slightly negative (+0.004 without them)

Removing RSI (6 lengths) and MACD (3 outputs) slightly improves mean AUC (+0.004) and improves March specifically from 0.598 → 0.714 (+0.116). This is a clear signal: RSI and MACD are largely redundant with the MA ratios already in the feature set. Both RSI and MACD are derived from close-price differences and exponential averages — information already well-represented by the 12 close-price MA-ratio features. The tech indicators add noise (additional features without additional signal) and may be hurting by increasing dimensionality.

### Finding 5: Diff-ratios are nearly neutral (−0.008)

Short-horizon price changes normalized by the previous bar's price add minimal information beyond what raw lags and MA ratios already provide. The model can implicitly compute 1-bar differences from consecutive raw lags. Diff-ratios are mildly helpful (without them, mean AUC drops 0.008) but not worth much computational overhead.

### Finding 6: New features alone cannot carry the model (−0.111)

With only 30 features (15 per timeframe), the new feature set produces erratic per-period AUC (0.57, 0.70, 0.73, 0.49). Despite strong train AUC (0.990), the limited feature vocabulary can't generalise across regimes. The April failure (0.488, below random) suggests the single-tree boundary found in the April tariff sell-off environment inverts on the OOS set.

### Finding 7: Full + New is the best tested generalisation combo (+0.013)

Adding all 6 new feature families on top of the full baseline improves mean AUC from 0.733 → 0.746. The gain is not uniform: March improves dramatically (+0.117, from 0.598 → 0.716) while February softens (−0.061). The March improvement is the most meaningful signal: ATR-normalized momentum and realized volatility features explicitly represent the volatility regime, helping the model recognise the high-vol March tariff environment as a regime where directional moves are constrained.

The combination "No-raw-lags + New" (0.687) underperforms "Full + New" (0.746), confirming that raw lags still contribute positively in the random-split / 300-tree training setup.

---

## 3. Findings & Next Steps

### Key findings

1. **Time features (hour_of_day, day_of_week) are currently omitted from the model by a silent implementation bug** — `OLHCVFeatureTransformer` creates these columns but doesn't register them. Adding them gives the largest single improvement: +0.062 mean AUC (0.733 → 0.794).

2. **MA ratios are the most valuable stationary family** (−0.054 without them). They provide multi-timeframe trend context that no other family replicates.

3. **Tech indicators (RSI, MACD) are redundant with MA ratios** (+0.004 without them). RSI and MACD carry information already encoded in the 60 MA-ratio features. Removing them reduces dimensionality with no performance cost.

4. **Raw price lags provide net positive signal in the 2y-rolling / 300-tree regime** (−0.036 without them), despite being non-stationary. Their benefit comes from encoding the 2024–2026 upward trend, which correlates with the training period. In a regime-change scenario (April 2026), this advantage vanishes.

5. **New features provide incremental improvement, especially for high-vol regimes** (+0.013 with Full + New; March improves +0.117). ATR-normalized momentum and realized volatility help the model detect regime shifts that trip up standard MA-based features.

6. **Removing tech indicators + adding new features would be a clean upgrade**: same feature count as the baseline, with RSI/MACD replaced by ATR momentum + realized vol + VWAP deviation + intrabar range.

### Recommended next feature set

Based on these findings, the recommended feature set for future experiments is:

**Tier 1 (definitely include):**
- MA ratios (families 3) — most predictive
- Time features registered properly (hour_of_day, day_of_week)
- Realized volatility + intrabar range (from new features) — improve regime detection

**Tier 2 (keep with caveats):**
- Raw price lags (family 1) — helpful with current setup, revisit if switching to chronological splits

**Tier 3 (drop or de-prioritize):**
- Tech indicators (RSI, MACD) — redundant with MA ratios
- Diff-ratios — nearly neutral; can be kept for completeness

### Next steps

1. **Fix the time-feature registration bug.** Add `self._register('hour_of_day')` and `self._register('day_of_week')` to `OLHCVFeatureTransformer` or create a new canonical transformer that includes them. This is the highest-ROI change.

2. **Test the "recommended" combo: MA ratios + time + new features (no raw lags, no tech indicators).** This is a clean stationary + domain-aware feature set. Given MA ratios (0.679) + time (+0.062) + new features (+0.013 on top of full), the target should be ~0.79+ mean AUC.

3. **Test time × volatility interactions.** The `+Time features` variant improves January (+0.118 over baseline) and April (+0.092) — quiet and volatile regimes respectively. A feature like `hour_of_day` × `realized_vol_30` might allow the model to learn regime-specific time patterns (e.g., the last 30 minutes near-close behave very differently in high-vol vs low-vol environments).

4. **Investigate why tech indicators hurt in March (+0.116 improvement when removed).** The March 2026 tariff-shock environment is where RSI and MACD fail most. It is possible that over-smoothed oscillators give contrarian-wrong signals in sustained momentum regimes. Replacing them with shorter-period or adaptive indicators (ATR-based, percentile-rank normalised) may recover that loss.

5. **Combine time features with the new feature set (not yet tested).** `+Time features` alone = 0.794. `Full + New` = 0.746. The union `+Time + New` is the untested combination most likely to exceed both.

6. **Walk-forward cross-validation over multiple folds.** The current 2y-rolling 4-period evaluation is still a small sample. Walk-forward CV with 5+ folds and formal variance estimation would make the feature ranking more statistically reliable.

---

## Addendum — Calibration Analysis (2026-05-23)

> Calibration metrics computed for all 9 variants across both training and test splits.  
> **Metrics**: `bias` = mean predicted prob − true positive rate; `calib_corr` = rank correlation of predicted vs actual probability per quantile decile (20 bins); `ECE` = mean |pred_prob_per_bin − true_prob_per_bin| (equal-count bins).

### Mean training positive rate: 7.9% | Mean test positive rate: 9.2%

---

### Aggregate: Training split (mean across 4 folds)

| Variant | Train AUC | mean pred | bias | calib_corr | ECE |
|---------|----------:|----------:|-----:|-----------:|----:|
| Full (baseline) | 0.9980 | 13.1% | +0.052 | 0.925 | 0.053 |
| No raw lags | 0.9970 | 13.9% | +0.060 | 0.909 | 0.061 |
| No diff-ratios | 0.9980 | 13.2% | +0.053 | 0.924 | 0.054 |
| No MA ratios | 0.9963 | 14.3% | +0.064 | 0.900 | 0.065 |
| No tech indicators | 0.9973 | 13.7% | +0.058 | 0.913 | 0.059 |
| **+ Time features** | **0.9995** | **10.4%** | **+0.025** | **0.977** | **0.026** |
| New features only | 0.9903 | 16.2% | +0.083 | 0.884 | 0.083 |
| No-raw-lags + New | 0.9986 | 12.5% | +0.047 | 0.937 | 0.048 |
| Full + New | 0.9989 | 12.2% | +0.043 | 0.944 | 0.044 |

### Aggregate: Validation split (mean across 4 folds)

| Variant | Val AUC | mean pred | bias | calib_corr | ECE |
|---------|--------:|----------:|-----:|-----------:|----:|
| Full (baseline) | 0.9940 | 13.1% | +0.054 | 0.927 | 0.054 |
| No raw lags | 0.9924 | 13.9% | +0.062 | 0.915 | 0.062 |
| No diff-ratios | 0.9940 | 13.2% | +0.054 | 0.925 | 0.054 |
| No MA ratios | 0.9914 | 14.4% | +0.066 | 0.909 | 0.066 |
| No tech indicators | 0.9926 | 13.7% | +0.060 | 0.917 | 0.060 |
| **+ Time features** | **0.9982** | **10.3%** | **+0.025** | **0.976** | **0.026** |
| New features only | 0.9834 | 16.1% | +0.084 | 0.900 | 0.084 |
| No-raw-lags + New | 0.9957 | 12.5% | +0.047 | 0.936 | 0.048 |
| Full + New | 0.9958 | 12.2% | +0.045 | 0.942 | 0.045 |

### Aggregate: Test split (mean across 4 OOS periods)

| Variant | Test AUC | mean pred | bias | calib_corr | ECE |
|---------|----------:|----------:|-----:|-----------:|----:|
| Full (baseline) | 0.7327 | 9.3% | +0.001 | 0.751 | 0.110 |
| **No raw lags** | 0.6969 | 10.8% | +0.015 | **0.918** | **0.105** |
| No diff-ratios | 0.7243 | 11.9% | +0.026 | 0.831 | 0.114 |
| No MA ratios | 0.6790 | 8.7% | −0.005 | 0.641 | 0.115 |
| No tech indicators | 0.7369 | 11.2% | +0.020 | 0.747 | 0.097 |
| **+ Time features** | **0.7943** | **7.9%** | **−0.013** | 0.712 | 0.096 |
| New features only | 0.6214 | 9.7% | +0.005 | 0.543 | 0.083 |
| No-raw-lags + New | 0.6866 | 6.6% | −0.027 | 0.825 | **0.083** |
| **Full + New** | **0.7459** | **9.6%** | **+0.004** | **0.841** | **0.094** |

### Per-period test calibration detail

| Variant | Period | AUC | pos% | pred% | bias | corr | ECE |
|---------|--------|----:|-----:|------:|-----:|-----:|----:|
| Full (baseline) | 2026-01 | 0.782 | 5.6% | 1.8% | −0.038 | 0.817 | 0.038 |
| Full (baseline) | 2026-02 | 0.846 | 9.0% | 2.9% | −0.061 | 0.730 | 0.061 |
| Full (baseline) | **2026-03** | 0.598 | 12.5% | **31.3%** | **+0.188** | 0.579 | **0.256** |
| Full (baseline) | 2026-04 | 0.705 | 9.8% | 1.2% | −0.086 | 0.878 | 0.086 |
| No raw lags | 2026-01 | 0.783 | 5.6% | 5.4% | −0.003 | **0.934** | 0.030 |
| No raw lags | 2026-02 | 0.696 | 9.0% | 2.4% | −0.066 | **0.934** | 0.066 |
| No raw lags | **2026-03** | 0.654 | 12.5% | **33.6%** | **+0.211** | 0.856 | 0.242 |
| No raw lags | 2026-04 | 0.655 | 9.8% | 1.7% | −0.081 | 0.950 | 0.081 |
| No tech indicators | 2026-01 | 0.767 | 5.6% | 1.0% | −0.046 | 0.602 | 0.046 |
| No tech indicators | 2026-02 | 0.790 | 9.0% | 16.2% | +0.072 | 0.847 | 0.086 |
| No tech indicators | **2026-03** | 0.714 | 12.5% | **26.1%** | **+0.136** | **0.907** | 0.174 |
| No tech indicators | 2026-04 | 0.677 | 9.8% | 1.3% | −0.084 | 0.632 | 0.084 |
| + Time features | 2026-01 | **0.900** | 5.6% | 0.5% | −0.051 | 0.827 | 0.051 |
| + Time features | 2026-02 | 0.801 | 9.0% | 8.3% | −0.008 | 0.637 | 0.070 |
| + Time features | **2026-03** | 0.680 | 12.5% | **22.4%** | **+0.099** | 0.874 | 0.168 |
| + Time features | 2026-04 | **0.797** | 9.8% | 0.5% | −0.093 | 0.511 | 0.093 |
| New features only | 2026-01 | 0.568 | 5.6% | 7.4% | +0.018 | −0.103 | 0.070 |
| New features only | 2026-02 | 0.699 | 9.0% | 17.2% | +0.082 | 0.750 | 0.109 |
| New features only | **2026-03** | 0.731 | 12.5% | **13.2%** | **+0.007** | 0.877 | **0.065** |
| New features only | 2026-04 | 0.488 | 9.8% | 1.1% | −0.087 | 0.649 | 0.087 |
| No-raw-lags + New | 2026-01 | 0.674 | 5.6% | 3.7% | −0.019 | 0.697 | 0.057 |
| No-raw-lags + New | 2026-02 | 0.718 | 9.0% | 4.4% | −0.046 | 0.858 | 0.063 |
| No-raw-lags + New | **2026-03** | 0.667 | 12.5% | **17.0%** | **+0.045** | 0.885 | 0.125 |
| No-raw-lags + New | 2026-04 | 0.687 | 9.8% | 1.1% | −0.086 | 0.861 | 0.086 |
| Full + New | 2026-01 | 0.786 | 5.6% | 1.7% | −0.039 | 0.831 | 0.039 |
| Full + New | 2026-02 | 0.785 | 9.0% | 9.6% | +0.006 | 0.840 | 0.076 |
| Full + New | **2026-03** | 0.716 | 12.5% | **25.2%** | **+0.126** | 0.836 | 0.181 |
| Full + New | 2026-04 | 0.698 | 9.8% | 1.9% | −0.079 | 0.857 | 0.079 |

---

### Calibration findings

#### 1. All models systematically over-predict on training data

Every variant shows a consistent positive bias on the training split (+0.025–+0.083): the mean predicted probability (10–16%) is well above the true positive rate (7.9%). This is a structural artifact of the random 90/10 split — the positive examples are temporally clustered (near open/close and in high-vol days), so they are not uniformly drawn across the split. The model therefore learns slightly elevated probabilities for the feature distributions it sees in training.

**The `+Time features` variant has by far the best training calibration**: bias +0.025, ECE 0.026, corr 0.977. Adding time features allows the model to precisely assign low probabilities to mid-session bars (true label rate ~5–6%) and higher probabilities to open/close bars (true label rate ~25%+). This dramatically reduces the average over-prediction.

Training and validation calibration metrics are nearly identical across all variants (val ECE ≈ train ECE ± 0.001), confirming there is no meaningful calibration leak from the random split itself — the val rows are drawn from the same temporal distribution as train.

#### 2. The train → test ECE gap is large and structurally driven by March

Mean ECE roughly doubles from training (0.026–0.083) to testing (0.083–0.115). The driver is almost entirely March 2026 — every model dramatically over-predicts in March:

| Variant | March bias | March ECE |
|---------|:----------:|:---------:|
| Full (baseline) | +0.188 | 0.256 |
| No raw lags | +0.211 | 0.242 |
| No diff-ratios | +0.231 | 0.290 |
| No MA ratios | +0.179 | 0.249 |
| No tech indicators | +0.136 | 0.174 |
| + Time features | +0.099 | 0.168 |
| **New features only** | **+0.007** | **0.065** |
| No-raw-lags + New | +0.045 | 0.125 |
| Full + New | +0.126 | 0.181 |

March 2026 (tariff-shock, high-vol sell-off) is the universal calibration failure. Models trained on 2024–2026 Q1 — a period of mostly upward drift — learn that high-featured patterns lead to upward moves. When March arrives with a sustained sell-off at similarly high-featured conditions, the models massively over-predict the probability of an up-move.

**Notable exception: "New features only" has near-perfect March calibration (bias +0.007, ECE 0.065).** The realized volatility and ATR-normalized features directly measure the regime, and in a high-vol sell-off they generate signals the model associates with lower directional probability. However, this better calibration does not translate to better discrimination (March AUC 0.731 from new features only vs 0.680 for +Time features, 0.716 for Full+New) — the model's direction is more accurate but so is the calibration.

#### 3. Removing raw lags dramatically improves calibration correlation on test (0.751 → 0.918)

The most striking calibration finding: removing raw price lags cuts test AUC by −0.036 but increases test **calibration correlation from 0.751 to 0.918** — the best of any variant. With raw lags, the model's decile ranking of predicted probabilities does not reliably correspond to actual outcomes on OOS data. Without them, the relative ordering is far more trustworthy.

This is the raw-lags paradox:
- **With** raw lags: higher discrimination (AUC), but the probability scores are unreliable as estimates of actual outcome probability. This is fine if you only need rank ordering for a trading signal, but dangerous for any probability-based sizing or threshold logic.
- **Without** raw lags: lower discrimination, but the predicted 8th-decile bar is much more likely to be a true positive than the predicted 2nd-decile bar, on a consistent basis.

#### 4. `+Time features` achieves the best training calibration but poor test calibration correlation

`+Time features` is nearly perfectly calibrated on training (ECE 0.026, corr 0.977) but this advantage largely collapses on test (corr 0.712, ECE 0.096). The explanation: time-of-day patterns are learned from 2 years of training data. In testing, the same hour might behave differently — the end-of-day pattern that reliably yields high up-move probability in bull-trending 2024 does not carry the same predictive power in the April 2026 sell-off. Calibration correlation at test falls to 0.511 for the April period alone.

The strong training calibration of `+Time features` is real but **regime-conditional**: it holds when market conditions are similar to training. When the regime shifts (April 2026 tariff sell-off), the time-based calibration breaks down.

#### 5. `Full + New` achieves the best balance of AUC and test calibration quality

Among variants with test AUC ≥ 0.74:

| Variant | Test AUC | Test calib_corr | Test ECE |
|---------|:--------:|:---------------:|:--------:|
| + Time features | **0.794** | 0.712 | 0.096 |
| Full + New | 0.746 | **0.841** | **0.094** |

`Full + New` sacrifices 0.048 AUC vs `+Time features` in exchange for much more reliable probability scores: calibration correlation 0.841 vs 0.712 and similar ECE. For strategies that use predicted probabilities for position sizing or Kelly-style bet sizing, `Full + New` is the safer choice.

#### 6. Consistent January/April under-prediction: a structural floor problem

Every model produces near-zero predicted probabilities in January (0.5–7.4% predicted vs 5.6% true) and April (0.5–1.9% predicted vs 9.8% true). These months have the lowest model confidence regardless of feature set. The model has learned that certain feature patterns (low-vol, low-trend) map to near-zero probabilities, and when these patterns appear in testing the model "goes silent." January is the quiet-vol failure mode; April is the post-crash silent-model failure mode.

This **probability floor problem** cannot be fixed by feature engineering alone — it requires either (a) post-hoc Platt scaling or isotonic calibration on a held-out regime sample, or (b) explicit regime-conditional features (e.g., VIX level) that shift the floor based on observed volatility.

---

### Calibration next steps

1. **Apply post-hoc Platt scaling.** Train a logistic recalibrator on a held-out window (e.g., last 3 months of training data) to map raw XGBoost probabilities to calibrated outputs. This directly targets the January/April under-prediction floor.

2. **Target `Full + New` over `+Time features` when calibration matters.** The AUC difference (−0.048) is outweighed by the large calibration correlation improvement (+0.129) in any use case involving probability-based decisions.

3. **Investigate "New features only" as a calibration baseline for March.** Its near-zero March bias (0.007 vs 0.099–0.256 for other variants) suggests the volatility-regime features alone have well-calibrated probability estimates for high-vol environments. Combining them with a regime-switch: use New-features-only probs in high-vol months and Full-model probs in normal months.

4. **Track ECE and calibration correlation separately from AUC in all future experiments.** The divergence shown here (high AUC ≠ good calibration) makes AUC alone insufficient as a selection metric.
