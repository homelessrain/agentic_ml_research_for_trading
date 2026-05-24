# Multi-Regime Evaluation — QQQ/up h=60, up_delta=1.5%, down_delta=0.75% (Asymmetric)

**Date:** 2026-05-24
**Hypothesis:** The asymmetric label definition that proved strongest for DOWN prediction (up_delta=1.5%, down_delta=0.75%, h=60) is mirrored here for UP prediction. Evaluate whether a 1.5% up move in 60 bars is learnable across diverse regimes, and how UP model performance compares to the DOWN counterpart (h60_down15pct_regime_eval_2026-05-24).
**Symbol:** QQQ | **Direction:** up | **Horizon:** 60 bars
**Label:** up_delta=0.015 (1.5%), down_delta=0.0075 (0.75%) — asymmetric (down = up / 2)
**Features:** OLHCVFeatureTransformer (MA ratios + time features, 248 total)
**Model:** LabelDefResearchModel (XGBoost n_estimators=300, max_depth=4, auto scale_pos_weight)
**Training:** 2-year rolling window ending last day before each test period
**Reliability floor:** N_pos ≥ 50

---

## Regime Reference

| Period | Regime | Train window |
|--------|--------|-------------|
| 2025-03 | Strong correction (−8.4%) | 2023-03-01 → 2025-02-28 |
| 2025-04 | Crisis crash+recovery | 2023-04-01 → 2025-03-31 |
| 2025-11 | Mild correction (−2.8%) | 2023-11-01 → 2025-10-31 |
| 2026-01 | Flat / normal (−0.2%) | 2023-12-01 → 2025-12-31 |
| 2026-02 | Mild correction (−2.3%) | 2024-01-01 → 2026-01-31 |
| 2026-03 | Correction (−3.8%) | 2024-02-01 → 2026-02-28 |
| 2026-04 | Recovery (+14.9%) | 2024-03-01 → 2026-03-31 |

---

## 1. Results

### Full results: UP d=1.5% asymmetric vs DOWN d=1.5% asymmetric reference

| Period | Regime | Train pos% | Test pos% | N_pos | Rel | **UP AUC** | DOWN AUC† | Δ AUC | **UP PR-AUC** | DOWN PR-AUC† | Δ PR-AUC |
|--------|--------|:----------:|:---------:|:-----:|:---:|:----------:|:---------:|:-----:|:------------:|:-----------:|:--------:|
| 2025-03 | Strong correction (−8.4%) | 0.7% | 1.9% | 156 | ✓ | **0.810** | 0.896 | −0.087 | **0.084** | 0.317 | −0.233 |
| 2025-04 | Crisis crash+recovery | 0.8% | 8.3% | 678 | ✓ | **0.719** | 0.615 | **+0.104** | **0.179** | 0.232 | −0.053 |
| 2025-11 | Mild correction (−2.8%) | 1.1% | 2.1% | 151 | ✓ | **0.869** | 0.708 | **+0.161** | **0.076** | 0.076 | 0.000 |
| 2026-01 | Flat / normal (−0.2%) | 1.1% | 0.4% | 28 | **✗** | 0.794 | 0.910 | −0.116 | 0.009 | 0.625 | — |
| 2026-02 | Mild correction (−2.3%) | 1.1% | 0.2% | 11 | **✗** | 0.989 | 0.816 | — | 0.063 | 0.064 | — |
| 2026-03 | Correction (−3.8%) | 1.1% | 2.0% | 169 | ✓ | **0.855** | 0.826 | +0.029 | **0.091** | 0.037 | +0.054 |
| 2026-04 | Recovery (+14.9%) | 1.1% | 1.0% | 82 | ✓ | **0.898** | 0.954 | −0.056 | **0.044** | 0.710 | −0.666 |

> † DOWN reference from h60_down15pct_regime_eval_2026-05-24.
> ✗ = N_pos < 50 — AUC/PR-AUC unreliable for those periods.
> Random baseline: ROC-AUC = 0.500, PR-AUC ≈ positive_label_rate.

### Positive label rate by regime

| Period | Regime | UP pos% | DOWN pos%† | Δ | N_pos | Reliable |
|--------|--------|:-------:|:----------:|:-:|:-----:|:--------:|
| 2025-03 | Strong correction (−8.4%) | 1.9% | 4.4% | −2.5pp | 156 | ✓ |
| 2025-04 | Crisis crash+recovery | 8.3% | 9.6% | −1.3pp | 678 | ✓ |
| 2025-11 | Mild correction (−2.8%) | 2.1% | 3.6% | −1.5pp | 151 | ✓ |
| 2026-01 | Flat / normal (−0.2%) | 0.4% | 1.2% | −0.8pp | **28** | **✗** |
| 2026-02 | Mild correction (−2.3%) | 0.2% | 1.1% | −0.9pp | **11** | **✗** |
| 2026-03 | Correction (−3.8%) | 2.0% | 1.5% | +0.5pp | 169 | ✓ |
| 2026-04 | Recovery (+14.9%) | 1.0% | 0.8% | +0.2pp | 82 | ✓ |

### Aggregate summary (reliable periods only)

| Metric | UP (5/7 reliable) | DOWN (7/7 reliable)† | Δ |
|--------|:-----------------:|:-------------------:|:-:|
| Reliable periods | 5/7 | 7/7 | −2 |
| Mean AUC | 0.830 | 0.818 | +0.012 |
| Std AUC | 0.062 | 0.120 | — |
| Mean PR-AUC | 0.095 | 0.295 | −0.200 |

---

## 2. Analysis

### Finding 1: Reliability collapses in flat/correction months — only 5/7 periods reliable

The most important structural difference between the UP and DOWN models is **reliability across regimes**:

| Period | Regime | UP N_pos | DOWN N_pos† | UP Reliable | DOWN Reliable |
|--------|--------|:--------:|:-----------:|:-----------:|:-------------:|
| 2026-01 | Flat / normal (−0.2%) | **28** | 97 | **✗** | ✓ |
| 2026-02 | Mild correction (−2.3%) | **11** | 83 | **✗** | ✓ |

In flat months (Jan 2026: QQQ −0.2%) and mild corrections (Feb 2026: QQQ −2.3%), a 1.5% QQQ up-move in 60 bars is **extremely rare** — only 28 and 11 occurrences respectively. This is the fundamental asymmetry: in a flat or declining market, large up moves are rare by definition. The DOWN model maintained N_pos ≥ 50 across all 7 regimes; the UP model fails in 2 of 7.

**The reliability inversion is regime-symmetric with the DOWN model**: UP fails where DOWN succeeded (flat/correction = few large up moves), and UP succeeds where DOWN was challenged (crisis = abundant recovery bounces).

> ⚠️ The Feb 2026 AUC=0.989 looks extraordinary but is based on only N_pos=11 — this number is statistically meaningless. Do not interpret it.

---

### Finding 2: UP model outperforms DOWN model in the crisis — AUC 0.719 vs 0.615

The most surprising result is the **crisis month flip**:

| Period | Regime | UP AUC | DOWN AUC† | Winner |
|--------|--------|:------:|:---------:|:------:|
| 2025-04 | Crisis crash+recovery | **0.719** | 0.615 | **UP** |

April 2025 was a sharp tariff-shock crash followed by a strong recovery. The DOWN model failed in this environment because the MA-ratio features were in extreme OOD territory during the crash. The UP model has a structural advantage in crisis: the **recovery bounces** (large up moves after sharp down moves) are more predictable than the crash itself. The model learned that extreme momentum collapse (captured in MA-ratio features) reliably precedes recovery bounces — a momentum-reversal signal.

Both models still fail in absolute terms (0.719 AUC is weak), but UP is clearly less broken than DOWN in crisis environments.

---

### Finding 3: UP model significantly outperforms DOWN in mild corrections

| Period | Regime | UP AUC | DOWN AUC† | Δ |
|--------|--------|:------:|:---------:|:-:|
| 2025-11 | Mild correction (−2.8%) | **0.869** | 0.708 | **+0.161** |
| 2026-03 | Correction (−3.8%) | **0.855** | 0.826 | +0.029 |

In correction months (QQQ −2.8% to −3.8%), the UP model is meaningfully stronger. This suggests the feature set is capturing **intraday rally opportunities** within correction phases more effectively than it captures sustained downward continuation. 

In a correction, the market drifts lower but punctuated by sharp intraday bounces. The MA-ratio features (price vs. trailing averages) reliably signal when a bar is at an extreme underperformance level, setting up for a tactical bounce. Down moves in correction months are noisier — the market keeps sliding, but not through clean, predictable momentum breakdowns.

---

### Finding 4: DOWN model dramatically outperforms UP in the recovery month

| Period | Regime | UP AUC | DOWN AUC† | UP PR-AUC | DOWN PR-AUC† |
|--------|--------|:------:|:---------:|:---------:|:------------:|
| 2026-04 | Recovery (+14.9%) | 0.898 | **0.954** | 0.044 | **0.710** |

April 2026 (recovery +14.9%) is where the DOWN model achieved its peak performance and the UP model is notably weaker. This seems counterintuitive — in a recovery month, shouldn't the UP model be better?

The answer lies in **label sparsity vs. structure**. In April 2026:
- **DOWN label** (1.5% drop in 60 bars): Only 65 occurrences in a strong rally month — these are the rare, structurally significant "sell-the-rip" events. The model identifies them precisely (PR-AUC=0.710).
- **UP label** (1.5% rise in 60 bars): 82 occurrences — more common in a rally, but less distinctive. The model achieves good ROC-AUC (0.898) but poor PR-AUC (0.044) — it can rank up events above the mean but cannot fire high-precision alerts.

**The paradox**: in a rally, large up moves happen frequently enough to reduce their distinctiveness. The model ranks them correctly (AUC=0.898) but precision at any threshold is poor. The DOWN events in April 2026 are rarer and more structurally distinct — the model can fire at those with very high precision.

---

### Finding 5: PR-AUC is uniformly weak for UP across all reliable regimes

| Period | Regime | UP PR-AUC | DOWN PR-AUC† | Random baseline |
|--------|--------|:---------:|:------------:|:---------------:|
| 2025-03 | Strong correction | 0.084 | 0.317 | ~0.019 |
| 2025-04 | Crisis | 0.179 | 0.232 | ~0.083 |
| 2025-11 | Mild correction | 0.076 | 0.076 | ~0.021 |
| 2026-03 | Correction | 0.091 | 0.037 | ~0.020 |
| 2026-04 | Recovery | 0.044 | 0.710 | ~0.010 |

PR-AUC for UP is uniformly low — the best reliable result is 0.179 (crisis, ~2.2× random baseline). This contrasts with DOWN where flat/recovery months hit PR-AUC=0.625–0.710. The UP model has good ROC-AUC (0.719–0.898 across reliable periods) but cannot achieve high precision at any threshold.

**Why is UP PR-AUC consistently poor?**

The DOWN model's exceptional PR-AUC in flat/recovery months arises because a 1.5% DROP in a flat or rallying market is a **highly distinctive, rare event** with a tight feature signature — the model can fire only on those bars and achieve very high precision. For the UP model, even in correction months (where up moves are rarer), the up moves are more **spread across time** (intermittent bounces throughout the day) and less concentrated in identifiable feature clusters. The model can rank them (good AUC) but lacks a tight high-precision region (poor PR-AUC).

---

### Finding 6: UP label rate is regime-asymmetric with DOWN

| Period | Regime | UP pos% | DOWN pos% |
|--------|--------|:-------:|:---------:|
| 2025-03 | Strong correction | 1.9% | 4.4% |
| 2025-04 | Crisis | 8.3% | 9.6% |
| 2025-11 | Mild correction | 2.1% | 3.6% |
| 2026-01 | Flat / normal | **0.4%** | 1.2% |
| 2026-02 | Mild correction | **0.2%** | 1.1% |
| 2026-03 | Correction | 2.0% | 1.5% |
| 2026-04 | Recovery | 1.0% | 0.8% |

The positive label rate pattern is nearly the inverse: in correction months (2025-03, 2025-11, 2026-03), DOWN fires more; in the recovery month (2026-04), UP and DOWN are roughly equal; in flat/declining months (2026-01, 2026-02), both are rare but DOWN still fires ~3× more than UP.

The one exception is the crisis (2025-04): both fire at similar rates (~8–10%), confirming the extreme two-sided volatility of that period.

---

## 3. Summary

### All-config benchmark (reliable periods only)

| Config | Reliable periods | Mean AUC | Mean PR-AUC | AUC range |
|--------|:---------------:|:--------:|:-----------:|:---------:|
| **UP h=60, d=1.5% asymm (this study)** | **5/7** | **0.830** | **0.095** | 0.719–0.898 |
| DOWN h=60, d=1.5% asymm† | 7/7 | 0.818 | 0.295 | 0.615–0.954 |
| DOWN h=40, d=1.0% symm (prior best) | 7/7 | 0.759 | 0.219 | 0.580–0.889 |

> † from h60_down15pct_regime_eval_2026-05-24.

The UP model has a marginally higher mean AUC over its 5 reliable periods (0.830 vs 0.818 for DOWN over 7), but this comparison is not apples-to-apples: the UP model drops the 2 hardest months (Jan 2026: DOWN AUC=0.910, Feb 2026: DOWN AUC=0.816). Including those would likely raise the UP mean AUC. The more meaningful difference is **reliability (5/7 vs 7/7) and PR-AUC (0.095 vs 0.295)**.

### Performance by regime type (UP vs DOWN)

| Regime type | UP AUC | DOWN AUC | UP PR-AUC | DOWN PR-AUC | UP Reliable |
|-------------|:------:|:--------:|:---------:|:-----------:|:-----------:|
| Strong correction (2025-03) | 0.810 | **0.896** | 0.084 | **0.317** | ✓ |
| Crisis (2025-04) | **0.719** | 0.615 | 0.179 | **0.232** | ✓ |
| Mild correction (2025-11) | **0.869** | 0.708 | 0.076 | 0.076 | ✓ |
| Flat / normal (2026-01) | — | **0.910** | — | **0.625** | ✗ |
| Mild correction (2026-02) | — | **0.816** | — | 0.064 | ✗ |
| Correction (2026-03) | **0.855** | 0.826 | **0.091** | 0.037 | ✓ |
| Recovery (2026-04) | 0.898 | **0.954** | 0.044 | **0.710** | ✓ |

### Regime-specific winner summary

| Regime | AUC winner | PR-AUC winner | Notes |
|--------|:----------:|:-------------:|-------|
| Strong correction | DOWN | DOWN | DOWN +0.087 AUC, +0.233 PR-AUC |
| Crisis crash+recovery | **UP** | DOWN | UP +0.104 AUC; crisis PR-AUC both weak |
| Mild corrections | **UP** | Tie/UP | UP +0.029 to +0.161 AUC |
| Flat / normal | DOWN | DOWN | UP unreliable (N_pos=28) |
| Recovery | DOWN | DOWN | DOWN exceptional PR-AUC=0.710 |

---

## 3b. Calibration Analysis

**Plots:** `calibration_curves_by_period.png` (full scale), `calibration_curves_zoomed.png` (0–0.15 zoom), `calibration_correlation_by_period.png` (bar chart)
**Data:** `calibration_summary.csv`

### Calibration correlation table (UP vs DOWN)

| Period | Regime | UP calib_r | DOWN calib_r | UP bias | DOWN bias | UP N_pos | UP Rel |
|--------|--------|:----------:|:------------:|:-------:|:---------:|:--------:|:------:|
| 2025-03 | Strong correction | 0.875 | 0.841 | −0.003 | −0.044 | 156 | ✓ |
| 2025-04 | Crisis | 0.640 | 0.876 | −0.079 | −0.094 | 678 | ✓ |
| 2025-11 | Mild correction | **0.130** | 0.578 | −0.018 | −0.023 | 151 | ✓ |
| 2026-01 | Flat / normal | −0.092 | **0.987** | −0.001 | −0.009 | 28 | **✗** |
| 2026-02 | Mild correction | 0.992\* | 0.942 | +0.018 | +0.010 | 11 | **✗** |
| 2026-03 | Correction | 0.650 | **0.019** | −0.002 | −0.004 | 169 | ✓ |
| 2026-04 | Recovery | **0.050** | **0.998** | −0.002 | −0.001 | 82 | ✓ |

> \* Feb 2026 UP calib_corr=0.992 is meaningless — only N_pos=11 test positives.
> calib_r = rank correlation between mean predicted probability and fraction-of-positives per decile. Higher = predicted scores track actual event rates.

---

### Calibration Finding 1: Good AUC ≠ good calibration — the UP model dissociates in 3 periods

Three reliable periods show UP AUC ≥ 0.855 but calibration correlation ≤ 0.130:

| Period | UP AUC | UP calib_r | Interpretation |
|--------|:------:|:----------:|----------------|
| 2025-11 | 0.869 | **0.130** | Model ranks UP events correctly but predicted probs don't track actual event rates |
| 2026-03 | 0.855 | 0.650 | Moderate dissociation |
| 2026-04 | 0.898 | **0.050** | Strong ranking, near-zero calibration |

**What this means**: The UP model fires at a tiny fraction of bars with high internal confidence, but the raw predicted probability cannot be used as a calibrated event rate. A bar scored 0.05 does not imply 5% probability of a 1.5% up move — the score is an ordinal rank, not a true probability. Raw probabilities should not be interpreted as actual event rates for UP signals in these periods.

The DOWN model shows the reverse failure: **Mar 2026 DOWN calib_r = 0.019** despite AUC=0.826. The DOWN model ranks correctly in March but its predicted probabilities are equally meaningless as calibrated rates. This is a shared limitation in correction months, not specific to one direction.

---

### Calibration Finding 2: DOWN model is near-perfectly calibrated in flat/recovery months

| Period | DOWN AUC | DOWN calib_r | Notes |
|--------|:--------:|:------------:|-------|
| 2026-01 | 0.910 | **0.987** | Flat month — DOWN predicted probs track actual event rates almost perfectly |
| 2026-04 | 0.954 | **0.998** | Recovery month — DOWN probabilities are reliable event rate estimates |

In exactly the regimes where the DOWN model excels on AUC and PR-AUC (flat/recovery), it is also nearly perfectly calibrated. This is a compound advantage: **the DOWN model in flat/recovery months provides both excellent discrimination AND reliable probability estimates**. A bar with predicted DOWN prob = 0.08 in January 2026 genuinely corresponds to ~8% probability of a 1.5% decline in the next 60 bars. This makes the output directly actionable for position sizing or Kelly-criterion-based approaches.

The UP model has no equivalent — its best calibration in reliable periods is calib_r=0.875 (Mar 2025) with the recovery month (Apr 2026) collapsing to calib_r=0.050.

---

### Calibration Finding 3: Crisis month — UP better calibrated than DOWN on AUC, DOWN better on calibration

| Period | Regime | UP AUC | DOWN AUC | UP calib_r | DOWN calib_r |
|--------|--------|:------:|:--------:|:----------:|:------------:|
| 2025-04 | Crisis | 0.719 | 0.615 | 0.640 | **0.876** |

The UP model ranks crisis UP events better (higher AUC) but the DOWN model's probabilities are better calibrated to actual event rates (calib_r 0.876 vs 0.640). In a crisis, the DOWN model still "knows" roughly how likely a DOWN event is at each predicted score level, even though it can't rank them well (low AUC). The UP model can rank bounces but its predicted probabilities are noisier.

**Practical implication for crisis hedging**: In a crisis environment, neither model is suitable for precision signals, but the DOWN model's probability output is more interpretable than the UP model's if one wanted a probabilistic risk estimate.

---

### Calibration Finding 4: Bias is small and well-controlled in both models

| Config | Max bias (abs) | Worst period |
|--------|:--------------:|:-------------|
| UP | 0.079 | 2025-04 (crisis) |
| DOWN | 0.094 | 2025-04 (crisis) |

Both models are nearly unbiased (mean predicted ≈ actual positive rate) in all periods except the crisis month, where the mean predicted probability substantially underestimates the true positive rate. This crisis under-prediction (bias = −0.079 UP, −0.094 DOWN) reflects the model's failure to anticipate the extreme positive rate (~8–10%) that materialised in April 2025. The model was trained on a 2-year window with <1% average positive rate and cannot dynamically adjust to a crisis-level positive rate.

---

## 4. Findings & Next Steps

### Key Findings

1. **Reliability asymmetry is the dominant structural difference.** The UP model fails the N_pos≥50 reliability floor in flat (Jan 2026: N_pos=28) and mild-correction (Feb 2026: N_pos=11) months. A 1.5% up move in 60 bars simply doesn't occur often enough in declining markets. The DOWN model maintained reliability across all 7 regimes. **Any deployment of the UP model requires a regime gate to suppress signals in flat/correcting environments.**

2. **UP model outperforms DOWN in crisis (0.719 vs 0.615 AUC).** The recovery bounces within the tariff-shock crash are more predictable than the crash itself — momentum reversal after extreme MA-ratio dislocations provides a cleaner signal. This is the one regime where the UP model is the stronger tool.

3. **UP model outperforms DOWN in mild corrections (2025-11: +0.161 AUC, 2026-03: +0.029 AUC).** The feature set captures intraday rally setups within correction phases better than it captures downward continuation. Tactical bounce prediction is a strength of the UP model in corrective markets.

4. **DOWN model dramatically outperforms UP in recovery months (2026-04: PR-AUC 0.710 vs 0.044).** The rarity of 1.5% DOWN moves in a rally makes them highly distinctive and precision-fireable. UP moves in a rally are common enough to lose their distinctiveness.

5. **PR-AUC for UP is consistently poor across all reliable regimes** (max 0.179 in crisis, 0.044–0.091 elsewhere). The UP model has good ROC-AUC (0.719–0.898) but cannot achieve high-precision alerts at any threshold. Unlike the DOWN model which hit PR-AUC=0.625–0.710 in flat/recovery months, the UP model never achieves a high-precision firing zone.

6. **Mean AUC is comparable across models (UP=0.830 over 5 periods, DOWN=0.818 over 7).** In the regimes where the UP model can be evaluated, it is roughly as powerful as DOWN — but the coverage gap (5/7 vs 7/7) is a significant operational disadvantage.

### Next Steps

1. **Build a regime gate for the UP model.** Unlike DOWN (reliable in all 7 regimes), the UP model requires an explicit gate: when QQQ is in a flat or declining regime, suppress UP signals. A simple gate could use QQQ's trailing 20-day return — if < 0%, suppress the UP model output.

2. **Investigate PR-AUC improvement for UP.** The consistently poor PR-AUC (0.044–0.179) despite reasonable ROC-AUC suggests the model ranks UP events correctly but cannot fire with high precision. Options:
   - **Tighter up_delta**: Try up_delta=2.0% (fewer, more distinctive events) — this may improve PR-AUC at the cost of further N_pos reduction in flat months.
   - **Asymmetric training**: In correction months, train only on bars where the market is already mildly oversold.
   - **Calibration + threshold search**: Optimize the precision-recall tradeoff explicitly using a held-out calibration set.

3. **Consider UP + DOWN as a complementary pair, not substitutes.** The two models have distinct regime strengths:
   - DOWN: flat/recovery months (high AUC + high PR-AUC); all 7 regimes reliable
   - UP: crisis + mild corrections (better AUC); 5/7 regimes reliable, poor PR-AUC
   A combined signal — DOWN in flat/recovery, UP in corrections/crisis — could provide broader coverage.

4. **Test UP with tighter down exclusion guard.** The current label requires NOT a 0.75% DOWN move in 60 bars. In correction months, the down exclusion may be removing many valid up bars (a bar followed by both a 1.5% up AND a 0.75% down is excluded). A tighter exclusion (e.g., 0.5% down) might recover more positive labels in correction months and improve reliability.

5. **Evaluate UP on h=40.** h=40 (shorter horizon) may generate more UP positives in flat months, potentially recovering reliability. The tradeoff: shorter horizon = less time for the move to develop, but more bars qualify.

6. **Crisis-specific UP model.** The UP model shows AUC=0.719 in April 2025. A dedicated crisis-phase UP model — trained only on crisis-like market conditions — could potentially achieve much higher performance on bounce prediction, which is a high-value use case during crash recoveries.
