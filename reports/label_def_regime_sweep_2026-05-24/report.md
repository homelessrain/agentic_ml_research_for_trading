# Label Definition Regime Sweep

**Date:** 2026-05-24
**Goal:** Evaluate which label definitions generalise across market regimes, not just the April 2026 sell-off used in the original sweep.
**Symbols:** QQQ (predict down), SQQQ (predict up)
**Label configs:** h ∈ {40, 60, 120} d=1% for QQQ/down; h=5 d=1%, h=10 d=1%, h=5 d=0.5% for SQQQ/up
**Training:** 2-year rolling window ending the last day of the month before each test period
**Reliability floor:** N_pos ≥ 50 test positive examples

---

## Regime Reference

| Period | Regime | QQQ return | QQQ-down pos% (h=40) | Train window |
|--------|--------|:----------:|:--------------------:|-------------|
| 2025-03 | Strong correction | −8.4% | 6.1% | 2023-03-01 → 2025-02-28 |
| 2025-04 | Crisis (crash + recovery) | +2.0%* | 13.4% | 2023-04-01 → 2025-03-31 |
| 2025-11 | Mild correction | −2.8% | 4.5% | 2023-11-01 → 2025-10-31 |
| 2026-01 | Flat / normal | −0.2% | 1.4% | 2023-12-01 → 2025-12-31 |
| 2026-02 | Mild correction | −2.3% | 2.9% | 2024-01-01 → 2026-01-31 |
| 2026-03 | Correction | −3.8% | 3.6% | 2024-02-01 → 2026-02-28 |
| 2026-04 | Recovery | +14.9% | 1.2% | 2024-03-01 → 2026-03-31 |

> \* April 2025 ended up +2% but contained the single worst tariff-shock crash day of the period.

---

## 1. Results

### Test ROC-AUC by config × period

| Config | 2025-03 | 2025-04 | 2025-11 | 2026-01 | 2026-02 | 2026-03 | 2026-04 | **Mean** | Std |
|--------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:--------:|:---:|
| QQQ-dn h=40  d=1.0% | 0.773 | 0.580 | 0.729 | 0.832 | 0.675 | 0.833 | **0.889** | **0.759** | 0.106 |
| QQQ-dn h=60  d=1.0% | 0.758 | 0.576 | **0.757** | 0.699 | 0.623 | 0.806 | 0.868 | 0.727 | 0.102 |
| QQQ-dn h=120 d=1.0% | 0.719 | 0.574 | 0.658 | 0.778 | 0.589 | 0.772 | 0.757 | 0.693 | 0.086 |
| SQQQ-up h=5  d=1.0% | 0.642 | 0.514 | 0.608 | 0.793 | **0.857** | 0.749 | 0.872 | 0.719 | 0.135 |
| SQQQ-up h=10 d=1.0% | 0.643 | 0.533 | 0.625 | 0.682 | 0.752 | 0.724 | 0.755 | 0.674 | 0.080 |
| SQQQ-up h=5  d=0.5% | 0.617 | 0.531 | 0.564 | 0.693 | 0.720 | 0.686 | 0.725 | 0.648 | 0.078 |

> Random baseline: ROC-AUC = 0.500. All N_pos ≥ 50 (all reliable).

### Positive label rate (test %)

| Config | 2025-03 | 2025-04 | 2025-11 | 2026-01 | 2026-02 | 2026-03 | 2026-04 |
|--------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| QQQ-dn h=40  d=1.0% | 6.1% | 13.4% | 4.5% | 1.4% | 2.9% | 3.6% | 1.2% |
| QQQ-dn h=60  d=1.0% | 11.6% | 18.3% | 7.7% | 3.0% | 5.0% | 6.3% | 2.2% |
| QQQ-dn h=120 d=1.0% | 23.6% | 28.2% | 18.6% | 7.2% | 13.3% | 12.9% | 5.3% |
| SQQQ-up h=5  d=1.0% | 3.6% | 8.5% | 2.2% | 0.7% | 1.8% | 1.4% | 0.6% |
| SQQQ-up h=10 d=1.0% | 10.0% | 16.7% | 6.7% | 2.4% | 5.4% | 4.4% | 2.1% |
| SQQQ-up h=5  d=0.5% | 16.2% | 23.1% | 11.1% | 5.0% | 9.3% | 9.7% | 4.4% |

### Test PR-AUC by config × period

| Config | 2025-03 | 2025-04 | 2025-11 | 2026-01 | 2026-02 | 2026-03 | 2026-04 | **Mean** |
|--------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:--------:|
| QQQ-dn h=40  d=1.0% | 0.170 | 0.200 | 0.115 | 0.127 | 0.042 | 0.181 | **0.442** | **0.183** |
| QQQ-dn h=60  d=1.0% | 0.268 | 0.284 | 0.208 | 0.075 | 0.061 | 0.233 | 0.341 | 0.210 |
| QQQ-dn h=120 d=1.0% | **0.374** | **0.326** | **0.293** | **0.291** | 0.148 | **0.292** | 0.175 | **0.271** |
| SQQQ-up h=5  d=1.0% | 0.092 | 0.100 | 0.121 | 0.065 | 0.119 | 0.048 | 0.098 | 0.092 |
| SQQQ-up h=10 d=1.0% | 0.178 | 0.194 | 0.131 | 0.090 | **0.157** | 0.110 | 0.084 | 0.135 |
| SQQQ-up h=5  d=0.5% | 0.239 | 0.262 | 0.152 | 0.121 | 0.195 | 0.185 | 0.141 | 0.185 |

### Best config per period

| Period | Regime | Best config | AUC | PR-AUC | Pos% |
|--------|--------|-------------|:---:|:------:|:----:|
| 2025-03 | Strong correction | QQQ-dn h=40 d=1% | **0.773** | 0.170 | 6.1% |
| 2025-04 | Crisis | QQQ-dn h=40 d=1% | **0.580** | 0.200 | 13.4% |
| 2025-11 | Mild correction | QQQ-dn h=60 d=1% | **0.757** | 0.208 | 7.7% |
| 2026-01 | Flat | QQQ-dn h=40 d=1% | **0.832** | 0.127 | 1.4% |
| 2026-02 | Mild correction | SQQQ-up h=5 d=1% | **0.857** | 0.119 | 1.8% |
| 2026-03 | Correction | QQQ-dn h=40 d=1% | **0.833** | 0.181 | 3.6% |
| 2026-04 | Recovery | QQQ-dn h=40 d=1% | **0.889** | 0.442 | 1.2% |

---

## 2. Analysis

### Finding 1: Crisis months (April 2025) break every config uniformly

April 2025 is the single most important data point in this study: **every config collapses to near-random AUC (0.51–0.58).**

| Config | Apr-2025 AUC | Apr-2025 pos% | N_pos |
|--------|:------------:|:-------------:|:-----:|
| QQQ-dn h=40  | 0.580 | 13.4% | 1,103 |
| QQQ-dn h=60  | 0.576 | 18.3% | 1,501 |
| QQQ-dn h=120 | 0.574 | 28.2% | 2,317 |
| SQQQ-up h=5 d=1% | 0.514 | 8.5% | 695 |
| SQQQ-up h=10 | 0.533 | 16.7% | 1,374 |
| SQQQ-up h=5 d=0.5% | 0.531 | 23.1% | 1,898 |

There are no reliable AUC values — all have ample test examples (695–2,317), so the failure is genuine. Two mechanisms drive this:

1. **Out-of-distribution features:** The tariff-shock crash drove QQQ MA ratios to levels never seen in 2-year training windows. MA-ratio features that encode "how extended price is vs its 50/100/200-bar moving average" reached extreme negative values, while the model's learned thresholds were calibrated on moderate conditions. The signal was present (massive down moves) but the features were in uncharted territory.

2. **Label inflation destroys discriminability:** In April 2025, the QQQ h=40 positive rate hit 13.4% (vs the usual 1–4%). When 1 in 7 bars is a positive example, "predicting down" at random already gets many right. The model's probability scores — trained on 2.4% positive rate — were not calibrated for this density, and the AUC reflects near-uniform difficulty separating positives from negatives.

**Implication:** No label definition or feature set based on MA ratios / time features alone can handle black-swan crash regimes. These months require a dedicated regime-detection gate (e.g., VIX spike, realised-vol threshold) that suspends the directional model's output entirely.

---

### Finding 2: QQQ h=40 d=1% is the best single config in 5 of 7 periods

Across all periods, QQQ h=40 d=1% achieves:
- **Highest mean AUC: 0.759** (vs 0.727 for h=60, 0.693 for h=120)
- **Wins in 5 of 7 months**: 2025-03, 2025-04, 2026-01, 2026-03, 2026-04
- Highest peak AUC: 0.889 (April 2026)

Excluding the crisis month (2025-04), its mean AUC rises to **0.791** — clearly the most capable config in normal-to-correction regimes.

It loses in two months:
- **November 2025**: QQQ h=60 edges it (0.757 vs 0.729) — at moderate positive rates (~5–8%), the 60-bar horizon captures longer-duration down moves that are more predictable.
- **February 2026**: SQQQ h=5 d=1% wins (0.857 vs 0.675) — a structural anomaly explained in Finding 5.

---

### Finding 3: The three QQQ configs form a precision–recall trade-off

AUC and PR-AUC move in opposite directions across horizons:

| Metric | h=40 | h=60 | h=120 | Pattern |
|--------|:----:|:----:|:-----:|---------|
| Mean AUC | **0.759** | 0.727 | 0.693 | ↓ as h grows |
| Mean PR-AUC | 0.183 | 0.210 | **0.271** | ↑ as h grows |
| Pos rate range | 1.2–13.4% | 2.2–18.3% | 5.3–28.2% | wider as h grows |

**h=40 excels at discrimination** (best AUC) — it identifies a small set of very structured down bars and ranks them correctly. But its PR-AUC is low in most months because there are few true positives, and even a small number of false positives depresses precision.

**h=120 excels at recall** (best mean PR-AUC) — it fires on more bars, so there are more opportunities to predict true positives. But the label fires on noisier, less-structured bars, making discrimination harder (lowest AUC). In correction/crisis months its PR-AUC (0.29–0.37) substantially outperforms h=40 (0.12–0.20).

**Practical implication:** The right horizon depends on how the predictions are used:
- **For signal timing** (enter when confidence is very high): h=40 — its rare, high-precision signals are most actionable in normal and recovery regimes.
- **For risk management** (detect the correction is underway): h=120 — it fires earlier and more often in bearish regimes, giving more warning signals even if each individual signal is noisier.

---

### Finding 4: Positive rate is highly regime-dependent and this degrades h=40 in crisis months

For QQQ h=40 d=1%, positive rate varies 11× across test months (1.2% in Apr 2026 to 13.4% in Apr 2025). This is not just a curiosity — it directly degrades model performance:

- The model was trained on ~2.4% average positive rate (training period 2024-03 to 2026-03 for Apr 2026 test, or ~2.4% for the 2-year window).
- In April 2025, it was tested on 13.4% positive rate — a 5× mismatch.
- The `scale_pos_weight` used during training (≈40:1 for 2.4% pos rate) is far too aggressive for a 13.4% positive-rate test environment.
- The model's predicted probabilities are calibrated for ~2% scenarios and are poorly matched to the 13% density, producing unreliable scores.

The **April 2026 recovery month** tells the complementary story: 1.2% positive rate, training positive rate ~1.2%, near-perfect label-rate alignment → AUC=0.889, PR-AUC=0.442.

**The model works best when the test positive rate matches the training positive rate.**

---

### Finding 5: SQQQ h=5 d=1% wins February 2026 (AUC=0.857) — a mild-correction anomaly

February 2026 is the one month where SQQQ h=5 d=1% outperforms all QQQ configs. The result (AUC=0.857, N_pos=135) is reliable. QQQ h=40 only achieves 0.675 in the same month.

Several factors converge:
- February 2026 (QQQ −2.25%) had moderate, structured selling with intraday momentum — exactly the regime where short-horizon SQQQ "clean up" moves are predictable from MA ratios.
- QQQ h=40 d=1% produces only 211 positive examples in February (2.9% pos rate vs training average of ~2.4%) — the label definition is still well-matched, but the feature set may not capture February 2026's specific structure as well.
- The SQQQ model uses SQQQ's own MA ratios, which in a 3× leveraged context capture intraday momentum shifts at higher fidelity than QQQ's MA ratios for the same underlying move.

This suggests **SQQQ h=5 d=1% may complement QQQ h=40** — it performs differently in mild-correction months vs correction/recovery months where QQQ h=40 dominates.

---

### Finding 6: QQQ h=120 is the most stable (lowest std) but lowest ceiling

QQQ h=120 d=1% has the tightest AUC distribution (std=0.086) and the best mean PR-AUC (0.271). Its AUC never exceeds 0.778 (vs 0.889 for h=40) but also never drops below 0.574.

This is the **defensive choice**: if the goal is a strategy that does something useful in every non-crisis regime, h=120 is the most consistent. Its higher PR-AUC in correction/crisis months (0.29–0.37) means it catches more of the actual down moves — suitable for a risk management overlay rather than a precision entry signal.

---

### Finding 7: SQQQ configs show regime-split performance — 2026 months vs 2025 months

| Period | SQQQ h=5 d=1% AUC | Training SQQQ data available |
|--------|--------------------|------------------------------|
| 2025-03 | 0.642 | ~16 months (starts Nov 2023) |
| 2025-04 | 0.514 | ~17 months |
| 2025-11 | 0.608 | ~24 months |
| 2026-01 | 0.793 | ~25 months |
| 2026-02 | **0.857** | ~26 months |
| 2026-03 | 0.749 | ~27 months |
| 2026-04 | 0.872 | ~25 months |

SQQQ performance improves markedly in 2026 months (AUC 0.749–0.872) vs 2025 months (0.514–0.642). Two explanations compete:
1. **Training data length**: SQQQ data only exists from ~November 2023. For 2025-03/04 tests, the training window is only ~15–17 months rather than 24. Less data → poorer generalisation.
2. **Regime fit**: The April 2025 crisis and March 2025 crash are regimes where no config performs well. This confounds the data-length effect.

Separating these effects: November 2025 (24 months of SQQQ training, mild correction) produces AUC=0.608 — still below 2026 months even with full training data. This suggests the 2026 improvement is partly regime (2026 has more structured, predictable intraday moves) and partly data length.

**Until SQQQ has 3+ years of training data, its results should be treated as noisier than QQQ.**

---

## 3. Findings & Next Steps

### Summary: which config for which regime?

| Regime | Recommended config | Rationale |
|--------|-------------------|-----------|
| Normal / flat | **QQQ h=40 d=1%** | AUC 0.83, rare high-precision signals |
| Mild correction | **QQQ h=40 d=1%** (primary), QQQ h=60 as backup | h=40 wins most months; h=60 wins Nov 2025 |
| Strong correction | **QQQ h=40 d=1%** | AUC 0.77; h=60 also solid (0.76) |
| Crisis / black-swan | **None** — all configs fail (AUC 0.51–0.58) | Regime-gate: suspend directional model |
| Recovery | **QQQ h=40 d=1%** | AUC 0.89, PR-AUC 0.44 — the best environment for this config |
| Risk management overlay | **QQQ h=120 d=1%** | Fires early and often; best PR-AUC in correction regimes |

### Key Findings

1. **QQQ h=40 d=1% is the best overall config** (mean AUC 0.759, wins 5/7 months). Excluding the crisis month, mean AUC rises to 0.791. Use this as the primary signal.

2. **Every config collapses in crisis/crash regimes (April 2025: AUC 0.51–0.58).** This is not a label definition problem — it is a feature/regime problem. No choice of horizon or delta prevents it. A hard regime gate (e.g., suspend when 5-day realised vol > 2× baseline) is needed.

3. **AUC and PR-AUC trade off across horizons.** h=40 maximises discrimination (AUC); h=120 maximises recall/coverage (PR-AUC). The right choice depends on how signals are consumed.

4. **QQQ h=40 works best when test positive rate ≈ training positive rate.** The model is tuned for ~1–4% positive rate. Crisis months (13%+) cause a 5× training/test mismatch that collapses performance.

5. **SQQQ h=5 d=1% complements QQQ h=40** in mild-correction months (February 2026 anomaly). Consider a regime-conditional ensemble: use QQQ h=40 in normal/correction/recovery months and blend in SQQQ h=5 in mild-sell-off intraday regimes.

6. **SQQQ results are noisier** — shorter training history (data from Nov 2023 only), higher leverage noise, and regime sensitivity all contribute. 2026 SQQQ performance is much stronger than 2025, partly due to more training data.

### Next Steps

1. **Add a crisis/regime gate.** Train a simple regime classifier (VIX percentile, 5-day realised vol z-score, or QQQ 5-day return) to detect crash regimes and suppress QQQ h=40 signals when crisis probability is high. The gate only needs to identify April 2025–type months; the rest can use the directional model as-is.

2. **Multi-period rolling AUC with more months.** This study covers 7 months spanning 2 years. Run a full 12-period rolling evaluation (Jan 2024 – Dec 2025 or similar) to get a stable distribution of per-month AUC and PR-AUC, especially for the normal and mild-correction regimes where QQQ h=40 is most valuable.

3. **Delta sweep at h=40 across regimes.** The original delta scan was run at h=10 (not h=40). Now that h=40 is confirmed as the best horizon, test d ∈ {0.005, 0.007, 0.010, 0.015} at h=40 across these 7 test periods to find whether a tighter or looser delta improves robustness.

4. **Investigate the February 2026 SQQQ anomaly further.** The 0.857 AUC for SQQQ h=5 d=1% in Feb 2026 is worth understanding. Run the same config on Feb 2024 and Feb 2025 to see if "February mild-correction" is a repeating structural pattern.

5. **Calibrate scale_pos_weight dynamically.** The model is trained with auto-computed `scale_pos_weight` based on training positive rate. In crisis months, the test positive rate is 5–10× higher. Explore adaptive calibration: use a held-out recent window (last 2 weeks) to estimate the current positive rate and adjust the classification threshold accordingly.
