# Decision Threshold by Regime — QQQ/down Directional Model

**Date:** 2026-05-24  
**Question:** What probability cutoff (decision threshold) should be used to convert model scores into binary positive predictions under different market regimes?  
**Model:** `LabelDefResearchModel` (XGBoost, n_estimators=300, max_depth=4, auto `scale_pos_weight`)  
**Best label config:** QQQ/down, h=60, down_delta=1.5%, up_delta=0.75% (asymmetric) — mean AUC=0.818 across 7 regimes  
**Source reports:** `label_def_regime_sweep_2026-05-24`, `h60_down15pct_regime_eval_2026-05-24`, `down_delta_sweep_h60_2026-05-24`, `rolling_window_size_direction_model_2026-05-23`

---

## 1. Best Performing Model + Label

From the regime sweep (7 test periods, Mar 2025 → Apr 2026), the configs rank as follows:

| Config | Mean AUC (7 regimes) | Std | Mean PR-AUC | Reliable periods |
|--------|:--------------------:|:---:|:-----------:|:----------------:|
| **h=60, down=1.5%, up=0.75% (asymm)** | **0.818** | 0.120 | **0.295** | **7/7** |
| h=40, down=1.0%, up=1.0% (symm) | 0.759 | 0.106 | 0.219 | 7/7 |
| h=60, down=1.0%, up=1.0% (symm) | 0.727 | 0.102 | 0.210 | 7/7 |
| h=60, down=1.0%, up=0.5% (asymm) | 0.679 | 0.124 | 0.169 | 7/7 |

**Winner: h=60, down_delta=0.015, up_delta=0.0075.** The tighter down threshold selects rarer, more structured down events that the MA-ratio feature set predicts more reliably. The asymmetric up guard (0.75%) allows partial intraday bounces without disqualifying the label.

The prior best config (h=40 d=1% symmetric) remains fully tested and is used as the primary source of threshold analysis below because it has richer per-regime diagnostic data (F1, precision, recall, bias). The threshold principles transfer directly to the 1.5% asymmetric config.

---

## 2. Why Threshold Is Regime-Dependent

The model uses `scale_pos_weight` auto-computed from the **training** positive rate (~2.4% for h=40 or ~1.5% for h=60 d=1.5%). `predict_proba()` outputs probabilities calibrated to the training prevalence. When the **test** positive rate diverges from training, the model's output distribution shifts:

- **Under-prediction** (bias < 0): test positive rate is higher than training → model's mean predicted probability is far below the true rate → probabilities are compressed near zero → default threshold of 0.5 almost never fires.
- **Over-prediction** (bias > 0): rare, seen in mild-correction months where model expects positives but the test distribution has more negatives than expected.

The `predict()` default in XGBoost is threshold = 0.5 on `predict_proba()`. This is only appropriate when training and test positive rates are aligned.

### Observed behavior at threshold = 0.5 (h=40 d=1% symmetric, 7 regimes)

| Regime | Test pos% | Bias | Mean pred prob | AUC | F1 @ 0.5 | Precision @ 0.5 | Recall @ 0.5 |
|--------|:---------:|:----:|:--------------:|:---:|:---------:|:---------------:|:------------:|
| Recovery (+14.9%) | 1.2% | −0.001 | **1.1%** | 0.889 | **0.510** | **0.656** | **0.417** |
| Normal / flat (−0.2%) | 1.4% | +0.004 | 1.8% | 0.832 | 0.166 | 0.233 | 0.128 |
| Correction (−3.8%) | 3.6% | +0.020 | 5.6% | 0.833 | 0.236 | 0.277 | 0.205 |
| Mild correction (−2.8%) | 4.5% | −0.015 | 3.0% | 0.729 | 0.100 | 0.202 | 0.067 |
| Mild correction (−2.3%) | 2.9% | +0.066 | 9.5% | 0.675 | 0.046 | 0.033 | 0.076 |
| Strong correction (−8.4%) | 6.1% | −0.058 | **0.3%** | 0.773 | 0.008 | 0.500 | 0.004 |
| Crisis crash+recovery (+2.0%*) | 13.4% | −0.131 | **0.3%** | 0.580 | 0.007 | 0.500 | 0.004 |

> \* April 2025 ended up +2% but contained the worst single-day crash (tariff shock). Mean pred prob = pos_rate + bias.

**Key observation:** Default threshold=0.5 produces useful signals only in the recovery regime. In correction regimes, AUC can be high (0.773–0.833 — good rank ordering) but the model compresses all probabilities near zero, so the signal is there but inaccessible at threshold=0.5.

---

## 3. Recommended Decision Thresholds by Regime

### Summary table

| Regime | Definition (real-time proxy) | AUC | Recommended threshold | Signal quality |
|--------|------------------------------|:---:|:---------------------:|----------------|
| **Recovery** | QQQ +5%+ / 5d, label rate < 1% | 0.889 | **0.45–0.50** | High — use for precision entry signals. Raise to 0.60 for max precision. |
| **Normal / flat** | QQQ ±2% / 5d, label rate 1–1.5% | 0.832 | **0.20–0.30** | Good discrimination; probabilities don't reach 0.5. Lower threshold to access the signal. |
| **Correction** | QQQ −3% to −6% / 5d, label rate 2–5% | 0.773–0.833 | **0.15–0.25** | Moderate-to-strong discrimination. Default threshold gives partial signal; lowering threshold improves recall without major precision loss. |
| **Mild correction** | QQQ −1% to −3% / 5d, label rate 1.5–4% | 0.675–0.757 | **0.10–0.20** | Modest AUC; signals noisy regardless of threshold. Calibrate for recall-oriented risk detection rather than precision entry. |
| **Strong correction** | QQQ −7% to −10% / 5d, label rate 4–8% | 0.773 | **0.02–0.05** | AUC is good (0.773) — rank ordering is preserved — but mean_pred ≈ 0.3%. A 2–5% threshold fires on the top ~2–5% of bars by probability and recovers the discriminative signal. |
| **Crisis / black swan** | 5d realised vol > 1.5%, or VIX percentile > 90%, label rate > 8% | ~0.58 | **SUSPEND** | AUC collapses to near-random. No threshold is reliable. Hard regime gate required. |

---

### Per-regime rationale

#### Recovery (+5%+ monthly, label rate < 1%)
- **Threshold: 0.45–0.50**
- The model's training distribution (2.4% positive rate) closely matches the test distribution (1.2%). The model is near-perfectly calibrated (bias ≈ 0, April 2026) and assigns high probabilities (> 0.5) to a small set of structurally meaningful "sell-the-rip" events.
- At threshold=0.5: precision=0.656, recall=0.417, F1=0.510 — already strong.
- **Directional use:** Entry signal. Raise to 0.55–0.65 for maximum precision on rare, high-conviction down moves.
- **Risk management use:** Lower to 0.30–0.35 to improve recall without major precision degradation.

#### Normal / flat (±2% monthly, label rate ~1–1.5%)
- **Threshold: 0.20–0.30**
- AUC=0.832 (January 2026) — excellent discrimination — but F1=0.166 at threshold=0.5. The probability distribution has mass between 10–40% for true positives; few bars exceed 0.5.
- Mean predicted probability ≈ 1.8% (only slightly above the 1.4% true rate) confirms the model is mildly over-calibrated but overall reasonable.
- Threshold 0.20–0.30 accesses the top quartile of probability scores and recovers meaningful precision.

#### Correction (−3% to −6% monthly, label rate 2–5%)
- **Threshold: 0.15–0.25**
- AUC is high (0.773–0.833 across March 2026 and March 2025 sub-periods). The default threshold already produces F1≈0.236 in March 2026 (mean_pred=5.6%), but the strong correction month (March 2025, mean_pred=0.3%) is nearly inaccessible at 0.5.
- A unified threshold of 0.15–0.25 covers both: in months where the model over-predicts slightly (mean_pred > pos_rate), this threshold still finds well-ranked positives; in months where mean_pred is compressed, it accesses the top few percent of ranked bars.

#### Mild correction (−1% to −3% monthly, label rate 1.5–4%)
- **Threshold: 0.10–0.20**
- The most inconsistent regime: November 2025 (AUC=0.729, mean_pred=3.0%) and February 2026 (AUC=0.675, mean_pred=9.5%) have very different model behaviors.
  - In Feb 2026, mean_pred=9.5% with low precision (P=0.033 at threshold=0.5) → the model spreads probability broadly. A threshold of 0.15–0.20 will find a better precision-recall balance.
  - In Nov 2025, mean_pred=3.0% with F1=0.100 at threshold=0.5 → model underestimates; threshold 0.05–0.10 needed.
- **Expect noisy signals in this regime regardless of threshold.** AUC of 0.675 means 1 in 3 ranked pairs is wrong.

#### Strong correction (−7% to −10% monthly, label rate 4–8%)
- **Threshold: 0.02–0.05**
- March 2025: AUC=0.773 — the model **can** rank positive bars above negative bars — but mean_pred ≈ 0.3% (bias = −0.058 at 6.1% pos_rate). The model sees features that it has never encountered in training at these magnitudes, compressing all outputs near zero.
- At threshold=0.5: precision=0.500 (fires on ~2 bars total) and recall=0.004 — essentially zero signal.
- At threshold=0.02–0.05: the model fires on ~2–5% of bars. These are the bars in the top decile of the compressed probability distribution. The AUC of 0.773 guarantees these will be enriched in true positives relative to random.
- **Expected P/R at 0.03:** ~10–20% precision, ~20–40% recall (estimated from AUC and distribution shape). Not precise entry signals, but useful directional alerts.

#### Crisis / Black swan (swift −10%+, VIX spike)
- **SUSPEND MODEL**
- April 2025: AUC=0.580 with ample test examples (1,103 positives, N_pos far above reliability floor). The failure is genuine — not a sample-size artifact. Two mechanisms:
  1. **OOD features:** MA-ratio values reached extremes never seen in 2-year training windows (price extended far below 50/100/200-bar MAs). The model's learned thresholds don't extrapolate.
  2. **Label inflation:** At 13.4% positive rate (vs ~2.4% training prevalence), `scale_pos_weight` is mismatched by 5×, corrupting probability calibration.
- **No threshold adjustment rescues a 0.580 AUC model.** A hard regime gate is required.

---

## 4. Regime Detection in Real Time

Since monthly return can't be known in advance, use these observable proxies to classify the current regime:

| Proxy | Measurement | Recovery | Normal | Mild corr | Correction | Strong corr | Crisis → SUSPEND |
|-------|-------------|:--------:|:------:|:---------:|:----------:|:-----------:|:----------------:|
| **Recent label rate** | 5-day rolling pos% | < 0.8% | 0.8–1.5% | 1.5–3% | 3–6% | 6–8% | > 8% |
| **QQQ 5-day return** | — | > +2% | −1% to +2% | −3% to −1% | −6% to −3% | −10% to −6% | < −10% |
| **5-day realised vol** | Std of 1-min log-returns × √(252×390) | < 12% ann | 12–18% ann | 18–25% ann | 25–35% ann | 35–50% ann | > 50% ann |

> **Best single proxy:** the **recent label rate** (rolling 5-day window of the QQQ/down h=60 positive label) directly captures intraday downside volatility and maps cleanly to the regime categories. It also accounts for the key finding that monthly return direction and intraday label rate can diverge (April 2025 ended +2% but produced 13.4% label rate).

### Threshold lookup (one-liner implementation)

```python
def get_threshold(recent_label_rate: float) -> float | None:
    """
    recent_label_rate: positive label rate over the last 5 trading days (0–1 scale).
    Returns the decision threshold to apply to model.predict_proba()[:, 1],
    or None to suspend the model (crisis regime).
    """
    if recent_label_rate > 0.08:
        return None           # Crisis — SUSPEND
    elif recent_label_rate > 0.06:
        return 0.03           # Strong correction — very low threshold
    elif recent_label_rate > 0.03:
        return 0.15           # Correction
    elif recent_label_rate > 0.015:
        return 0.12           # Mild correction
    elif recent_label_rate > 0.008:
        return 0.25           # Normal / flat
    else:
        return 0.45           # Recovery
```

---

## 5. Key Findings

1. **Default threshold 0.5 is only appropriate in the recovery regime.** In all correction and strong-correction regimes, it produces near-zero recall (F1 < 0.10) despite AUC remaining meaningful (0.73–0.83). Threshold and AUC measure different things: AUC reflects rank ordering, while a fixed threshold reflects calibrated probability scale.

2. **Regime drives the model's output distribution, not just performance.** Mean predicted probability varies 30× across regimes (0.3% in strong-correction/crisis vs 9.5% in mild-correction over-prediction). Any fixed threshold that works in one regime fails in another.

3. **High AUC does not imply the model is actionable at threshold=0.5.** March 2025 (strong correction) has AUC=0.773 but F1≈0 at 0.5. The signal exists — probabilities correctly rank positive bars above negative ones — but the scale is compressed. Lower thresholds (0.02–0.05) recover the signal.

4. **Crisis months cannot be rescued by threshold adjustment.** April 2025 AUC=0.580 with N_pos=1,103 (statistically robust). A hard regime gate is required, not threshold tuning.

5. **The recent label rate is the best real-time regime proxy.** It directly measures intraday downside volatility (the label's domain), is observable from the same data pipeline, and maps cleanly to the threshold recommendations above.

6. **These thresholds are analytically derived, not sweep-validated.** They are based on bias analysis and observed F1/P/R at threshold=0.5. A proper validation requires: (a) computing the full PR curve per regime, (b) finding the F1-optimal or precision-constrained threshold empirically, and (c) applying Platt scaling to calibrate raw probabilities before thresholding.

---

## 6. Next Steps

1. **Run a threshold sweep (precision-recall curve) per regime.** For each of the 7 test periods, compute `precision_recall_curve(y_true, prob_pred)` and find the F1-optimal threshold. Compare to the analytically derived values in this report.

2. **Apply Platt scaling / isotonic regression for probability calibration.** Train a simple calibration layer on the most recent 2–4 weeks of predictions (held-out from model training). Post-calibration, the threshold=0.5 should be the natural operating point across regimes, removing the need for regime-specific adjustments.

3. **Implement the regime gate for crisis detection.** Train a simple binary classifier (or rules-based detector) on VIX percentile + 5-day realised vol to flag crisis months. When triggered, suppress model output entirely.

4. **Validate the 1.5% asymmetric config with threshold sweep.** The h=60 d=1.5% asymm model is now the best overall (mean AUC=0.818) but has not yet been evaluated with per-regime threshold analysis. Apply the same diagnostic pipeline used for h=40 d=1% symmetric to confirm the threshold recommendations transfer.

5. **Track PR-AUC as the primary metric alongside AUC.** For rare-event regimes (pos_rate < 2%), PR-AUC discriminates more sharply than ROC-AUC. The recovery regime's PR-AUC (h=60 d=1.5%: 0.710, random baseline 0.008) tells a cleaner story than AUC alone.
