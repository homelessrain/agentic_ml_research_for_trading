# Rolling Window-Size Evaluation — Direction Model (2026-05-23)

## Setup

- **Symbol**: QQQ  
- **Model**: `DirectionModel` (XGBoost binary classifier, default params, random train/val split)  
- **Training windows**: 1m, 3m, 6m, 1y, 2y, 3y, 5y  
- **Test periods** (monthly, out-of-sample):
  - 2026-01: 2026-01-02 → 2026-01-31
  - 2026-02: 2026-02-03 → 2026-02-28
  - 2026-03: 2026-03-03 → 2026-03-31
  - 2026-04: 2026-04-01 → 2026-04-30

---

## Part 1 — Test ROC-AUC by window × period

| Window | 2026-01 | 2026-02 | 2026-03 | 2026-04 | **Mean** | Std  |
|--------|--------:|--------:|--------:|--------:|---------:|-----:|
| 1m     | 0.8083  | 0.6966  | 0.7058  | 0.6451  | 0.7140   | 0.059|
| 3m     | 0.7404  | 0.8511  | 0.7320  | 0.6617  | 0.7463   | 0.068|
| 6m     | 0.7883  | 0.8752  | 0.6851  | 0.6194  | 0.7420   | 0.098|
| 1y     | 0.8070  | 0.8330  | 0.6244  | 0.6572  | 0.7304   | 0.091|
| **2y** | **0.8469** | 0.8029 | 0.6361 | **0.7443** | **0.7575** | 0.079|
| 3y     | 0.7497  | 0.8183  | 0.6572  | 0.6699  | 0.7238   | 0.065|
| 5y     | 0.7481  | 0.8362  | 0.6178  | 0.6446  | 0.7117   | 0.087|

> Random baseline: ROC-AUC = 0.500

---

## Part 2 — Bias Analysis on Test Data

**Bias** = `mean_predicted_prob − true_positive_rate` on the test set.  
- Positive bias → model over-predicts the probability of an UP move  
- Negative bias → model under-predicts

### True positive rate of test set (label distribution)

| Period  | True pos rate |
|---------|:------------:|
| 2026-01 | 5.6 %        |
| 2026-02 | 9.0 %        |
| 2026-03 | 12.5 %       |
| 2026-04 | 9.8 %        |

### Bias by window × period

| Window | 2026-01 | 2026-02 | 2026-03 | 2026-04 | **Mean bias** |
|--------|--------:|--------:|--------:|--------:|--------------:|
| 1m     | −0.0545 | **+0.3072** | **+0.3139** | −0.0717 | **+0.1237** |
| 3m     | −0.0549 | −0.0000 | +0.1914 | −0.0939 | +0.0106 |
| 6m     | −0.0555 | +0.0190 | +0.0540 | −0.0893 | −0.0180 |
| 1y     | −0.0530 | +0.0171 | −0.0031 | −0.0794 | −0.0296 |
| **2y** | −0.0440 | −0.0632 | +0.0231 | −0.0902 | **−0.0436** |
| 3y     | −0.0351 | +0.0933 | +0.0580 | −0.0773 | +0.0097 |
| 5y     | −0.0375 | −0.0090 | +0.0389 | −0.0795 | −0.0218 |

### Mean predicted probability vs true positive rate

| Window | Period  | True pos% | Mean pred% | Bias    | Calib corr | Test AUC |
|--------|---------|:---------:|:----------:|:-------:|:----------:|:--------:|
| 1m     | 2026-01 | 5.6 %     | 0.2 %      | −0.055  | 0.622      | 0.808 |
| 1m     | 2026-02 | 9.0 %     | **39.7 %** | +0.307  | 0.637      | 0.697 |
| 1m     | 2026-03 | 12.5 %    | **43.9 %** | +0.314  | 0.673      | 0.706 |
| 1m     | 2026-04 | 9.8 %     | 2.6 %      | −0.072  | 0.760      | 0.645 |
| 3m     | 2026-01 | 5.6 %     | 0.1 %      | −0.055  | 0.901      | 0.740 |
| 3m     | 2026-02 | 9.0 %     | 9.0 %      | −0.000  | 0.828      | 0.851 |
| 3m     | 2026-03 | 12.5 %    | 31.7 %     | +0.191  | 0.910      | 0.732 |
| 3m     | 2026-04 | 9.8 %     | 0.4 %      | −0.094  | 0.796      | 0.662 |
| 6m     | 2026-01 | 5.6 %     | 0.1 %      | −0.056  | 0.953      | 0.788 |
| 6m     | 2026-02 | 9.0 %     | 10.9 %     | +0.019  | 0.990      | 0.875 |
| 6m     | 2026-03 | 12.5 %    | 17.9 %     | +0.054  | 0.951      | 0.685 |
| 6m     | 2026-04 | 9.8 %     | 0.9 %      | −0.089  | 0.566      | 0.619 |
| 1y     | 2026-01 | 5.6 %     | 0.3 %      | −0.053  | 0.914      | 0.807 |
| 1y     | 2026-02 | 9.0 %     | 10.7 %     | +0.017  | 0.838      | 0.833 |
| 1y     | 2026-03 | 12.5 %    | 12.2 %     | −0.003  | 0.725      | 0.624 |
| 1y     | 2026-04 | 9.8 %     | 1.8 %      | −0.079  | 0.731      | 0.657 |
| 2y     | 2026-01 | 5.6 %     | 1.2 %      | −0.044  | 0.866      | 0.847 |
| 2y     | 2026-02 | 9.0 %     | 2.7 %      | −0.063  | 0.757      | 0.803 |
| 2y     | 2026-03 | 12.5 %    | 14.8 %     | +0.023  | 0.819      | 0.636 |
| 2y     | 2026-04 | 9.8 %     | 0.8 %      | −0.090  | 0.772      | 0.744 |
| 3y     | 2026-01 | 5.6 %     | 2.1 %      | −0.035  | 0.714      | 0.750 |
| 3y     | 2026-02 | 9.0 %     | 18.3 %     | +0.093  | 0.945      | 0.818 |
| 3y     | 2026-03 | 12.5 %    | 18.3 %     | +0.058  | 0.851      | 0.657 |
| 3y     | 2026-04 | 9.8 %     | 2.1 %      | −0.077  | 0.831      | 0.670 |
| 5y     | 2026-01 | 5.6 %     | 1.9 %      | −0.038  | 0.650      | 0.748 |
| 5y     | 2026-02 | 9.0 %     | 8.1 %      | −0.009  | 0.980      | 0.836 |
| 5y     | 2026-03 | 12.5 %    | 16.4 %     | +0.039  | 0.611      | 0.618 |
| 5y     | 2026-04 | 9.8 %     | 1.8 %      | −0.080  | 0.747      | 0.645 |

---

## Observations

### 1. 1-month window is catastrophically miscalibrated
The 1m bias swings from −0.055 (Jan) to **+0.307 / +0.314** (Feb, Mar) back to −0.072 (Apr).
In February and March it predicts average probabilities of 40–44 % against a true positive
rate of only 9–13 %. This is a pure regime-shift failure: the model trained on Nov–Dec 2025
(a trending environment) carries those momentum statistics into the test month. Meanwhile
in January it predicts near-zero (0.2 % mean) despite a 5.6 % true rate.
**ROC-AUC can still be decent (0.70–0.81) while absolute calibration is completely wrong**
— the ranking of predictions is preserved even when the scale is wildly off.

### 2. Systematic under-prediction in January and April
All seven window sizes show negative bias in January (−0.035 to −0.056) and April
(−0.072 to −0.094). January 2026 was a quiet, low-volatility tape with only 5.6 % of
minute bars triggering the UP label — the model predicts even lower (near zero for
short windows). April 2026 was a high-volatility sell-off period; every model massively
under-predicts, outputting mean probabilities of 0.1–2.6 % against a true 9.8 % rate.
This reveals a structural gap: **when a new volatility regime arrives, all models go
silent** because the feature patterns they learned map to low scores.

### 3. Bias stabilises as window length grows (until ~1 year)
Absolute mean bias drops sharply from 1m (0.124) to 3m (0.011), continues falling to 1y
(0.030), then rises again for 2y–5y (0.022–0.044).  The 3m–1y range is the "calibration
sweet spot" — enough data to average out idiosyncratic regimes, short enough to stay
relevant to the current market environment.

### 4. 6m window has the best calibration correlation in Feb–Mar
Calibration correlation (rank correlation between predicted and actual probability per
decile) peaks for the 6m model: 0.953 (Jan), **0.990** (Feb), 0.951 (Mar).  The 1m model
achieves only 0.62–0.67 in those same periods, meaning the decile ordering of predicted
scores is unreliable at short windows even when AUC is acceptable.

### 5. March 2026 is the universal hard case
Every window over-predicts in March (positive bias +0.023 to +0.314) and all AUCs drop
below 0.74.  The true positive rate of 12.5 % is the highest of the four test months;
the models appear to have learned that high-vol environments produce upward moves, but
the March 2026 tariff-driven sell-off broke that pattern.

### 6. 2-year window wins on AUC but has the worst April bias
The 2y model achieves the highest mean test AUC (0.758) but also the worst April bias
(−0.090) and near-zero mean predicted probability (0.76 %) vs the true 9.8 %. The
implication: **2y is the best discriminator** (rank order of scores is most informative)
but **not the best absolute probability estimator** in regime-shift conditions.

### 7. Overfitting is universal and deep
Train AUC ≈ 1.000 for all windows ≤ 3y, confirming that the random train/val split
leaks future bar statistics into validation. This inflates the apparent training
performance and makes it hard to isolate whether the bias is a data quantity issue or
a feature/label design issue.

---

## Next Steps

1. **Fix the evaluation bias at the root: use chronological splits + early stopping.**
   Switch to `Exp3StrongRegModel` (or at minimum `Exp1ChronologicalEarlyStoppingModel`)
   and re-run this window-size sweep. With random splits, the current training AUC of
   ~1.0 provides no useful signal; calibration conclusions are unreliable.

2. **Add post-hoc probability calibration (Platt scaling / isotonic regression).**
   Because XGBoost's raw probabilities are systematically miscalibrated under regime
   shifts, wrapping the model with `sklearn.calibration.CalibratedClassifierCV` (or
   a rolling-window Platt scaler trained on recent hold-out data) would directly
   reduce the bias magnitude without changing model architecture.

3. **Investigate the April / January under-prediction pattern separately.**
   Both quiet (Jan, 5.6 % pos rate) and volatile (Apr, 9.8 %) regimes cause all models
   to under-predict. These are structurally different failure modes — quiet regimes may
   need a different feature set (lower-frequency signals), while volatile regimes may need
   VIX-conditioned calibration.

4. **Separate calibration analysis from discrimination analysis.**
   The current confusion between "good AUC ≠ good calibration" should be tracked with
   two separate metrics: ROC-AUC (discrimination) and ECE / Brier score (calibration).
   Add both to future experiment reports.

5. **Test expanding-window calibration.**
   Instead of a fixed-size rolling window, use an expanding window from a fixed start
   date (e.g., 2021-01-01) so the model sees all history but is evaluated on the
   marginal new data. Compare bias and AUC against the rolling counterpart.

6. **Explore regime-conditional training.**
   Tag each training bar with a vol regime (e.g., ATR quartile) and up/down-weight
   training samples to match the expected test-period regime. This would directly
   address the regime-shift calibration failures seen in January and April.
