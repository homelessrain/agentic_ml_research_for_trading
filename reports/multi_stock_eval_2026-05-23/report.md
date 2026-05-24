# Multi-Stock Evaluation — MA Ratios + Time Features Baseline

**Date:** 2026-05-24  
**Model:** `DirectionModel` with updated `OLHCVFeatureTransformer` (MA ratios + time features only; 124 features)  
**Label:** `DirectionLabelTransformer` — direction-up 1% / fwd-120 bars  
**Setup:** 2-year rolling training window, 4 monthly OOS test periods (Jan–Apr 2026), identical to prior QQQ experiments  
**Feature set:** 60 MA-ratio features (5 OHLCV cols × 6 MA windows × 2 variants) + 2 time features (hour_of_day, day_of_week) × 2 timeframes (minute + daily) = 124 total

---

## Results

| Category | Symbol | Mean AUC | Jan-26 | Feb-26 | Mar-26 | Apr-26 | Std | Pos rate | Train AUC |
|----------|--------|:--------:|-------:|-------:|-------:|-------:|:---:|:--------:|:---------:|
| ETF | **SPY** | **0.855** | 0.893 | 0.941 | 0.782 | 0.805 | 0.065 | 4.5% | 1.000 |
| ETF | QQQ *(ref)* | 0.769 | 0.808 | 0.660 | 0.749 | 0.858 | 0.082 | 9.2% | 0.999 |
| ETF | IWM | 0.672 | 0.751 | 0.620 | 0.625 | 0.692 | 0.054 | 13.0% | 0.999 |
| ETF | SQQQ | 0.498 | 0.423 | 0.525 | 0.541 | 0.505 | 0.045 | 33.7% | 0.983 |
| | | | | | | | | | |
| Large-cap | AAPL | 0.684 | 0.696 | 0.692 | 0.655 | 0.692 | 0.017 | 15.0% | 0.997 |
| Large-cap | MSFT | 0.634 | 0.610 | 0.652 | 0.651 | 0.622 | 0.018 | 17.8% | 0.999 |
| Large-cap | NVDA | 0.640 | 0.672 | 0.605 | 0.638 | 0.646 | 0.024 | 22.8% | 0.986 |
| | | | | | | | | | |
| Medium-cap | SNAP | 0.534 | 0.489 | 0.593 | 0.479 | 0.574 | 0.050 | 34.8% | 0.981 |
| Medium-cap | ROKU | 0.541 | 0.516 | 0.521 | 0.536 | 0.592 | 0.030 | 34.4% | 0.980 |
| Medium-cap | DKNG | 0.525 | 0.520 | 0.520 | 0.505 | 0.554 | 0.018 | 31.9% | 0.983 |
| | | | | | | | | | |
| Small-cap | WOLF | 0.528 | 0.542 | 0.500 | 0.557 | 0.512 | 0.023 | 30.9% | 0.973 |
| Small-cap | XPEL | 0.627 | 0.737 | 0.693 | 0.534 | 0.545 | 0.089 | 37.5% | 0.994 |
| Small-cap | ARRY | 0.529 | 0.487 | 0.574 | 0.503 | 0.553 | 0.035 | 36.0% | 0.974 |

> Random baseline: ROC-AUC = 0.500

---

## Findings

### 1. Performance is driven almost entirely by the positive label rate

The single strongest predictor of model performance is the **positive label rate** — how often a 1% clean up-move occurs within 120 bars. This is a direct measure of how large the 1% threshold is relative to the stock's typical intraday volatility:

| Positive rate | Instruments | Mean AUC |
|:---:|---|:---:|
| 2–9% | SPY, QQQ | **0.77–0.86** |
| 9–15% | IWM, AAPL | **0.67–0.68** |
| 15–23% | MSFT, NVDA | **0.63–0.64** |
| 30–45% | Medium-cap, Small-cap, SQQQ | **0.50–0.54** |

When a stock hits the 1% threshold on ~35% of all bars, the label is structurally uninformative — it fires so often that predicting it amounts to predicting near-continuous up-movement, which the MA ratio and time features cannot meaningfully separate from noise. The model degenerates toward predicting the base rate.

### 2. SPY is the strongest signal (0.855), better than QQQ (0.769)

SPY's positive rate of 4.5% is the lowest in the set, making it the hardest label to fire and the cleanest signal to learn. The MA ratio and time-of-day features capture SPY's highly structured intraday behaviour — the S&P 500 ETF is dominated by institutional participants with predictable open/close activity — exceptionally well. February 2026 alone produced **AUC 0.941**.

### 3. SQQQ is at random despite being an ETF — 3× leverage destroys the label

SQQQ (ProShares 3× inverse QQQ) is technically in the ETF category but behaves nothing like SPY or QQQ from a label perspective. Its 3× daily leverage multiplies intraday volatility so aggressively that the 1% threshold is hit on 33.7% of bars — on par with medium-cap individual stocks. As a result, mean test AUC is **0.498**, indistinguishable from random.

January 2026 is especially striking: AUC **0.423**, meaningfully *below* random. During January's mild QQQ sell-off, SQQQ rallied; the model — trained on 2 years of mostly upward QQQ drift where SQQQ was trending down — learned patterns that actively pointed the wrong direction when the underlying reversed. This is the inverse-instrument regime problem: the model's learned relationship between MA-ratio structure and up-moves is the exact opposite for SQQQ vs QQQ, and a regime flip exposes it.

**SQQQ is not a viable target for this label/feature combination.** To model it properly would require either (a) using QQQ features as predictors of SQQQ labels (since they are mechanically linked), or (b) reframing the label as a QQQ direction-down signal.

### 4. Large-cap tech shows real but moderate edge (0.63–0.68)

AAPL, MSFT, NVDA all sit well above random, confirming the MA ratio + time features carry genuine signal for individual large-cap names. AAPL is notably the most **consistent** (std 0.017, range 0.655–0.696 across all four months) — its intraday structure is the most institutionally regular of the three. NVDA's higher volatility (22.8% pos rate) and idiosyncratic news-driven moves limit performance relative to AAPL.

### 5. Medium-cap and most small-caps are effectively unlearnable with this label definition

SNAP, ROKU, DKNG, WOLF, and ARRY cluster between 0.525–0.541 — right at the noise floor. Two issues compound:
- **High positive rates** (30–44%) reduce label informativeness
- **Idiosyncratic volatility** (earnings shocks, news events, thin order books) creates price moves that MA-ratio features have no way to anticipate

**XPEL is the outlier**: 0.627 mean AUC despite a 37.5% positive rate. Its Jan/Feb performance (0.737, 0.693) shows genuine signal, which then collapses in March/April (0.534, 0.545). XPEL has far fewer training rows (~80K vs 200K+ for others) because it is lightly traded — the model may be fitting to idiosyncratic XPEL-specific patterns rather than general ones, and those patterns don't generalise across regime shifts. The high variance (std 0.089) supports this interpretation.

### 6. The MA-ratio + time feature set is purpose-built for liquid broad-market instruments

The consistent pattern: MA ratios measure where price sits relative to recent trend, and time-of-day features encode the open/close participation structure. These two signals are most powerful when:
- The instrument has a dominant, persistent trend (so MA ratios are informative)
- Intraday behaviour is structurally regular (so time features are informative)
- The label threshold is tight enough to be selective (pos rate < ~15%)

SPY and QQQ satisfy all three. Individual large-caps satisfy the first two but have higher pos rates. Medium/small-caps and leveraged ETFs satisfy none.

---

## Recommended per-symbol label thresholds

To bring positive rates into the learnable 5–15% range across all instruments, the 1% threshold needs to be scaled to each instrument's volatility:

| Symbol | Current pos rate | Suggested threshold | Target pos rate |
|--------|:---:|:---:|:---:|
| SPY | 4.5% | 0.5% | ~10% |
| QQQ | 9.2% | 1.0% | ✓ already calibrated |
| IWM | 13.0% | 1.5% | ~8% |
| AAPL | 15.0% | 1.5–2.0% | ~8–10% |
| MSFT | 17.8% | 2.0% | ~8% |
| NVDA | 22.8% | 2.5–3.0% | ~8–10% |
| SNAP/ROKU/DKNG | 32–35% | 4.0–5.0% | ~8–10% |
| SQQQ | 33.7% | model via QQQ instead | — |

Alternatively, a **vol-adaptive threshold** (`N × ATR(14)` where N is tuned so the positive rate ≈ 10%) would generalise across all symbols without manual per-symbol tuning.

---

## Next steps

1. **Re-run with vol-adaptive label thresholds.** Replace the fixed 1% with `1.5 × ATR(14, daily)` or an equivalent that normalises to each symbol's volatility. This is the single highest-impact change for extending the model across the universe.

2. **Model SQQQ via QQQ features.** Since SQQQ is mechanically derived from QQQ (−3× daily), use QQQ's MA ratios and time features as inputs to predict SQQQ's direction. This respects the mechanical link and avoids the regime-inversion problem.

3. **Deeper investigation of XPEL's Jan/Feb signal.** The strong Jan/Feb performance (0.71–0.74) warrants inspection: is it a structural signal (thin order book behaves predictably) or an in-sample fluke? Run a longer backtest (3–5 years) to assess stability.

4. **Separate feature importance analysis per category.** MA ratios for SPY may be dominated by slow MAs (50/100/200 capturing trend) while large-cap stocks may rely more on fast MAs (5/10 capturing intraday mean-reversion). A per-symbol feature importance breakdown would inform category-specific feature engineering.
