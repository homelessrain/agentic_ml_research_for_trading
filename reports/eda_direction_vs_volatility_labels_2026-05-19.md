# EDA: Direction-Based vs Volatility-Based Labels

**Date:** 2026-05-19  
**Data:** QQQ 1-minute bars, 2026-01-02 → 2026-05-01 (32,453 bars, ~83 trading days)  
**Labels:** `DirectionLabelTransformer` (up) and `VolatilityLabelTransformer` from `utils/label_transformer.py`  
**Parameter grid:** thresholds = {0.5%, 1%, 2%} × lookforward periods = {30, 60, 120} bars

---

## Label Definitions (recap)

| Label | Fires 1 when… |
|---|---|
| **Direction-up** | max gain in next N bars ≥ thresh AND max loss < thresh |
| **Direction-down** | max loss in next N bars ≥ thresh AND max gain < thresh |
| **Volatility** | max gain ≥ thresh OR max loss ≥ thresh (either direction) |

The key structural difference: direction requires an *unambiguous* move — it excludes bars where the price swings both up AND down beyond the threshold ("whipsaws"). Volatility fires whenever *any* large move occurs.

---

## 1. Class Balance (Positive Rate)

| fwd | thresh | dir_pos_rate | vol_pos_rate |
|-----|--------|-------------|-------------|
| 30  | 0.5%   | 6.2%        | 12.7%       |
| 30  | 1.0%   | 1.4%        | 2.9%        |
| 30  | 2.0%   | 0.1%        | 0.2%        |
| 60  | 0.5%   | 13.5%       | 27.6%       |
| 60  | 1.0%   | 3.7%        | 7.9%        |
| 60  | 2.0%   | 0.4%        | 0.7%        |
| 120 | 0.5%   | 25.4%       | 52.6%       |
| 120 | 1.0%   | 9.1%        | 18.7%       |
| 120 | 2.0%   | 0.8%        | 1.8%        |

**Key finding:** Volatility labels are consistently ~2× more frequent than direction labels at the same parameter settings. This makes sense structurally: volatility = direction-up ∪ direction-down ∪ whipsaws.

At 1% / fwd-120, direction fires 9.1% of the time while volatility fires 18.7% — still a very imbalanced problem, but less so than with tighter thresholds (1.4% at fwd-30).

---

## 2. Label Stability (Autocorrelation & Mean Run Length)

| fwd | thresh | dir_ac1 | vol_ac1 | dir_mean_run | vol_mean_run |
|-----|--------|---------|---------|--------------|--------------|
| 30  | 0.5%   | 0.863   | 0.868   | 7.8          | 8.7          |
| 30  | 1.0%   | 0.905   | 0.900   | 10.7         | 10.2         |
| 30  | 2.0%   | 0.884   | 0.896   | 8.6          | 9.6          |
| 60  | 0.5%   | 0.909   | 0.906   | 12.7         | 14.7         |
| 60  | 1.0%   | 0.935   | 0.938   | 15.9         | 17.4         |
| 60  | 2.0%   | 0.971   | 0.954   | 34.0         | 21.8         |
| 120 | 0.5%   | 0.933   | 0.928   | 20.1         | 29.1         |
| 120 | 1.0%   | 0.954   | 0.956   | 23.8         | 28.2         |
| 120 | 2.0%   | 0.980   | 0.950   | 51.2         | 20.5         |

**Key findings:**

- Both label types have very high lag-1 autocorrelation (0.86–0.98), driven by the rolling lookforward window — neighboring bars almost always share the same label. This is expected and is **not** predictive autocorrelation; it is a structural artifact of the rolling window.
- Autocorrelation grows with lookforward period (longer window = slower transitions).
- At high thresholds (2%) with long lookforward (120), direction labels have *longer* mean runs (51 bars) than volatility (20 bars). This happens because a rare clean 2% up-move lasts many consecutive bars of the rolling window, while a 2% volatile bar might "expire" sooner when the opposite reversal enters the window.
- **Implication:** When training, this autocorrelation means consecutive bars are not independent samples. Chronological train/test splits are essential; k-fold cross-validation is inappropriate.

---

## 3. Overlap Between Label Types

Direction label is a strict subset of volatility label (direction-up fires only when volatility also fires).

| fwd | thresh | both_pos | dir_only | vol_only | both_neg | pct vol→dir |
|-----|--------|----------|----------|----------|----------|-------------|
| 30  | 0.5%   | 2,005    | 0        | 2,113    | 28,305   | 48.7%       |
| 30  | 1.0%   | 459      | 0        | 473      | 31,491   | 49.2%       |
| 30  | 2.0%   | 43       | 0        | 34       | 32,346   | 55.8%       |
| 60  | 0.5%   | 4,369    | 0        | 4,578    | 23,446   | 48.8%       |
| 60  | 1.0%   | 1,207    | 0        | 1,368    | 29,818   | 46.9%       |
| 60  | 2.0%   | 136      | 0        | 82       | 32,175   | 62.4%       |
| 120 | 0.5%   | 8,208    | 0        | 8,812    | 15,313   | 48.2%       |
| 120 | 1.0%   | 2,931    | 0        | 3,125    | 26,277   | 48.4%       |
| 120 | 2.0%   | 256      | 0        | 339      | 31,738   | 43.0%       |

- **dir_only = 0 always:** Every direction-positive bar is also a volatility-positive bar. Direction is logically contained within volatility.
- **~50% of volatility-positive bars are NOT captured by direction-up**, across almost all settings. This "vol_only" half is where the price made an equally large move — but downward. It is captured by direction-down, not direction-up.
- The near-perfect 50/50 split (direction-up vs direction-down within volatility positives) reflects QQQ's approximate symmetry over this period.

---

## 4. What Does Volatility Capture That Direction Misses?

Decomposing each volatility-positive bar into which direction category it falls:

| fwd | thresh | up_only | down_only | both_move (whipsaw) |
|-----|--------|---------|-----------|---------------------|
| 30  | 0.5%   | 48.7%   | 51.2%     | 0.1%                |
| 30  | 1.0%   | 49.2%   | 50.8%     | 0.0%                |
| 60  | 0.5%   | 48.8%   | 50.3%     | 0.8%                |
| 60  | 1.0%   | 46.9%   | 53.1%     | 0.0%                |
| 120 | 0.5%   | 48.2%   | 47.4%     | 4.4%                |
| 120 | 1.0%   | 48.4%   | 51.6%     | 0.0%                |

- At 1%+ thresholds: whipsaw (both directions) is essentially zero. The market very rarely moves ≥1% in both directions within the same 120-minute window and still qualifies as clean.
- At 0.5% / fwd-120: ~4.4% of volatility positives are whipsaws (both directions fired). This is the one setting where volatility and direction genuinely diverge in meaning.
- **The practical conclusion:** At ≥1% threshold, choosing direction-up vs direction-down vs volatility is almost entirely a question of which *direction* you want to predict — not a question of signal quality. Volatility at ≥1% is just direction-up OR direction-down.

---

## 5. Ambiguous Zone: The "Neither" Category

Direction labeling has an inherent ambiguous zone — bars where price moves less than the threshold in both directions (labeled 0 as negative examples even though no strong signal exists).

| fwd | thresh | dir_up | dir_down | neither | both_move |
|-----|--------|--------|----------|---------|-----------|
| 30  | 0.5%   | 6.2%   | 6.5%     | **87.3%**   | 0.0%  |
| 30  | 1.0%   | 1.4%   | 1.5%     | **97.1%**   | 0.0%  |
| 30  | 2.0%   | 0.1%   | 0.1%     | **99.8%**   | 0.0%  |
| 60  | 0.5%   | 13.5%  | 13.9%    | **72.4%**   | 0.2%  |
| 60  | 1.0%   | 3.7%   | 4.2%     | **92.1%**   | 0.0%  |
| 120 | 0.5%   | 25.4%  | 24.9%    | **47.4%**   | 2.3%  |
| 120 | 1.0%   | 9.1%   | 9.7%     | **81.3%**   | 0.0%  |
| 120 | 2.0%   | 0.8%   | 1.0%     | **98.2%**   | 0.0%  |

- At strict thresholds (1–2%), the vast majority of bars are labeled negative by direction — not because they moved down, but because they barely moved at all. The negative class is dominated by "quiet" bars.
- **Implication for modeling:** A direction model is really learning "will the price make a clean unidirectional move" vs "will it stay quiet." The negative class mixes three regimes: quiet, down-move, and whipsaw. This can confuse a model if it encounters a large down-move it has never been asked to label specifically.

---

## 6. Time-of-Day Patterns (fwd=120, thresh=1%)

Both label types are strongly time-of-day dependent:

- **Open (9:30–9:45 ET):** Direction rate ~8–11%, Volatility rate ~18–24%
- **Mid-morning dip (9:45–10:00):** Rates drop (~5–10% direction, ~11–16% vol)
- **Midday (10:00–14:00):** Rates moderate and relatively stable
- **Close approach (15:30–16:00):** Direction rises to **~28%**, Volatility to **~57%** — the highest rates of the session

The top-10 minutes by either label rate all cluster in the 15:36–15:59 window. This is a mechanical artifact: bars near the close have a longer effective lookforward horizon relative to remaining session volatility, and end-of-day momentum / volume spikes increase the chance of hitting the threshold.

**Implication:** Any model trained without time-of-day features will be exposed to this regime shift. Models may implicitly learn time of day as a proxy for label rate. Stratifying train/test splits by time of day, or including time features explicitly, is important.

---

## 7. Direction Up vs Direction Down Symmetry

At all parameter settings, direction-up and direction-down fire at essentially equal rates and **never co-fire** (the mutual exclusion is enforced by design — if both conditions are met, neither fires as a positive example):

| fwd | thresh | up_rate | down_rate | co-fire |
|-----|--------|---------|-----------|---------|
| 30  | 0.5%   | 6.2%    | 6.5%      | 0       |
| 60  | 1.0%   | 3.7%    | 4.2%      | 0       |
| 120 | 1.0%   | 9.1%    | 9.7%      | 0       |

QQQ shows slight downside bias (down_rate ≥ up_rate in most settings), consistent with the early-2026 period that included market drawdowns.

---

## Summary: When to Use Each Label

| Label | Use when… | Avoid when… |
|---|---|---|
| **Direction-up** | You want to predict clean up-moves only; directional trading signals | Class imbalance is a concern at tight thresholds; you want to avoid confusing "quiet" with "bearish" |
| **Direction-down** | Short-selling / hedging signals | Same caveats as direction-up |
| **Volatility** | You care about *any* large move (e.g., options, vol trading); ~2× more positives helps class balance | You need directional trades — volatility label gives no edge on direction |

**Recommended baseline setting:** `thresh=1%, fwd=120` for direction experiments. Positive rate is ~9%, which is imbalanced but feasible with class-weight adjustments. At fwd=30 with 1%, the positive rate drops to 1.4%, making learning very difficult.

**Key structural insight:** At ≥1% threshold, volatility and direction labels are almost perfectly complementary — volatility ≈ direction-up OR direction-down with negligible whipsaw contamination. Choosing between them is a choice of objective, not a choice of data quality.
