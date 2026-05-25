"""
Regime-adaptive decision threshold for the DirectionModel family.

The QQQ/down h=60 d=1.5% asymmetric model's raw predict_proba() output requires
regime-specific thresholds because:
  - Default threshold=0.5 is only reliable in the recovery regime.
  - In correction/crisis regimes, the model compresses all probabilities near zero
    (mean_pred ≈ 0.3% vs 6% true pos rate), so high AUC signal exists but is
    inaccessible at a fixed threshold of 0.5.
  - In crisis regimes (AUC ≈ 0.58, statistically robust), NO threshold rescues
    the model — a hard gate is the only safe response.

Best real-time regime proxy: the rolling 5-day positive label rate from the same
data pipeline. It directly measures intraday downside volatility and maps cleanly
to the categories below. See decision_threshold_by_regime_2026-05-24.md for the
full per-regime analysis.
"""


def get_threshold(recent_label_rate: float) -> float | None:
    """
    Return the decision threshold for DirectionModel.predict_proba() output.

    Parameters
    ----------
    recent_label_rate : float
        Fraction of bars labeled 1 (QQQ/down h=60 d=1.5%) over the trailing
        5 trading days (0–1 scale).

    Returns
    -------
    threshold : float | None
        Apply as: (prob_pred >= threshold).astype(int).
        Returns None in the crisis regime — caller must suspend model output.

    Regime map
    ----------
    > 8%  : Crisis / black swan   — AUC collapses to ~0.58; SUSPEND.
    > 6%  : Strong correction     — 0.03  (probabilities compressed near zero; fire on top ~3%)
    > 3%  : Correction            — 0.15
    > 1.5%: Mild correction       — 0.12
    > 0.8%: Normal / flat         — 0.25
    else  : Recovery              — 0.45  (near default; model well-calibrated at this regime)
    """
    if recent_label_rate > 0.08:
        return None   # Crisis — SUSPEND model output
    elif recent_label_rate > 0.06:
        return 0.03   # Strong correction
    elif recent_label_rate > 0.03:
        return 0.15   # Correction
    elif recent_label_rate > 0.015:
        return 0.12   # Mild correction
    elif recent_label_rate > 0.008:
        return 0.25   # Normal / flat
    else:
        return 0.45   # Recovery
