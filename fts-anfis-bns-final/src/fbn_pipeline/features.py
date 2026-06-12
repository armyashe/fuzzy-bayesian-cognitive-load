"""
Feature Extractors for Ablation Experiments
============================================
This module provides two alternative feature extraction strategies used when
parts of the main pipeline are bypassed in ablation experiments.

Stage mapping
-------------
  exp1 : FTSEncoder  ->  ANFIS  ->  BNFusion        (standard pipeline)
  exp2 : FTSEncoder  ->  ANFIS  ->  mean/weighted   (no BN - uses pipeline.py)
  exp3 : FTSEncoder  ->  summarise  ->  BNFusion     (no ANFIS - uses summarise_fts_per_signal)
  exp4 : StatFeatures ->  ANFIS  ->  BNFusion        (no FTS - uses extract_stat_features)

Public API
----------
  extract_stat_features(X)          -- (N,4,T) -> (N, 4*N_STAT_FEATURES)
  feature_names_stat(n_signals)     -- list of feature-name strings
  split_features_per_signal(F, n)   -- (N, n*d) -> list of n arrays (N, d)
  summarise_fts_per_signal(F, n_sets, mode) -- (N, 4*n_sets^2) -> (N, 4)
"""

from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------
# Statistical feature names (order must match extract_stat_features)
# ------------------------------------------------------------------
STAT_FEATURE_NAMES: list[str] = [
    "mean",
    "std",
    "min",
    "max",
    "range",
    "slope",
    "first_diff_mean",
    "first_diff_std",
]

N_STAT_FEATURES: int = len(STAT_FEATURE_NAMES)  # 8


# ------------------------------------------------------------------
# exp4 helper: handcrafted statistical features
# ------------------------------------------------------------------

def extract_stat_features(X: np.ndarray) -> np.ndarray:
    """
    Extract per-signal handcrafted statistical features from raw windows.

    Used in **exp4** to replace FTSEncoder (Stage 1 bypass).

    Parameters
    ----------
    X : ndarray, shape (N, 4, T)
        Raw time-series windows.  N samples, 4 physiological signals,
        T time steps per window.

    Returns
    -------
    F : ndarray, shape (N, 4 * N_STAT_FEATURES)
        Concatenated features: [signal_0_feats, signal_1_feats, ...].
        Feature order per signal: mean, std, min, max, range, slope,
        first_diff_mean, first_diff_std.
    """
    N, S, T = X.shape
    F = np.zeros((N, S * N_STAT_FEATURES), dtype=np.float64)
    t_idx = np.arange(T, dtype=np.float64)
    t_mean = t_idx.mean()
    t_var = float(np.sum((t_idx - t_mean) ** 2))

    for s in range(S):
        xs = X[:, s, :]          # (N, T)
        off = s * N_STAT_FEATURES

        F[:, off + 0] = xs.mean(axis=1)
        F[:, off + 1] = xs.std(axis=1)
        F[:, off + 2] = xs.min(axis=1)
        F[:, off + 3] = xs.max(axis=1)
        F[:, off + 4] = xs.max(axis=1) - xs.min(axis=1)

        # Linear slope via ordinary least squares
        if t_var > 0:
            xs_centered = xs - xs.mean(axis=1, keepdims=True)
            F[:, off + 5] = np.dot(xs_centered, t_idx - t_mean) / t_var
        else:
            F[:, off + 5] = 0.0

        # First-difference statistics
        diffs = np.diff(xs, axis=1)  # (N, T-1)
        F[:, off + 6] = diffs.mean(axis=1)
        F[:, off + 7] = diffs.std(axis=1)

    return F


# ------------------------------------------------------------------
# Generic signal splitter (shared by exp2, exp3, exp4)
# ------------------------------------------------------------------

def split_features_per_signal(
    F: np.ndarray,
    n_signals: int = 4,
) -> list[np.ndarray]:
    """
    Split a flat feature matrix into per-signal sub-matrices.

    Parameters
    ----------
    F : ndarray, shape (N, n_signals * n_feat_per_signal)
    n_signals : int

    Returns
    -------
    list of n_signals arrays, each shape (N, n_feat_per_signal)
    """
    if F.shape[1] % n_signals != 0:
        raise ValueError(
            f"Feature dim {F.shape[1]} is not divisible by n_signals={n_signals}"
        )
    feat_per_sig = F.shape[1] // n_signals
    return [
        F[:, s * feat_per_sig : (s + 1) * feat_per_sig]
        for s in range(n_signals)
    ]


# ------------------------------------------------------------------
# exp3 helper: FTS summary scores per signal
# ------------------------------------------------------------------

_VALID_SUMMARY_MODES = frozenset({"diag", "high", "diag_high", "entropy"})


def summarise_fts_per_signal(
    F: np.ndarray,
    n_sets: int,
    mode: str = "diag_high",
) -> np.ndarray:
    """
    Collapse FTS transition features into one continuous evidence score
    per signal.  Used in **exp3** where ANFIS is bypassed.

    The output is intended for direct input to BNFusion, which uses quantile
    binning on continuous values - not necessarily probabilities.

    Parameters
    ----------
    F : ndarray, shape (N, 4 * n_sets^2)
        FTS transition features produced by FTSEncoder.  Each signal
        contributes n_sets^2 normalised transition values (row-sum or
        total-sum normalised, as configured by cfg.fts_norm).
    n_sets : int
        Number of fuzzy sets per signal (e.g. 3 -> 9 transitions/signal).
    mode : str
        Summary strategy per signal:

        ``"diag"``
            Sum of self-transition probabilities T[i, i].
            High value => the signal stays in the same fuzzy state
            (low dynamics / stable).

        ``"high"``
            Sum of all transitions that involve the highest fuzzy state
            (index n_sets-1), i.e. T[:, h] + T[h, :] - T[h, h].
            High value => the signal frequently enters/exits high state.

        ``"diag_high"`` *(default)*
            Normalised average of ``"diag"`` and ``"high"`` scores.
            Balances stability and high-activation evidence.

        ``"entropy"``
            ``1 - H(T) / log(n_sets^2)`` where H is Shannon entropy of
            the flattened transition vector.  High value => more
            concentrated (predictable) transition pattern.

    Returns
    -------
    scores : ndarray, shape (N, 4)
        Per-signal continuous evidence scores, clipped to [0, 1].
        These are **not** ANFIS probabilities; they capture the signal's
        transition dynamics as derived from the FTS feature space.
    """
    if mode not in _VALID_SUMMARY_MODES:
        raise ValueError(
            f"fts_summary_mode must be one of {_VALID_SUMMARY_MODES}, got '{mode!r}'"
        )

    N = F.shape[0]
    n_trans = n_sets * n_sets
    if F.shape[1] % n_trans != 0:
        raise ValueError(
            f"F.shape[1]={F.shape[1]} is not divisible by n_sets^2={n_trans}"
        )
    n_signals = F.shape[1] // n_trans
    scores = np.zeros((N, n_signals), dtype=np.float64)

    for s in range(n_signals):
        tm = F[:, s * n_trans : (s + 1) * n_trans]   # (N, n_sets^2)
        tm_3d = tm.reshape(N, n_sets, n_sets)          # (N, from_state, to_state)

        if mode == "diag":
            # sum_i  T[i, i]
            scores[:, s] = np.einsum("nii->n", tm_3d)

        elif mode == "high":
            h = n_sets - 1
            to_high = tm_3d[:, :, h].sum(axis=1)    # sum_i T[i, h]
            from_high = tm_3d[:, h, :].sum(axis=1)  # sum_j T[h, j]
            # subtract double-counted T[h, h]
            scores[:, s] = to_high + from_high - tm_3d[:, h, h]

        elif mode == "diag_high":
            h = n_sets - 1
            diag_score = np.einsum("nii->n", tm_3d)
            to_high = tm_3d[:, :, h].sum(axis=1)
            from_high = tm_3d[:, h, :].sum(axis=1)
            high_raw = to_high + from_high - tm_3d[:, h, h]
            # normalise high_raw: theoretical max = 2 - 1/n_sets
            max_high = 2.0 - 1.0 / n_sets
            high_norm = high_raw / max_high if max_high > 0 else high_raw
            scores[:, s] = 0.5 * diag_score + 0.5 * high_norm

        elif mode == "entropy":
            # 1 - H_normalised  (more concentrated -> higher score)
            eps = 1e-10
            tm_safe = np.clip(tm, eps, 1.0)
            h_vec = -np.sum(tm_safe * np.log(tm_safe), axis=1)  # (N,)
            h_max = np.log(n_trans) if n_trans > 1 else 1.0
            scores[:, s] = 1.0 - h_vec / h_max

    return np.clip(scores, 0.0, 1.0)
