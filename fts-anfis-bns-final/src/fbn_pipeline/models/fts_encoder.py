"""
Fuzzy Time Series Encoder
=========================
Converts raw (N, n_signals, T) time-series windows into fuzzy transition
matrix features.  For each signal the encoder:

1. Fits n_sets membership functions (Low / Med / High for n=3) from training
   data.  Partitioning: quantile-based for triangular, grid (even spacing)
   for gaussian / trapezoid / gbell.
   Supported mf_type: "triangular", "gaussian", "trapezoid", "gbell".
2. Fuzzifies every time point holistically (keeps all n_sets membership
   degrees).
3. Builds an n_setsxn_sets transition matrix
   T_ij = sum_{t} mu_i(x_t) * mu_j(x_{t+1})
   and normalises by total sum  ->  n_sets^2 features per signal.
4. Concatenates across signals  ->  n_signals x n_sets^2  features per window.
"""

import numpy as np


class FTSEncoder:
    """Multi-MF FTS encoder (triangular / gaussian / trapezoid / gbell)."""

    VALID_MF = {"triangular", "gaussian", "trapezoid", "gbell"}

    VALID_NORM = {"total_sum", "row"}

    def __init__(self, n_sets: int = 3, mf_type: str = "triangular",
                 mf_sigma_scale: float = 0.5, gbell_b: float = 2.0,
                 norm_mode: str = "total_sum", robust: bool = False):
        assert mf_type in self.VALID_MF, (
            f"mf_type must be one of {self.VALID_MF}, got '{mf_type}'"
        )
        assert norm_mode in self.VALID_NORM, (
            f"norm_mode must be one of {self.VALID_NORM}, got '{norm_mode}'"
        )
        self.n_sets = n_sets
        self.mf_type = mf_type
        self.mf_sigma_scale = mf_sigma_scale  # gaussian sigma = spacing * this
        self.gbell_b = gbell_b                # gbell shape exponent
        self.norm_mode = norm_mode            # total_sum | row
        self.robust = robust                  # percentile clipping + quantile centers
        self.fitted_ = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "FTSEncoder":
        """
        Learn MF parameters from training data.

        Parameters
        ----------
        X : ndarray, shape (N, n_signals, T)
        """
        N, S, T = X.shape
        self.n_signals = S

        # Per-signal median for NaN imputation
        self.medians_ = np.nanmedian(X, axis=(0, 2))  # (S,)

        if self.mf_type == "triangular":
            self._fit_triangular(X)
        else:
            self._fit_grid(X)

        self.fitted_ = True
        return self

    # ---- partitioning strategies ----

    def _fit_triangular(self, X: np.ndarray) -> None:
        """Quantile-based partitioning for triangular MFs."""
        S = self.n_signals
        n = self.n_sets
        self.mf_params_: list[list[tuple]] = []
        for s in range(S):
            vals = X[:, s, :].ravel()
            vals = vals[~np.isnan(vals)]
            if self.robust:
                p2, p98 = np.percentile(vals, [2, 98])
                vals = vals[(vals >= p2) & (vals <= p98)]
            if n == 3:
                # Original quantile positions for backward compatibility
                q = np.quantile(vals, [0.0, 1 / 6, 0.5, 5 / 6, 1.0])
                margin = (q[4] - q[0]) * 0.10
                self.mf_params_.append([
                    (q[0] - margin, q[1], q[2]),
                    (q[1], q[2], q[3]),
                    (q[2], q[3], q[4] + margin),
                ])
            else:
                # General: n+2 quantile reference points -> n triangles
                qs = np.quantile(vals, np.linspace(0, 1, n + 2))
                margin = (qs[-1] - qs[0]) * 0.10
                params = []
                for k in range(n):
                    a = qs[k] - (margin if k == 0 else 0)
                    b = qs[k + 1]
                    c = qs[k + 2] + (margin if k == n - 1 else 0)
                    params.append((a, b, c))
                self.mf_params_.append(params)

    def _fit_grid(self, X: np.ndarray) -> None:
        """Grid (even-spacing) partitioning for gaussian / trapezoid / gbell.

        When self.robust is True, uses 2nd-98th percentile clipping and
        quantile-based centers instead of evenly-spaced min/max centres.
        """
        S = self.n_signals
        self.grid_: list[dict] = []
        for s in range(S):
            vals = X[:, s, :].ravel()
            vals = vals[~np.isnan(vals)]

            if self.robust:
                p2, p98 = np.percentile(vals, [2, 98])
                vals_clipped = vals[(vals >= p2) & (vals <= p98)]
                if len(vals_clipped) < 2:
                    vals_clipped = vals
                centers = np.quantile(vals_clipped,
                                      np.linspace(0, 1, self.n_sets))
                if self.n_sets > 1:
                    spacing = float(np.mean(np.diff(centers)))
                else:
                    spacing = float(p98 - p2)
                spacing = max(spacing, 1e-6)
            else:
                vmin, vmax = float(vals.min()), float(vals.max())
                rng = vmax - vmin
                if rng < 1e-10:
                    rng = 1.0
                margin = rng * 0.05
                vmin_e, vmax_e = vmin - margin, vmax + margin
                centers = np.linspace(vmin_e, vmax_e, self.n_sets)
                spacing = ((vmax_e - vmin_e) / (self.n_sets - 1)
                           if self.n_sets > 1 else rng)

            g: dict = {"centers": centers, "spacing": spacing}

            # Pre-compute trapezoid (a,b,c,d) tuples
            if self.mf_type == "trapezoid":
                half_p = spacing * 0.15          # narrow plateau
                g["trap_params"] = [
                    (centers[k] - spacing, centers[k] - half_p,
                     centers[k] + half_p, centers[k] + spacing)
                    for k in range(self.n_sets)
                ]

            self.grid_.append(g)

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Encode raw windows into FTS transition features.

        Parameters
        ----------
        X : ndarray, shape (N, n_signals, T)

        Returns
        -------
        features : ndarray, shape (N, n_signals * n_sets^2)
        """
        assert self.fitted_, "Call .fit() first"
        X = self._impute(X)
        N, S, T = X.shape

        all_feats: list[np.ndarray] = []
        for s in range(S):
            mu = self._memberships(X[:, s, :], s)  # (N, T, 3)

            # Transition matrix via outer product of consecutive memberships
            mu_prev = mu[:, :-1, :]  # (N, T-1, 3)
            mu_next = mu[:, 1:, :]   # (N, T-1, 3)
            trans = np.einsum("nti,ntj->nij", mu_prev, mu_next)  # (N, 3, 3)

            # Normalise transition matrix
            if self.norm_mode == "row":
                rsum = trans.sum(axis=2, keepdims=True)
                rsum = np.where(rsum < 1e-10, 1.0, rsum)
                trans = trans / rsum
            else:  # total_sum
                tsum = trans.sum(axis=(1, 2), keepdims=True)
                tsum = np.where(tsum < 1e-10, 1.0, tsum)
                trans = trans / tsum

            all_feats.append(trans.reshape(N, -1))  # (N, 9)

        return np.hstack(all_feats)  # (N, 36)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------
    # Single-sample transition matrix (for explainability)
    # ------------------------------------------------------------------
    def get_transition_matrix(self, X_single: np.ndarray,
                              signal_idx: int) -> np.ndarray:
        """
        Return the n_sets x n_sets transition matrix for one sample / one signal.

        Parameters
        ----------
        X_single : ndarray, shape (n_signals, T)
        signal_idx : int

        Returns
        -------
        trans : ndarray, shape (n_sets, n_sets)
        """
        X_single = X_single[np.newaxis, ...]          # (1, S, T)
        X_single = self._impute(X_single)
        vals = X_single[:, signal_idx, :]              # (1, T)
        mu = self._memberships(vals, signal_idx)[0]    # (T, n_sets)

        trans = np.zeros((self.n_sets, self.n_sets))
        for t in range(mu.shape[0] - 1):
            trans += np.outer(mu[t], mu[t + 1])

        if self.norm_mode == "row":
            for i in range(self.n_sets):
                rs = trans[i].sum()
                if rs > 1e-10:
                    trans[i] /= rs
        else:
            ts = trans.sum()
            if ts > 1e-10:
                trans /= ts
        return trans

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _impute(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        for s in range(self.n_signals):
            ch = X[:, s, :]
            ch[np.isnan(ch)] = self.medians_[s]
        return X

    @staticmethod
    def _triangle(x, a, b, c):
        """Triangular membership function."""
        eps = 1e-10
        left = (x - a) / (b - a + eps)
        right = (c - x) / (c - b + eps)
        return np.clip(np.minimum(left, right), 0.0, 1.0)

    @staticmethod
    def _gaussian(x, center, sigma):
        """Gaussian MF: exp(-0.5*((x-c)/sigma)^2)."""
        return np.exp(-0.5 * ((x - center) / max(sigma, 1e-10)) ** 2)

    @staticmethod
    def _trapezoid_mf(x, a, b, c, d):
        """Trapezoidal MF: plateau on [b, c], feet at a and d."""
        eps = 1e-10
        rise = (x - a) / (b - a + eps)
        fall = (d - x) / (d - c + eps)
        return np.clip(np.minimum(np.minimum(rise, fall), 1.0), 0.0, 1.0)

    @staticmethod
    def _gbell(x, a, b, c):
        """Generalized bell MF: 1 / (1 + |(x-c)/a|^(2b))."""
        return 1.0 / (1.0 + np.abs((x - c) / max(a, 1e-10)) ** (2 * b))

    def _memberships(self, values: np.ndarray,
                     signal_idx: int) -> np.ndarray:
        """
        Holistic fuzzification - keeps all n_sets mu values.

        Parameters
        ----------
        values : ndarray, shape (N, T)
        signal_idx : int

        Returns
        -------
        mu : ndarray, shape (N, T, n_sets)
        """
        N, T = values.shape
        mu = np.zeros((N, T, self.n_sets))

        if self.mf_type == "triangular":
            for k, (a, b, c) in enumerate(self.mf_params_[signal_idx]):
                mu[:, :, k] = self._triangle(values, a, b, c)

        elif self.mf_type == "gaussian":
            g = self.grid_[signal_idx]
            sigma = g["spacing"] * self.mf_sigma_scale
            for k in range(self.n_sets):
                mu[:, :, k] = self._gaussian(
                    values, g["centers"][k], sigma
                )

        elif self.mf_type == "trapezoid":
            g = self.grid_[signal_idx]
            for k in range(self.n_sets):
                a, b, c, d = g["trap_params"][k]
                mu[:, :, k] = self._trapezoid_mf(values, a, b, c, d)

        elif self.mf_type == "gbell":
            g = self.grid_[signal_idx]
            a_w = g["spacing"] * self.mf_sigma_scale
            for k in range(self.n_sets):
                mu[:, :, k] = self._gbell(
                    values, a_w, self.gbell_b, g["centers"][k]
                )

        # Normalise so memberships sum to 1 at each time step
        mu_sum = mu.sum(axis=2, keepdims=True)
        mu_sum = np.where(mu_sum < 1e-10, 1.0, mu_sum)
        mu = mu / mu_sum
        return mu
