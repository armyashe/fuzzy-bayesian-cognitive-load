"""
ANFIS 5-Layer (TSK / Sugeno) Classifier
========================================
Adaptive-Network-based Fuzzy Inference System with:

  Layer 1 — Gaussian MFs (premise): one bell per (rule, dimension), centres
             initialised by KMeans to avoid rule explosion.
  Layer 2 — Firing strength: product of per-dimension memberships (computed
             in log-space for numerical stability).
  Layer 3 — Normalised firing strength (softmax of log-firing).
  Layer 4 — Linear consequent (TSK): f_r(x) = a_r^T x + b_r.
  Layer 5 — Output: y_hat = Σ w̄_r · f_r(x).

Hybrid learning
---------------
* Forward pass  — compute w̄ with current premise, then solve consequent
                   parameters via ridge (weighted) least-squares.
* Backward pass — gradient descent on premise parameters (centres, sigmas)
                   to reduce MSE(y_hat, y).

The raw output is mapped through sigmoid → p ∈ [0, 1].
"""

import numpy as np
from sklearn.cluster import KMeans


class ANFIS_TSK:
    """One-signal ANFIS binary classifier (9-dim FTS features → p_task)."""

    def __init__(
        self,
        n_rules: int = 8,
        sigma_scale: float = 1.5,
        ridge_lambda: float = 0.1,
        n_epochs: int = 15,
        lr: float = 0.01,
        random_state: int = 42,
    ):
        self.n_rules = n_rules
        self.sigma_scale = sigma_scale
        self.ridge_lambda = ridge_lambda
        self.n_epochs = n_epochs
        self.lr = lr
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ANFIS_TSK":
        """
        Train ANFIS on feature matrix X and binary labels y.

        Parameters
        ----------
        X : ndarray, shape (N, d)   — e.g. 9 FTS features for one signal
        y : ndarray, shape (N,)     — binary {0, 1}
        """
        N, d = X.shape
        self.d_ = d

        # Cap rules to avoid KMeans failure on tiny folds
        actual_rules = min(self.n_rules, max(N // 3, 2))
        self.n_rules_ = actual_rules

        # --- Initialise premise via KMeans ---
        km = KMeans(
            n_clusters=actual_rules,
            random_state=self.random_state,
            n_init=10,
        )
        km.fit(X)
        self.centers_ = km.cluster_centers_.copy()  # (R, d)
        labels = km.labels_

        global_std = np.std(X, axis=0)
        global_std = np.maximum(global_std, 1e-6)

        self.sigmas_ = np.zeros((actual_rules, d))
        for r in range(actual_rules):
            mask = labels == r
            if mask.sum() > 1:
                self.sigmas_[r] = np.std(X[mask], axis=0) * self.sigma_scale
            else:
                self.sigmas_[r] = global_std * self.sigma_scale

        # Floor sigmas
        self.sigmas_ = np.maximum(self.sigmas_, global_std * 0.1)
        self.sigmas_ = np.maximum(self.sigmas_, 1e-6)

        # --- Hybrid learning ---
        self._hybrid_train(X, y)
        return self

    # ------------------------------------------------------------------
    # Layers 1-3:  normalised firing strengths
    # ------------------------------------------------------------------
    def _normalised_firing(self, X: np.ndarray) -> np.ndarray:
        """
        Compute w̄  (N, R)  via softmax over log firing strengths.

        Layer 1:  μ_{r,j}(x_j) = exp(−(x_j − c_{r,j})² / (2σ_{r,j}²))
        Layer 2:  w_r = ∏_j μ_{r,j}  ⟹  log w_r = Σ_j log μ_{r,j}
        Layer 3:  w̄ = softmax(log w)
        """
        # (N,1,d) − (1,R,d)  →  (N,R,d)
        diff = X[:, np.newaxis, :] - self.centers_[np.newaxis, :, :]
        log_mu = -0.5 * (diff / self.sigmas_[np.newaxis, :, :]) ** 2  # L1
        log_w = log_mu.sum(axis=2)                                     # L2

        # Numerically stable softmax                                   # L3
        log_w -= log_w.max(axis=1, keepdims=True)
        w = np.exp(log_w)
        w_sum = w.sum(axis=1, keepdims=True)
        w_sum = np.where(w_sum < 1e-10, 1.0, w_sum)
        return w / w_sum

    # ------------------------------------------------------------------
    # Design matrix  Φ  for WLS
    # ------------------------------------------------------------------
    def _build_phi(self, X: np.ndarray,
                   W_bar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Φ[i, r*(d+1) : (r+1)*(d+1)] = w̄_r(i) · [x_i ; 1]
        """
        N, d = X.shape
        R = self.n_rules_
        X_aug = np.hstack([X, np.ones((N, 1))])            # (N, d+1)
        Phi = np.zeros((N, R * (d + 1)))
        for r in range(R):
            Phi[:, r * (d + 1): (r + 1) * (d + 1)] = (
                W_bar[:, r: r + 1] * X_aug
            )
        return Phi, X_aug

    # ------------------------------------------------------------------
    # Hybrid train loop
    # ------------------------------------------------------------------
    def _hybrid_train(self, X: np.ndarray, y: np.ndarray) -> None:
        N, d = X.shape
        R = self.n_rules_

        for epoch in range(self.n_epochs):
            # --- Forward: WLS for consequent ---
            W_bar = self._normalised_firing(X)
            Phi, X_aug = self._build_phi(X, W_bar)

            lam = self.ridge_lambda
            A = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
            self.theta_ = np.linalg.solve(A, Phi.T @ y)

            y_hat = Phi @ self.theta_

            # Skip GD on last iteration (unnecessary)
            if epoch == self.n_epochs - 1:
                break

            # --- Backward: GD on premise (centres, sigmas) ---
            residual = (2.0 / N) * (y_hat - y)  # dL/dy_hat  (N,)

            # Per-rule consequent output
            f_r = np.zeros((N, R))
            for r in range(R):
                f_r[:, r] = X_aug @ self.theta_[
                    r * (d + 1): (r + 1) * (d + 1)
                ]

            # dL/dW_bar
            dW = residual[:, np.newaxis] * f_r  # (N, R)

            # softmax backward:  dL/dlog_w_s = w̄_s · (dL/dw̄_s − Σ_r dL/dw̄_r · w̄_r)
            weighted = (dW * W_bar).sum(axis=1, keepdims=True)  # (N,1)
            d_logw = W_bar * (dW - weighted)                    # (N, R)

            # dlog_w / d_centers  and  dlog_w / d_sigmas
            diff = (
                X[:, np.newaxis, :]
                - self.centers_[np.newaxis, :, :]
            )  # (N, R, d)

            dc = diff / (self.sigmas_[np.newaxis, :, :] ** 2)           # (N,R,d)
            ds = (diff ** 2) / (self.sigmas_[np.newaxis, :, :] ** 3)    # (N,R,d)

            grad_c = np.einsum("nr,nrd->rd", d_logw, dc)
            grad_s = np.einsum("nr,nrd->rd", d_logw, ds)

            # Gradient clipping
            for g in (grad_c, grad_s):
                gn = np.linalg.norm(g)
                if gn > 10.0:
                    g *= 10.0 / gn

            self.centers_ -= self.lr * grad_c
            self.sigmas_ -= self.lr * grad_s
            self.sigmas_ = np.maximum(self.sigmas_, 1e-4)

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Layer 4-5 output (before sigmoid)."""
        W_bar = self._normalised_firing(X)
        Phi, _ = self._build_phi(X, W_bar)
        return Phi @ self.theta_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Sigmoid of raw output  →  P(task)."""
        raw = self.predict_raw(X)
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -500, 500)))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
