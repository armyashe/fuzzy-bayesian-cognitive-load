"""
ANFIS 5-Layer (TSK / Sugeno) Classifier
========================================
Adaptive-Network-based Fuzzy Inference System with:

  Layer 1 - Premise MFs (gaussian or gbell per (rule, dimension)), centres
             initialised by KMeans to avoid rule explosion.
             Gaussian : mu = exp(-(x-c)^2/(2sigma^2))
             Gbell    : mu = 1/(1+|(x-c)/sigma|^(2b))
  Layer 2 - Firing strength: product of per-dimension memberships (computed
             in log-space for numerical stability).
  Layer 3 - Normalised firing strength (softmax of log-firing).
  Layer 4 - Linear consequent (TSK): f_r(x) = a_r^T x + b_r.
  Layer 5 - Output: y_hat = sum w_bar_r * f_r(x).

Hybrid learning
---------------
* Forward pass  - compute w_bar with current premise, then solve consequent
                   parameters via ridge (weighted) least-squares.
* Backward pass - gradient descent on premise parameters (centres, sigmas)
                   to reduce MSE(y_hat, y).

The raw output is mapped through sigmoid -> p in [0, 1].
"""

import numpy as np
from sklearn.cluster import KMeans


class ANFIS_TSK:
    """One-signal ANFIS binary classifier (9-dim FTS features -> p_task)."""

    def __init__(
        self,
        n_rules: int = 8,
        sigma_scale: float = 1.5,
        ridge_lambda: float = 0.1,
        n_epochs: int = 15,
        lr: float = 0.01,
        random_state: int = 42,
        mf_type: str = "gaussian",
        gbell_b: float = 2.0,
    ):
        if mf_type not in ("gaussian", "gbell"):
            raise ValueError(f"ANFIS mf_type must be 'gaussian' or 'gbell', got '{mf_type}'")
        self.n_rules = n_rules
        self.sigma_scale = sigma_scale
        self.ridge_lambda = ridge_lambda
        self.n_epochs = n_epochs
        self.lr = lr
        self.random_state = random_state
        self.mf_type = mf_type
        self.gbell_b = gbell_b

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ANFIS_TSK":
        """
        Train ANFIS on feature matrix X and binary labels y.

        Parameters
        ----------
        X : ndarray, shape (N, d)   - e.g. 9 FTS features for one signal
        y : ndarray, shape (N,)     - binary {0, 1}
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
        Compute w_bar  (N, R)  via softmax over log firing strengths.

        Gaussian: mu_{r,j}(x_j) = exp(-(x_j-c_{r,j})^2/(2sigma_{r,j}^2))
        Gbell   : mu_{r,j}(x_j) = 1/(1+|(x_j-c_{r,j})/sigma_{r,j}|^(2b))
        Layer 2: w_r = prod_j mu_{r,j}  =>  log w_r = sum_j log mu_{r,j}
        Layer 3: w_bar = softmax(log w)
        """
        # (N,1,d) - (1,R,d)  ->  (N,R,d)
        diff = X[:, np.newaxis, :] - self.centers_[np.newaxis, :, :]
        sig  = self.sigmas_[np.newaxis, :, :]                         # (1,R,d)

        if self.mf_type == "gaussian":
            log_mu = -0.5 * (diff / sig) ** 2                        # L1 Gaussian
        else:  # gbell
            u_b = ((diff / sig) ** 2) ** self.gbell_b                # |d/sigma|^{2b}
            log_mu = -np.log1p(u_b)                                  # L1 Gbell

        log_w = log_mu.sum(axis=2)                                    # L2

        # Numerically stable softmax                                  # L3
        log_w -= log_w.max(axis=1, keepdims=True)
        w = np.exp(log_w)
        w_sum = w.sum(axis=1, keepdims=True)
        w_sum = np.where(w_sum < 1e-10, 1.0, w_sum)
        return w / w_sum

    # ------------------------------------------------------------------
    # Design matrix  Phi  for WLS
    # ------------------------------------------------------------------
    def _build_phi(self, X: np.ndarray,
                   W_bar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Phi[i, r*(d+1) : (r+1)*(d+1)] = w_bar_r(i) * [x_i ; 1]
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
            try:
                self.theta_ = np.linalg.solve(A, Phi.T @ y)
            except np.linalg.LinAlgError:
                # Auto-fix: bump ridge_lambda to avoid singular matrix
                for fallback_lam in [1e-3, 1e-2, 1e-1, 1.0]:
                    if fallback_lam <= lam:
                        continue
                    try:
                        A = Phi.T @ Phi + fallback_lam * np.eye(Phi.shape[1])
                        self.theta_ = np.linalg.solve(A, Phi.T @ y)
                        print(f"[ANFIS] auto-fix ridge_lambda {lam} -> "
                              f"{fallback_lam} to avoid singular matrix")
                        lam = fallback_lam
                        break
                    except np.linalg.LinAlgError:
                        continue
                else:
                    # Last resort: pseudo-inverse
                    print(f"[ANFIS] singular matrix even with ridge=1.0, "
                          f"using pseudo-inverse")
                    self.theta_ = np.linalg.lstsq(A, Phi.T @ y, rcond=None)[0]

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

            # softmax backward:  dL/dlog_w_s = w_bar_s * (dL/dw_bar_s - sum_r dL/dw_bar_r * w_bar_r)
            weighted = (dW * W_bar).sum(axis=1, keepdims=True)  # (N,1)
            d_logw = W_bar * (dW - weighted)                    # (N, R)

            # dlog_w / d_centers  and  dlog_w / d_sigmas
            diff = (
                X[:, np.newaxis, :]
                - self.centers_[np.newaxis, :, :]
            )  # (N, R, d)
            sig = self.sigmas_[np.newaxis, :, :]                         # (1,R,d)

            if self.mf_type == "gaussian":
                # d(log mu)/dc = d/sigma^2,   d(log mu)/dsigma = d^2/sigma^3
                dc = diff / (sig ** 2)                                   # (N,R,d)
                ds = (diff ** 2) / (sig ** 3)                            # (N,R,d)
            else:  # gbell
                # d(log mu)/dc = 2b(1-mu)/d,   d(log mu)/dsigma = 2b(1-mu)/sigma
                # where mu = 1/(1+|d/sigma|^{2b}),  1-mu = |d/sigma|^{2b}/(1+|d/sigma|^{2b})
                u_b   = ((diff / sig) ** 2) ** self.gbell_b              # (N,R,d)
                one_m = u_b / (1.0 + u_b)                               # 1 - mu per dim
                fac   = 2.0 * self.gbell_b * one_m                      # 2b(1-mu)
                # Safe division by diff: limit is 0 when diff->0
                dc = np.where(np.abs(diff) > 1e-8, fac / diff, 0.0)     # (N,R,d)
                ds = fac / sig                                           # (N,R,d)

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
        """Sigmoid of raw output  ->  P(task)."""
        raw = self.predict_raw(X)
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -500, 500)))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Explainability  (does NOT change training / inference)
    # ------------------------------------------------------------------
    def get_rule_params(self) -> dict:
        """Return premise and consequent parameters safely.

        Returns
        -------
        dict with keys ``centers``, ``sigmas``, ``theta``, ``n_rules``, ``d``.
        """
        return {
            "centers": self.centers_.copy(),
            "sigmas": self.sigmas_.copy(),
            "theta": self.theta_.copy(),
            "n_rules": self.n_rules_,
            "d": self.d_,
        }

    def rule_strings(self, feature_names=None, top_k: int = 4,
                     decimals: int = 4) -> list:
        """Return human-readable IF-THEN rules (top_k antecedents per rule).

        Parameters
        ----------
        feature_names : list[str] or None
            Names for each input dimension.  Falls back to ``x0, x1, ...``.
        top_k : int
            Keep only the *top_k* antecedent features with the largest
            absolute consequent weight ``|a_j|`` to keep rules concise.
        decimals : int
            Decimal places for numeric formatting.

        Returns
        -------
        list[str]   - one string per rule, e.g.
            ``"R1: IF x2 is Gauss(c=0.12, sigma=0.05) AND ... THEN y = 0.3*x2 + ... + 0.01"``
        """
        R = self.n_rules_
        d = self.d_
        if feature_names is None:
            feature_names = [f"x{j}" for j in range(d)]

        fmt = f".{decimals}f"
        rules = []
        for r in range(R):
            block = self.theta_[r * (d + 1): (r + 1) * (d + 1)]
            a = block[:d]
            b = block[d]

            # Select top_k features by |a_j|
            k = min(top_k, d)
            idx = np.argsort(np.abs(a))[::-1][:k]
            idx_sorted = np.sort(idx)

            # Antecedent (premise)
            parts = []
            for j in idx_sorted:
                c_val = self.centers_[r, j]
                s_val = self.sigmas_[r, j]
                if self.mf_type == "gaussian":
                    mf_str = f"Gauss(c={c_val:{fmt}}, sigma={s_val:{fmt}})"
                else:
                    mf_str = f"Gbell(c={c_val:{fmt}}, a={s_val:{fmt}}, b={self.gbell_b})"
                parts.append(f"{feature_names[j]} is {mf_str}")
            ante = " AND ".join(parts)

            # Consequent
            cons_parts = []
            for j in idx_sorted:
                sign = "+" if a[j] >= 0 else "-"
                cons_parts.append(f"{sign}{abs(a[j]):{fmt}}*{feature_names[j]}")
            sign_b = "+" if b >= 0 else "-"
            cons = " ".join(cons_parts) + f" {sign_b}{abs(b):{fmt}}"
            # Clean leading '+'
            if cons.startswith("+"):
                cons = cons[1:]

            rules.append(f"R{r + 1}: IF {ante} THEN y = {cons}")
        return rules

    def rule_strings_full(self, feature_names=None,
                          decimals: int = 4) -> list:
        """Return full rules (all antecedent features, no truncation)."""
        return self.rule_strings(feature_names=feature_names,
                                 top_k=self.d_, decimals=decimals)
