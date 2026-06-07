"""
Bayesian Network Fusion (Naive Bayes structure)
================================================
DAG:   Task  ->  E_Temp,  E_HR,  E_GSR,  E_RR

Each evidence node E_s is a **discretised** version of a per-signal
continuous evidence score.  Depending on the ablation experiment, this
can be either:

  * The ANFIS probability output P(task | signal_s)  -- exp1, exp2, exp3
  * A continuous FTS summary score per signal         -- exp4
    (e.g. diagonal self-transition sum, high-state activity, entropy)

Discretisation uses **quantile-based binning** fitted on the training fold
only to avoid data leakage.

CPTs are estimated by counting with Laplace smoothing (alpha >= 0).

Inference modes
---------------
* Hard evidence (default):
    P(Task=c | e) proportional to P(c) * prod_s P(E_s = e_s | c)

* Soft evidence (soft=True):
    P(Task=c | e) proportional to P(c) * prod_s [ sum_b  mu_b(v_s) * P(bin_b | c) ]
    where mu_b is a Gaussian kernel membership over bin midpoints.

* Reliability weighting (signal_weights):
    log P(c|e) proportional to log P(c) + sum_s  w_s * log P(E_s | c)
    w_s in (0, 1] derived from per-signal AUC on the training fold.
"""

import numpy as np

SIGNAL_NAMES = ["Temp", "HR", "GSR", "RR"]
BIN_NAMES = ["low", "med", "high"]  # kept for backward compat


def _make_bin_names(n_bins: int) -> list[str]:
    """Generate bin labels for arbitrary n_bins."""
    if n_bins == 3:
        return ["low", "med", "high"]
    if n_bins == 5:
        return ["very_low", "low", "med", "high", "very_high"]
    return [f"b{i}" for i in range(n_bins)]


class BNFusion:
    """
    Naive Bayes BN for fusing 4 per-signal continuous evidence scores.

    Evidence inputs may be ANFIS probabilities (exp1, exp2, exp3) or
    FTS-derived summary scores (exp4).  All inputs are discretised via
    quantile binning fitted on the training fold only.
    """

    def __init__(self, n_bins: int = 3, alpha: float = 1.0,
                 soft: bool = False):
        self.n_bins = n_bins
        self.alpha = alpha          # Laplace smoothing
        self.soft = soft            # soft evidence mode
        self.bin_names_ = _make_bin_names(n_bins)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, probs: np.ndarray, y: np.ndarray,
            signal_weights: np.ndarray | None = None) -> "BNFusion":
        """
        Learn bin edges and CPTs from training-fold ANFIS outputs.

        Parameters
        ----------
        probs : ndarray, shape (N, 4)
            Per-signal continuous evidence scores.  Can be ANFIS probability
            outputs P(task | signal_s) or any per-signal continuous value
            such as FTS summary scores (exp4).
        y     : ndarray, shape (N,)    -- binary labels {0, 1}
        signal_weights : ndarray, shape (4,), optional
            Per-signal reliability weights in (0, 1].  Default: all 1.0.
        """
        N, S = probs.shape
        self.n_signals_ = S
        self.signal_weights_ = (
            signal_weights if signal_weights is not None
            else np.ones(S)
        )

        # --- Quantile bin edges (fit on train only) ---
        self.bin_edges_: list[np.ndarray] = []
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]  # e.g. [1/3, 2/3] for 3
        for s in range(S):
            edges = np.quantile(probs[:, s], qs)
            self.bin_edges_.append(edges)

        evidence = self._discretise(probs)  # (N, 4)  values in {0,1,...}

        # --- Prior  P(Task = c) ---
        self.prior_ = np.zeros(2)
        for c in range(2):
            self.prior_[c] = (np.sum(y == c) + self.alpha) / (
                N + 2 * self.alpha
            )

        # --- CPTs  P(E_s = b | Task = c) ---
        self.cpts_: list[np.ndarray] = []
        for s in range(S):
            cpt = np.zeros((2, self.n_bins))       # cpt[class, bin]
            for c in range(2):
                mask = y == c
                for b in range(self.n_bins):
                    cpt[c, b] = (
                        np.sum(evidence[mask, s] == b) + self.alpha
                    )
                cpt[c] /= cpt[c].sum()
            self.cpts_.append(cpt)

        # --- Soft evidence parameters ---
        if self.soft:
            self._compute_soft_params(probs)

        return self

    # ------------------------------------------------------------------
    # Soft evidence helpers
    # ------------------------------------------------------------------
    def _compute_soft_params(self, probs: np.ndarray) -> None:
        """Compute bin midpoints and Gaussian kernel sigma for soft evidence."""
        self.bin_mids_: list[np.ndarray] = []
        self.bin_sigmas_: list[float] = []
        for s in range(self.n_signals_):
            edges = self.bin_edges_[s]
            p_min = float(probs[:, s].min())
            p_max = float(probs[:, s].max())

            # Extended edges: [p_min, e1, ..., e_{n-1}, p_max]
            ext = np.concatenate([[p_min], edges, [p_max]])
            mids = 0.5 * (ext[:-1] + ext[1:])  # midpoint of each bin interval

            # sigma = half average bin width -> decent overlap
            avg_width = (p_max - p_min) / self.n_bins if self.n_bins > 0 else 0.1
            sigma = max(avg_width * 0.5, 1e-6)

            self.bin_mids_.append(mids)
            self.bin_sigmas_.append(sigma)

    def _soft_membership(self, probs: np.ndarray) -> np.ndarray:
        """
        Gaussian kernel soft membership over bins.

        Returns
        -------
        mem : ndarray, shape (N, S, n_bins)   - row-normalised
        """
        N, S = probs.shape
        mem = np.zeros((N, S, self.n_bins))
        for s in range(S):
            mids = self.bin_mids_[s]
            sigma = self.bin_sigmas_[s]
            for b in range(self.n_bins):
                mem[:, s, b] = np.exp(
                    -0.5 * ((probs[:, s] - mids[b]) / sigma) ** 2
                )
            # Normalise so memberships sum to 1 per sample
            mem_sum = mem[:, s, :].sum(axis=1, keepdims=True)
            mem[:, s, :] /= np.maximum(mem_sum, 1e-10)
        return mem

    # ------------------------------------------------------------------
    # Discretise
    # ------------------------------------------------------------------
    def _discretise(self, probs: np.ndarray) -> np.ndarray:
        N, S = probs.shape
        ev = np.zeros((N, S), dtype=int)
        for s in range(S):
            ev[:, s] = np.digitize(probs[:, s], self.bin_edges_[s])
            ev[:, s] = np.clip(ev[:, s], 0, self.n_bins - 1)
        return ev

    def get_evidence_bins(self, probs: np.ndarray) -> np.ndarray:
        """Return discretised evidence matrix (for logging)."""
        return self._discretise(probs)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        """
        Compute posterior P(Task=1 | evidence) for each sample.

        Dispatches to soft or hard evidence path based on self.soft.
        Both paths support signal_weights_.

        Parameters
        ----------
        probs : ndarray, shape (N, 4)

        Returns
        -------
        posteriors : ndarray, shape (N,)
        """
        if self.soft:
            return self._predict_soft(probs)
        return self._predict_hard(probs)

    def _predict_hard(self, probs: np.ndarray) -> np.ndarray:
        """Hard-evidence Naive Bayes with optional signal weighting."""
        evidence = self._discretise(probs)
        N = evidence.shape[0]
        posteriors = np.zeros(N)
        w = self.signal_weights_

        for i in range(N):
            log_p = np.zeros(2)
            for c in range(2):
                log_p[c] = np.log(self.prior_[c] + 1e-10)
                for s in range(self.n_signals_):
                    log_p[c] += w[s] * np.log(
                        self.cpts_[s][c, evidence[i, s]] + 1e-10
                    )
            # Stable softmax
            log_p -= log_p.max()
            p = np.exp(log_p)
            p /= p.sum()
            posteriors[i] = p[1]

        return posteriors

    def _predict_soft(self, probs: np.ndarray) -> np.ndarray:
        """Soft-evidence Naive Bayes: sum_b mu_b(p) * CPT[c, b]."""
        mem = self._soft_membership(probs)
        N = probs.shape[0]
        posteriors = np.zeros(N)
        w = self.signal_weights_

        for i in range(N):
            log_p = np.zeros(2)
            for c in range(2):
                log_p[c] = np.log(self.prior_[c] + 1e-10)
                for s in range(self.n_signals_):
                    # Weighted CPT lookup: sum_b mu_b * P(b|c)
                    p_ev = np.dot(mem[i, s, :], self.cpts_[s][c, :])
                    log_p[c] += w[s] * np.log(p_ev + 1e-10)
            # Stable softmax
            log_p -= log_p.max()
            p = np.exp(log_p)
            p /= p.sum()
            posteriors[i] = p[1]

        return posteriors

    def predict(self, probs: np.ndarray,
                threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(probs) >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Explainability helpers
    # ------------------------------------------------------------------
    def get_cpt_table(self) -> dict:
        """Return CPTs as a nested dict for pretty-printing."""
        tables: dict = {}
        for s in range(self.n_signals_):
            name = SIGNAL_NAMES[s] if s < len(SIGNAL_NAMES) else f"S{s}"
            table: dict = {}
            for c in range(2):
                cls = "rest" if c == 0 else "task"
                table[cls] = {
                    self.bin_names_[b]: round(float(self.cpts_[s][c, b]), 4)
                    for b in range(self.n_bins)
                }
            tables[name] = table
        return tables

    def get_signal_weights(self) -> dict:
        """Return signal weights for interpretability."""
        return {
            (SIGNAL_NAMES[s] if s < len(SIGNAL_NAMES) else f"S{s}"): round(
                float(self.signal_weights_[s]), 4
            )
            for s in range(self.n_signals_)
        }

    def explain_sample(self, probs_single: np.ndarray) -> dict:
        """
        Full inference trace for one sample.

        Dispatches to hard or soft evidence path based on self.soft.
        The posterior in the returned dict is guaranteed to match
        predict_proba(probs_single.reshape(1,-1))[0] within 1e-8.

        Parameters
        ----------
        probs_single : ndarray, shape (4,)

        Returns
        -------
        dict - see _explain_sample_hard / _explain_sample_soft for keys.
        """
        if self.soft:
            return self._explain_sample_soft(probs_single)
        return self._explain_sample_hard(probs_single)

    def _explain_sample_hard(self, probs_single: np.ndarray) -> dict:
        """Hard-evidence inference trace (original logic)."""
        ev = self._discretise(probs_single.reshape(1, -1))[0]
        w = self.signal_weights_

        log_p = np.zeros(2)
        contribs: dict = {0: {}, 1: {}}
        for c in range(2):
            log_p[c] = np.log(self.prior_[c] + 1e-10)
            for s in range(self.n_signals_):
                name = (
                    SIGNAL_NAMES[s] if s < len(SIGNAL_NAMES) else f"S{s}"
                )
                ll = np.log(self.cpts_[s][c, ev[s]] + 1e-10)
                contribs[c][name] = float(w[s] * ll)
                log_p[c] += w[s] * ll

        log_p -= log_p.max()
        p = np.exp(log_p)
        p /= p.sum()

        ev_bins = {
            SIGNAL_NAMES[s]: self.bin_names_[ev[s]]
            for s in range(self.n_signals_)
        }
        return {
            "mode": "hard",
            "evidence_bins": ev_bins,
            "posterior_task": float(p[1]),
            "posterior_rest": float(p[0]),
            "log_likelihoods": contribs,
            "signal_weights": self.get_signal_weights(),
        }

    def _explain_sample_soft(self, probs_single: np.ndarray) -> dict:
        """
        Soft-evidence inference trace - mirrors _predict_soft() exactly.

        Soft evidence mode:
        We do not use a single hard bin as evidence.
        Instead, the effective likelihood for each signal is the expectation
        of CPT likelihoods under the Gaussian membership distribution over bins.
        """
        p_arr = probs_single.reshape(1, -1)
        # shape (1, S, n_bins)
        mem = self._soft_membership(p_arr)
        w = self.signal_weights_

        log_p = np.zeros(2)
        contribs: dict = {0: {}, 1: {}}
        effective_likelihoods: dict = {"rest": {}, "task": {}}

        for c in range(2):
            cls = "rest" if c == 0 else "task"
            log_p[c] = np.log(self.prior_[c] + 1e-10)
            for s in range(self.n_signals_):
                name = SIGNAL_NAMES[s] if s < len(SIGNAL_NAMES) else f"S{s}"
                # Soft evidence mode:
                # We do not use a single hard bin as evidence.
                # Instead, the effective likelihood for each signal is the expectation
                # of CPT likelihoods under the Gaussian membership distribution over bins.
                p_ev = float(np.dot(mem[0, s, :], self.cpts_[s][c, :]))
                effective_likelihoods[cls][name] = p_ev
                ll = w[s] * np.log(p_ev + 1e-10)
                contribs[c][name] = float(ll)
                log_p[c] += ll

        log_p -= log_p.max()
        p = np.exp(log_p)
        p /= p.sum()

        # Build anfis_probs dict
        anfis_probs: dict = {
            (SIGNAL_NAMES[s] if s < len(SIGNAL_NAMES) else f"S{s}"): float(probs_single[s])
            for s in range(self.n_signals_)
        }

        # Soft memberships
        soft_memberships: dict = {}
        for s in range(self.n_signals_):
            name = SIGNAL_NAMES[s] if s < len(SIGNAL_NAMES) else f"S{s}"
            soft_memberships[name] = {
                self.bin_names_[b]: float(mem[0, s, b])
                for b in range(self.n_bins)
            }

        # Dominant bin - argmax of soft membership, for visualization only.
        # In soft mode, evidence_bins only means dominant bin for visualization.
        # Actual inference uses soft_memberships.
        dominant_bins = {
            (SIGNAL_NAMES[s] if s < len(SIGNAL_NAMES) else f"S{s}"): self.bin_names_[int(np.argmax(mem[0, s, :]))]
            for s in range(self.n_signals_)
        }

        # log_likelihood_ratio[s] = LL_task(s) - LL_rest(s)
        # Positive -> supports TASK; negative -> supports REST.
        log_likelihood_ratio = {
            (SIGNAL_NAMES[s] if s < len(SIGNAL_NAMES) else f"S{s}"): (
                contribs[1][SIGNAL_NAMES[s] if s < len(SIGNAL_NAMES) else f"S{s}"]
                - contribs[0][SIGNAL_NAMES[s] if s < len(SIGNAL_NAMES) else f"S{s}"]
            )
            for s in range(self.n_signals_)
        }

        return {
            "mode": "soft",
            "anfis_probs": anfis_probs,
            # dominant_bins: argmax-bin per signal, for visualization only.
            "dominant_bins": dominant_bins,
            # In soft mode, evidence_bins only means dominant bin for visualization.
            # Actual inference uses soft_memberships.
            "evidence_bins": dominant_bins,
            "soft_memberships": soft_memberships,
            "effective_likelihoods": effective_likelihoods,
            "log_likelihoods": contribs,
            "log_likelihood_ratio": log_likelihood_ratio,
            "posterior_task": float(p[1]),
            "posterior_rest": float(p[0]),
            "final_posterior_task": float(p[1]),
            "final_posterior_rest": float(p[0]),
            "signal_weights": self.get_signal_weights(),
        }
