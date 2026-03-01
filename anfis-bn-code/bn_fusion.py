"""
Bayesian Network Fusion (Naive Bayes structure)
================================================
DAG:   Task  →  E_Temp,  E_HR,  E_GSR,  E_RR

Each evidence node E_s is a discretised version of the ANFIS probability
output for signal s.  Discretisation uses quantile-based binning (3 bins:
low / med / high) fitted on the **training fold** to avoid leakage.

CPTs are estimated by counting with Laplace smoothing (alpha = 1).

Inference is exact Naive Bayes:
    P(Task=c | e) ∝ P(Task=c) · ∏_s P(E_s = e_s | Task=c)
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
    """Naive Bayes BN for fusing 4 ANFIS probability outputs."""

    def __init__(self, n_bins: int = 3, alpha: float = 1.0):
        self.n_bins = n_bins
        self.alpha = alpha          # Laplace smoothing
        self.bin_names_ = _make_bin_names(n_bins)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, probs: np.ndarray, y: np.ndarray) -> "BNFusion":
        """
        Learn bin edges and CPTs from training-fold ANFIS outputs.

        Parameters
        ----------
        probs : ndarray, shape (N, 4)  — P(task) from 4 ANFIS models
        y     : ndarray, shape (N,)    — binary labels {0, 1}
        """
        N, S = probs.shape
        self.n_signals_ = S

        # --- Quantile bin edges (fit on train only) ---
        self.bin_edges_: list[np.ndarray] = []
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]  # e.g. [1/3, 2/3] for 3
        for s in range(S):
            edges = np.quantile(probs[:, s], qs)
            self.bin_edges_.append(edges)

        evidence = self._discretise(probs)  # (N, 4)  values in {0,1,2}

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

        return self

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

        Parameters
        ----------
        probs : ndarray, shape (N, 4)

        Returns
        -------
        posteriors : ndarray, shape (N,)
        """
        evidence = self._discretise(probs)
        N = evidence.shape[0]
        posteriors = np.zeros(N)

        for i in range(N):
            log_p = np.zeros(2)
            for c in range(2):
                log_p[c] = np.log(self.prior_[c] + 1e-10)
                for s in range(self.n_signals_):
                    log_p[c] += np.log(
                        self.cpts_[s][c, evidence[i, s]] + 1e-10
                    )
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

    def explain_sample(self, probs_single: np.ndarray) -> dict:
        """
        Full inference trace for one sample.

        Parameters
        ----------
        probs_single : ndarray, shape (4,)

        Returns
        -------
        dict with keys: evidence_bins, posterior_task, posterior_rest,
                        log_likelihoods
        """
        ev = self._discretise(probs_single.reshape(1, -1))[0]

        log_p = np.zeros(2)
        contribs: dict = {0: {}, 1: {}}
        for c in range(2):
            log_p[c] = np.log(self.prior_[c] + 1e-10)
            for s in range(self.n_signals_):
                name = (
                    SIGNAL_NAMES[s] if s < len(SIGNAL_NAMES) else f"S{s}"
                )
                ll = np.log(self.cpts_[s][c, ev[s]] + 1e-10)
                contribs[c][name] = float(ll)
                log_p[c] += ll

        log_p -= log_p.max()
        p = np.exp(log_p)
        p /= p.sum()

        return {
            "evidence_bins": {
                SIGNAL_NAMES[s]: self.bin_names_[ev[s]]
                for s in range(self.n_signals_)
            },
            "posterior_task": float(p[1]),
            "posterior_rest": float(p[0]),
            "log_likelihoods": contribs,
        }
