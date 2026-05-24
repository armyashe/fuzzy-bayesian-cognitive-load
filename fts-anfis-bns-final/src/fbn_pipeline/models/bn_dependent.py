"""
Tree-Augmented Naive Bayes (TAN) Fusion
========================================
Experimental variant of BNFusion that allows evidence nodes to depend on
each other in addition to the class node Task.

DAG structure (TAN):
    Task -> E_Temp
    Task -> E_HR
    Task -> E_GSR
    Task -> E_RR
    + evidence-to-evidence edges forming a directed tree rooted at a chosen
      evidence node (default: "HR"), learned from training data only.

Factorization:
    P(c | e) ∝ P(c) × P(e_root | c) × ∏_{s ≠ root} P(e_s | c, e_parent(s))

Structure learning:
    The tree over evidence nodes is the maximum spanning tree of the
    complete graph where edge weight = I(E_i ; E_j | Task), the pairwise
    conditional mutual information computed from **discretised training data
    only**.  Edges are directed outward from a fixed root (default: "HR").

CPT estimation (Laplace-smoothed counting on training fold only):
    Root:  P(E_root = b | Task = c)              shape (2, n_bins)
    Child: P(E_s = b | Task = c, E_parent = p)   shape (2, n_bins, n_bins)
           index order: [class_idx, parent_bin, child_bin]

Limitations:
    - Signal weights (reliability weighting) are not supported.  The TAN
      factorization already encodes signal dependencies through the tree
      structure; an additional per-signal scalar weight has no principled
      place in this factorization.  Passing signal_weights= is accepted for
      API compatibility but has no effect.
    - Only hard evidence is implemented.  Soft (Gaussian-kernel) evidence is
      not extended to TAN because the 3-way CPT (class, parent_bin, child_bin)
      makes the integral over parent membership significantly more expensive
      and less interpretable.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .bn_fusion import SIGNAL_NAMES, _make_bin_names

# Map signal name -> column index in the (N, 4) ANFIS prob matrix
_SIGNAL_IDX: dict[str, int] = {name: i for i, name in enumerate(SIGNAL_NAMES)}


# ------------------------------------------------------------------
# Conditional Mutual Information
# ------------------------------------------------------------------

def _cmi(
    ev_i: np.ndarray,
    ev_j: np.ndarray,
    y: np.ndarray,
    n_bins: int,
) -> float:
    """
    Empirical I(E_i ; E_j | Task) from discretised evidence.

    Uses small additive smoothing (1e-10) so that zero-count cells do
    not produce -inf or NaN contributions.
    """
    alpha = 1e-10
    # joint counts: shape (2, n_bins, n_bins)
    counts = np.full((2, n_bins, n_bins), alpha)
    for k in range(len(y)):
        counts[int(y[k]), int(ev_i[k]), int(ev_j[k])] += 1.0

    total = counts.sum()
    count_c  = counts.sum(axis=(1, 2))          # (2,)
    count_ci = counts.sum(axis=2)               # (2, n_bins)  sum over j
    count_cj = counts.sum(axis=1)               # (2, n_bins)  sum over i

    cmi = 0.0
    for c in range(2):
        for bi in range(n_bins):
            for bj in range(n_bins):
                p_cbij  = counts[c, bi, bj] / total
                p_bij_c = counts[c, bi, bj] / count_c[c]
                p_bi_c  = count_ci[c, bi]   / count_c[c]
                p_bj_c  = count_cj[c, bj]   / count_c[c]
                if p_cbij > 0 and p_bij_c > 0 and p_bi_c > 0 and p_bj_c > 0:
                    cmi += p_cbij * np.log(p_bij_c / (p_bi_c * p_bj_c))
    return float(cmi)


# ------------------------------------------------------------------
# Maximum Spanning Tree (Kruskal's)
# ------------------------------------------------------------------

def _max_spanning_tree(n: int, weights: np.ndarray) -> list[tuple[int, int]]:
    """
    Return edges of the maximum spanning tree over n nodes.

    Parameters
    ----------
    weights : (n, n) symmetric weight matrix (higher = keep edge)
    """
    # collect all undirected edges sorted by weight descending
    edges = sorted(
        [(weights[i, j], i, j) for i in range(n) for j in range(i + 1, n)],
        reverse=True,
    )

    # Union-Find with path compression
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> bool:
        ra, rb = _find(a), _find(b)
        if ra == rb:
            return False
        parent[ra] = rb
        return True

    mst: list[tuple[int, int]] = []
    for _, u, v in edges:
        if _union(u, v):
            mst.append((u, v))
        if len(mst) == n - 1:
            break
    return mst


# ------------------------------------------------------------------
# Tree orientation (BFS from root)
# ------------------------------------------------------------------

def _orient_tree(
    mst_edges: list[tuple[int, int]],
    root: int,
    n: int,
) -> dict[int, int | None]:
    """
    Orient MST edges away from root via BFS.

    Returns
    -------
    parent_map : {node_idx -> parent_idx or None if root}
    """
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for u, v in mst_edges:
        adj[u].append(v)
        adj[v].append(u)

    parent_map: dict[int, int | None] = {root: None}
    visited = {root}
    queue: deque[int] = deque([root])
    while queue:
        node = queue.popleft()
        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb)
                parent_map[nb] = node
                queue.append(nb)
    return parent_map


# ------------------------------------------------------------------
# TANFusion
# ------------------------------------------------------------------

class TANFusion:
    """
    Tree-Augmented Naive Bayes (TAN) for fusing 4 per-signal evidence scores.

    Compared to the naive BN (BNFusion), TANFusion adds a directed spanning
    tree over the evidence nodes whose edges represent the strongest pairwise
    conditional dependencies given the class.

    Parameters
    ----------
    n_bins : int
        Number of quantile bins for evidence discretisation.
    alpha : float
        Laplace smoothing coefficient for CPT estimation.
    root : str
        Name of the evidence node used as the tree root (default "HR").
        Must be one of SIGNAL_NAMES = ["Temp", "HR", "GSR", "RR"].
    """

    def __init__(
        self,
        n_bins: int = 5,
        alpha: float = 0.5,
        root: str = "HR",
        soft: bool = False,
    ) -> None:
        self.n_bins = n_bins
        self.alpha = alpha
        self.root = root
        self.soft = soft
        self.bin_names_ = _make_bin_names(n_bins)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        probs: np.ndarray,
        y: np.ndarray,
        signal_weights: np.ndarray | None = None,  # ignored — see module docstring
    ) -> "TANFusion":
        """
        Learn TAN structure and CPTs from training data only.

        Parameters
        ----------
        probs : (N, 4)  per-signal continuous evidence (ANFIS probs / FTS scores)
        y     : (N,)    binary labels {0, 1}
        signal_weights : ignored (kept for API compatibility with BNFusion)
        """
        N, S = probs.shape
        self.n_signals_ = S

        # 1. Quantile bin edges (fit on training data only — no leakage)
        self.bin_edges_: list[np.ndarray] = []
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        for s in range(S):
            edges = np.quantile(probs[:, s], qs)
            self.bin_edges_.append(edges)

        evidence = self._discretise(probs)  # (N, 4)

        # 2. Prior P(Task = c)
        self.prior_ = np.zeros(2)
        for c in range(2):
            self.prior_[c] = (np.sum(y == c) + self.alpha) / (N + 2 * self.alpha)

        # 3. Pairwise conditional mutual information matrix
        cmi_matrix = np.zeros((S, S))
        for i in range(S):
            for j in range(i + 1, S):
                val = _cmi(evidence[:, i], evidence[:, j], y, self.n_bins)
                cmi_matrix[i, j] = val
                cmi_matrix[j, i] = val
        self.cmi_matrix_ = cmi_matrix

        # 4. Maximum spanning tree over evidence nodes
        self.mst_edges_ = _max_spanning_tree(S, cmi_matrix)

        # 5. Orient tree from root
        root_idx = _SIGNAL_IDX.get(self.root, 1)  # fall back to HR (=1) if unknown
        self.root_idx_ = root_idx
        self.parent_map_: dict[int, int | None] = _orient_tree(
            self.mst_edges_, root_idx, S
        )

        # 6. CPT estimation
        self.cpt_root_: np.ndarray = self._fit_root_cpt(evidence, y)
        self.cpts_child_: dict[int, np.ndarray] = {
            s: self._fit_child_cpt(evidence, y, s, self.parent_map_[s])  # type: ignore[arg-type]
            for s in range(S)
            if s != root_idx
        }

        # 7. Soft evidence parameters (fit on train probs only)
        if self.soft:
            self._compute_soft_params(probs)

        return self

    # ---------- CPT helpers ----------

    def _fit_root_cpt(self, evidence: np.ndarray, y: np.ndarray) -> np.ndarray:
        """P(E_root = b | Task = c), shape (2, n_bins)."""
        r = self.root_idx_
        cpt = np.zeros((2, self.n_bins))
        for c in range(2):
            mask = y == c
            for b in range(self.n_bins):
                cpt[c, b] = np.sum(evidence[mask, r] == b) + self.alpha
            cpt[c] /= cpt[c].sum()
        return cpt

    def _fit_child_cpt(
        self,
        evidence: np.ndarray,
        y: np.ndarray,
        child: int,
        parent: int,
    ) -> np.ndarray:
        """
        P(E_child = b | Task = c, E_parent = p_val)
        shape (2, n_bins, n_bins)  —  [class, parent_bin, child_bin]
        """
        cpt = np.zeros((2, self.n_bins, self.n_bins))
        for c in range(2):
            for p_val in range(self.n_bins):
                mask = (y == c) & (evidence[:, parent] == p_val)
                for b in range(self.n_bins):
                    cpt[c, p_val, b] = np.sum(evidence[mask, child] == b) + self.alpha
                cpt[c, p_val] /= cpt[c, p_val].sum()
        return cpt

    # ------------------------------------------------------------------
    # Soft evidence helpers  (same approach as BNFusion)
    # ------------------------------------------------------------------

    def _compute_soft_params(self, probs: np.ndarray) -> None:
        """Compute bin midpoints and Gaussian kernel σ for soft evidence."""
        self.bin_mids_: list[np.ndarray] = []
        self.bin_sigmas_: list[float] = []
        for s in range(self.n_signals_):
            edges = self.bin_edges_[s]
            p_min = float(probs[:, s].min())
            p_max = float(probs[:, s].max())
            ext = np.concatenate([[p_min], edges, [p_max]])
            mids = 0.5 * (ext[:-1] + ext[1:])
            avg_width = (p_max - p_min) / self.n_bins if self.n_bins > 0 else 0.1
            sigma = max(avg_width * 0.5, 1e-6)
            self.bin_mids_.append(mids)
            self.bin_sigmas_.append(sigma)

    def _soft_membership(self, probs: np.ndarray) -> np.ndarray:
        """
        Gaussian kernel soft membership over bins.

        Returns
        -------
        mem : ndarray, shape (N, S, n_bins)  — row-normalised per signal
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
        return self._discretise(probs)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        """
        Posterior P(Task=1 | evidence) for each sample.

        Dispatches to soft or hard evidence path based on self.soft.

        Factorization (hard):
            P(c | e) ∝ P(c) × P(e_root | c)
                           × ∏_{s ≠ root} P(e_s | c, e_parent(s))

        Factorization (soft):
            L(c) = Σ_b [Π_s μ_{b_s}(v_s)] · P(b | c)  (TAN joint)
            P(c | v) ∝ P(c) × L(c)

        Parameters
        ----------
        probs : (N, 4)

        Returns
        -------
        posteriors : (N,)
        """
        if self.soft:
            return self._predict_soft(probs)
        return self._predict_hard(probs)

    def _predict_hard(self, probs: np.ndarray) -> np.ndarray:
        evidence = self._discretise(probs)
        N = evidence.shape[0]
        posteriors = np.zeros(N)

        for i in range(N):
            log_p = np.zeros(2)
            for c in range(2):
                log_p[c] = np.log(self.prior_[c] + 1e-10)

                # Root term: P(e_root | c)
                b_root = evidence[i, self.root_idx_]
                log_p[c] += np.log(self.cpt_root_[c, b_root] + 1e-10)

                # Child terms: P(e_s | c, e_parent(s))
                for s in range(self.n_signals_):
                    if s == self.root_idx_:
                        continue
                    p = self.parent_map_[s]
                    b_s = evidence[i, s]
                    b_p = evidence[i, p]  # type: ignore[index]
                    log_p[c] += np.log(self.cpts_child_[s][c, b_p, b_s] + 1e-10)

            # Stable softmax
            log_p -= log_p.max()
            q = np.exp(log_p)
            q /= q.sum()
            posteriors[i] = q[1]

        return posteriors

    def _predict_soft(self, probs: np.ndarray) -> np.ndarray:
        """
        Soft-evidence TAN inference.

        Marginalises over all n_bins^S bin combinations:
            L(c) = Σ_b [Π_s μ_{b_s}(v_s)] · P(b_root|c) · Π_{s≠root} P(b_s|c,b_pa(s))

        The joint P(b|c) under the TAN factorisation sums to 1 for each c,
        so L(c) is a proper soft likelihood in (0, 1].
        """
        mem = self._soft_membership(probs)  # (N, S, n_bins)
        N, S = probs.shape
        posteriors = np.zeros(N)

        # Enumerate all bin combinations once: shape (K, S), K = n_bins^S
        bins = np.arange(self.n_bins)
        grids = np.meshgrid(*([bins] * S), indexing="ij")
        b_combos = np.stack([g.ravel() for g in grids], axis=1)  # (K, S)

        # Precompute log P(b_combo | c) for each class: (2, K)
        # P(b|c) = P(b_root|c) × Π_{s≠root} P(b_s|c, b_parent(s))
        log_p_b_c = np.zeros((2, b_combos.shape[0]))
        for c in range(2):
            log_p_b_c[c] = np.log(
                self.cpt_root_[c, b_combos[:, self.root_idx_]] + 1e-10
            )
            for s in range(S):
                if s == self.root_idx_:
                    continue
                p_idx = self.parent_map_[s]
                log_p_b_c[c] += np.log(
                    self.cpts_child_[s][
                        c,
                        b_combos[:, p_idx],  # type: ignore[index]
                        b_combos[:, s],
                    ]
                    + 1e-10
                )
        # Normalise to get proper joint distribution per class
        p_b_c = np.exp(log_p_b_c - log_p_b_c.max(axis=1, keepdims=True))  # (2, K)
        p_b_c /= p_b_c.sum(axis=1, keepdims=True)

        for i in range(N):
            # Joint soft membership Π_s μ_{b_s}(v_s): shape (K,)
            joint_mem = np.ones(b_combos.shape[0])
            for s in range(S):
                joint_mem *= mem[i, s, b_combos[:, s]]

            log_p = np.zeros(2)
            for c in range(2):
                likelihood = float(np.dot(joint_mem, p_b_c[c]))
                log_p[c] = np.log(self.prior_[c] + 1e-10) + np.log(
                    likelihood + 1e-10
                )

            log_p -= log_p.max()
            q = np.exp(log_p)
            q /= q.sum()
            posteriors[i] = q[1]

        return posteriors

    def predict(self, probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(probs) >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------

    def get_structure(self) -> dict:
        """
        Return the learned TAN structure.

        Returns
        -------
        dict with keys:
            edges      : list of (parent_node, child_node) string tuples
            root       : name of the root evidence node
            parent_map : {signal_name -> parent_signal_name | None}
            cmi_matrix : pairwise CMI values as a nested dict
        """
        # All Task -> E_s edges
        edges: list[tuple[str, str]] = []
        for s in range(self.n_signals_):
            sig = SIGNAL_NAMES[s] if s < len(SIGNAL_NAMES) else f"S{s}"
            edges.append(("Task", f"E_{sig}"))

        # Evidence-to-evidence edges (from oriented spanning tree)
        for s, p in self.parent_map_.items():
            if p is not None:
                p_name = SIGNAL_NAMES[p] if p < len(SIGNAL_NAMES) else f"S{p}"
                s_name = SIGNAL_NAMES[s] if s < len(SIGNAL_NAMES) else f"S{s}"
                edges.append((f"E_{p_name}", f"E_{s_name}"))

        parent_map_named = {
            (SIGNAL_NAMES[s] if s < len(SIGNAL_NAMES) else f"S{s}"): (
                (SIGNAL_NAMES[p] if p < len(SIGNAL_NAMES) else f"S{p}")
                if p is not None
                else None
            )
            for s, p in self.parent_map_.items()
        }

        cmi_named = {
            (SIGNAL_NAMES[i] if i < len(SIGNAL_NAMES) else f"S{i}"): {
                (SIGNAL_NAMES[j] if j < len(SIGNAL_NAMES) else f"S{j}"): round(
                    float(self.cmi_matrix_[i, j]), 6
                )
                for j in range(self.n_signals_)
            }
            for i in range(self.n_signals_)
        }

        return {
            "edges": edges,
            "root": SIGNAL_NAMES[self.root_idx_] if self.root_idx_ < len(SIGNAL_NAMES) else f"S{self.root_idx_}",
            "parent_map": parent_map_named,
            "cmi_matrix": cmi_named,
        }

    def get_cpt_table(self) -> dict:
        """
        Return CPTs as a nested dict (API-compatible with BNFusion).

        For child nodes the CPT is averaged over parent bins for display
        (the full 3-way table is available via cpts_child_[s]).
        """
        tables: dict = {}
        for s in range(self.n_signals_):
            name = SIGNAL_NAMES[s] if s < len(SIGNAL_NAMES) else f"S{s}"
            if s == self.root_idx_:
                table: dict = {}
                for c in range(2):
                    cls = "rest" if c == 0 else "task"
                    table[cls] = {
                        self.bin_names_[b]: round(float(self.cpt_root_[c, b]), 4)
                        for b in range(self.n_bins)
                    }
            else:
                p = self.parent_map_[s]
                p_name = SIGNAL_NAMES[p] if p is not None and p < len(SIGNAL_NAMES) else str(p)
                table = {
                    "parent_signal": p_name,
                    "note": (
                        "P(E_s=b | c, E_parent=p) averaged over parent bins for display. "
                        "Full 3-way CPT available in cpts_child_ attribute."
                    ),
                }
                for c in range(2):
                    cls = "rest" if c == 0 else "task"
                    avg = self.cpts_child_[s][c].mean(axis=0)  # mean over parent bins
                    table[cls] = {
                        self.bin_names_[b]: round(float(avg[b]), 4)
                        for b in range(self.n_bins)
                    }
            tables[name] = table
        return tables

    def get_signal_weights(self) -> dict:
        """Signal weights are uniform (reliability weighting not used in TAN)."""
        return {
            (SIGNAL_NAMES[s] if s < len(SIGNAL_NAMES) else f"S{s}"): 1.0
            for s in range(self.n_signals_)
        }

    def explain_sample(self, probs_single: np.ndarray) -> dict:
        """
        Full inference trace for one sample.

        Parameters
        ----------
        probs_single : (4,)

        Returns
        -------
        dict with keys:
            evidence_bins    : {signal_name -> bin_name}
            posterior_task   : P(Task=1 | evidence)
            posterior_rest   : P(Task=0 | evidence)
            log_likelihoods  : {class_idx -> {signal_name -> log_contribution}}
            structure        : learned BN structure (see get_structure())
        """
        ev = self._discretise(probs_single.reshape(1, -1))[0]

        log_p = np.zeros(2)
        contribs: dict = {0: {}, 1: {}}
        for c in range(2):
            log_p[c] = np.log(self.prior_[c] + 1e-10)

            # Root
            root_name = SIGNAL_NAMES[self.root_idx_] if self.root_idx_ < len(SIGNAL_NAMES) else f"S{self.root_idx_}"
            b_root = ev[self.root_idx_]
            ll_root = np.log(self.cpt_root_[c, b_root] + 1e-10)
            contribs[c][root_name] = float(ll_root)
            log_p[c] += ll_root

            # Children
            for s in range(self.n_signals_):
                if s == self.root_idx_:
                    continue
                sig_name = SIGNAL_NAMES[s] if s < len(SIGNAL_NAMES) else f"S{s}"
                p = self.parent_map_[s]
                b_s = ev[s]
                b_p = ev[p]  # type: ignore[index]
                ll = np.log(self.cpts_child_[s][c, b_p, b_s] + 1e-10)
                contribs[c][sig_name] = float(ll)
                log_p[c] += ll

        log_p -= log_p.max()
        q = np.exp(log_p)
        q /= q.sum()

        return {
            "evidence_bins": {
                SIGNAL_NAMES[s]: self.bin_names_[ev[s]]
                for s in range(self.n_signals_)
            },
            "posterior_task": float(q[1]),
            "posterior_rest": float(q[0]),
            "log_likelihoods": contribs,
            "structure": self.get_structure(),
        }
