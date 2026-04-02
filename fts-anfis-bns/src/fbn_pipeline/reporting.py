from __future__ import annotations

import csv
import os
from typing import Any

from .models import SIGNAL_NAMES


CSV_COLUMNS = [
    "exp", "eval_protocol", "seed", "k_folds",
    "n_sets", "mf_type", "mf_sigma_scale", "gbell_b", "fts_norm",
    "n_rules", "anfis_sigma_scale", "ridge_lambda", "n_epochs", "lr",
    "n_bins", "alpha",
    "scale_fts", "robust_fts", "bn_soft", "bn_reliability", "calibrate",
    "rank_metric", "thr_metric", "best_thr",
    "oof_acc", "oof_bal_acc", "oof_macro_f1", "oof_f1", "oof_auc",
    "test_acc", "test_bal_acc", "test_macro_f1", "test_f1", "test_auc",
]


def log_to_csv(filepath: str, row: dict[str, Any]) -> None:
    file_exists = os.path.isfile(filepath)
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def make_result_row(
    cfg: dict[str, Any],
    exp_name: str,
    best_thr: float,
    *,
    oof: dict[str, Any] | None = None,
    test: dict[str, Any] | None = None,
    eval_protocol: str = "",
    rank_metric: str = "",
    thr_metric: str = "",
) -> dict[str, Any]:
    oof = oof or {}
    test = test or {}
    return {
        "exp": exp_name,
        "eval_protocol": eval_protocol,
        "seed": cfg.get("seed", ""),
        "k_folds": cfg.get("k_folds", ""),
        "n_sets": cfg["n_sets"],
        "mf_type": cfg["mf_type"],
        "mf_sigma_scale": cfg["mf_sigma_scale"],
        "gbell_b": cfg["gbell_b"],
        "fts_norm": cfg["fts_norm"],
        "n_rules": cfg["n_rules"],
        "anfis_sigma_scale": cfg["sigma_scale"],
        "ridge_lambda": cfg["ridge_lambda"],
        "n_epochs": cfg["n_epochs"],
        "lr": cfg["lr"],
        "n_bins": cfg["n_bins"],
        "alpha": cfg["alpha"],
        "scale_fts": cfg.get("scale_fts", False),
        "robust_fts": cfg.get("robust_fts", False),
        "bn_soft": cfg.get("bn_soft", False),
        "bn_reliability": cfg.get("bn_reliability", False),
        "calibrate": cfg.get("calibrate", "none"),
        "rank_metric": rank_metric,
        "thr_metric": thr_metric,
        "best_thr": best_thr,
        "oof_acc": oof.get("acc", ""),
        "oof_bal_acc": oof.get("bal_acc", ""),
        "oof_macro_f1": oof.get("macro_f1", ""),
        "oof_f1": oof.get("f1", ""),
        "oof_auc": oof.get("auc", ""),
        "test_acc": test.get("acc", ""),
        "test_bal_acc": test.get("bal_acc", ""),
        "test_macro_f1": test.get("macro_f1", ""),
        "test_f1": test.get("f1", ""),
        "test_auc": test.get("auc", ""),
    }


def feature_names_for_fts(n_sets: int) -> list[str]:
    if n_sets == 3:
        labels = ["L", "M", "H"]
    elif n_sets == 4:
        labels = ["VL", "L", "M", "H"]
    elif n_sets == 5:
        labels = ["VL", "L", "M", "H", "VH"]
    else:
        labels = [f"S{k}" for k in range(n_sets)]
    return [f"P({to_}|{from_})" for from_ in labels for to_ in labels]


def dump_anfis_rules(anfis_models, n_sets: int, top_k: int = 4) -> None:
    feat_names = feature_names_for_fts(n_sets)
    print("\n" + "=" * 68)
    print("  ANFIS RULE BASE  (IF-THEN rules per signal channel)")
    print("=" * 68)
    for s, name in enumerate(SIGNAL_NAMES):
        anfis = anfis_models[s]
        rules = anfis.rule_strings(feature_names=feat_names, top_k=top_k)
        print(f"\n--- ANFIS RULE BASE - {name} ({anfis.n_rules_} rules) ---")
        for rule in rules:
            print(f"  {rule}")
    print()


def dump_fts_transitions(fts, X_sample, top_k: int = 5) -> None:
    n_sets = fts.n_sets
    if n_sets == 3:
        labels = ["L", "M", "H"]
    elif n_sets == 4:
        labels = ["VL", "L", "M", "H"]
    elif n_sets == 5:
        labels = ["VL", "L", "M", "H", "VH"]
    else:
        labels = [f"S{k}" for k in range(n_sets)]

    print("=" * 68)
    print("  FTS TRANSITION RULES  (illustrative sample, top transitions)")
    print("=" * 68)
    for s, name in enumerate(SIGNAL_NAMES):
        trans = fts.get_transition_matrix(X_sample, s)
        triples = []
        for i in range(n_sets):
            for j in range(n_sets):
                triples.append((labels[i], labels[j], trans[i, j]))
        triples.sort(key=lambda x: x[2], reverse=True)
        print(f"\n--- FTS Transitions - {name} ---")
        for rank, (fr, to, p) in enumerate(triples[:top_k], 1):
            print(f"  T{rank}: IF state(t) is {fr} THEN state(t+1) is {to}  (p={p:.4f})")
    print()


def dump_bn_cpt(bn, sample_probs=None) -> None:
    print("=" * 68)
    print("  BAYESIAN NETWORK - CPT  (Conditional Probability Tables)")
    print("=" * 68)
    cpt = bn.get_cpt_table()
    for sig_name, table in cpt.items():
        print(f"\n--- CPT - E_{sig_name} ---")
        header = "  {:>12s}".format("Class")
        for bname in bn.bin_names_:
            header += f"  {bname:>8s}"
        print(header)
        for cls, bins in table.items():
            row = "  {:>12s}".format(cls)
            for bname in bn.bin_names_:
                row += f"  {bins[bname]:>8.4f}"
            print(row)

    weights = bn.get_signal_weights()
    print(f"\n  Signal weights: {weights}")
    print(f"  Prior: P(rest)={bn.prior_[0]:.4f}  P(task)={bn.prior_[1]:.4f}")

    if sample_probs is not None:
        print("\n--- BN Inference Trace (sample) ---")
        trace = bn.explain_sample(sample_probs)
        print(f"  Evidence bins   : {trace['evidence_bins']}")
        print(f"  P(task|evidence): {trace['posterior_task']:.4f}")
        print(f"  P(rest|evidence): {trace['posterior_rest']:.4f}")
        print("  Log-likelihoods :")
        for c in (0, 1):
            cls = "rest" if c == 0 else "task"
            print(f"    Class={cls}: {trace['log_likelihoods'][c]}")
    print()
