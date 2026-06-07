from __future__ import annotations

from .models import SIGNAL_NAMES


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


def dump_anfis_rules(anfis_models, n_sets: int, top_k: int | None = None) -> None:
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


def dump_fts_transitions(fts, X_sample, top_k: int | None = None) -> None:
    n_sets = fts.n_sets
    if n_sets == 3:
        labels = ["L", "M", "H"]
    elif n_sets == 4:
        labels = ["VL", "L", "M", "H"]
    elif n_sets == 5:
        labels = ["VL", "L", "M", "H", "VH"]
    else:
        labels = [f"S{k}" for k in range(n_sets)]

    title = (
        "FTS TRANSITION RULES  (all transitions)"
        if top_k is None
        else f"FTS TRANSITION RULES  (top {top_k} transitions)"
    )

    print("=" * 68)
    print(f"  {title}")
    print("=" * 68)
    for s, name in enumerate(SIGNAL_NAMES):
        trans = fts.get_transition_matrix(X_sample, s)
        triples = []
        for i in range(n_sets):
            for j in range(n_sets):
                triples.append((labels[i], labels[j], trans[i, j]))
        triples.sort(key=lambda x: x[2], reverse=True)
        triples_to_print = triples if top_k is None else triples[:top_k]
        print(f"\n--- FTS Transitions - {name} ---")
        for rank, (fr, to, p) in enumerate(triples_to_print, 1):
            print(f"  T{rank}: IF state(t) is {fr} THEN state(t+1) is {to}  (p={p:.4f})")
    print()


def dump_bn_cpt(bn, sample_probs=None) -> None:
    """
    Print BN details.  Dispatches to dump_tan_bn() if bn is a TANFusion.
    """
    # Import here to avoid circular dependency at module load time
    from .models.bn_dependent import TANFusion  # noqa: PLC0415
    if isinstance(bn, TANFusion):
        dump_tan_bn(bn, sample_probs=sample_probs)
        return

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
        trace = bn.explain_sample(sample_probs)
        _print_bn_trace(trace, sample_probs)
    print()


def dump_tan_bn(bn, sample_probs=None) -> None:
    """
    Print TAN (TANFusion) BN details including learned structure,
    CMI matrix, CPTs, and an optional sample inference trace.
    """
    structure = bn.get_structure()

    print("=" * 68)
    print("  TREE-AUGMENTED NAIVE BAYES (TAN) - STRUCTURE")
    print("=" * 68)
    print(f"\n  Root evidence node : {structure['root']}")
    print("\n  Directed edges:")
    for parent, child in structure["edges"]:
        print(f"    {parent} -> {child}")

    print("\n  Evidence parent map:")
    for sig, par in structure["parent_map"].items():
        par_str = par if par is not None else "(root)"
        print(f"    E_{sig}  parent = {par_str}")

    print("\n  Conditional Mutual Information matrix  I(E_i ; E_j | Task):")
    signals = [s for s in structure["cmi_matrix"]]
    header = "  {:>6s}".format("")
    for s in signals:
        header += f"  {s:>7s}"
    print(header)
    for s_i in signals:
        row = f"  {s_i:>6s}"
        for s_j in signals:
            row += f"  {structure['cmi_matrix'][s_i][s_j]:>7.4f}"
        print(row)

    print("\n" + "=" * 68)
    print("  TAN - CPT SUMMARY  (child nodes: averaged over parent bins)")
    print("=" * 68)
    cpt = bn.get_cpt_table()
    for sig_name, table in cpt.items():
        if sig_name == structure["root"]:
            print(f"\n--- CPT - E_{sig_name}  (root) ---")
        else:
            par = table.get("parent_signal", "?")
            print(f"\n--- CPT - E_{sig_name}  (parent: E_{par}) ---")
        header = "  {:>12s}".format("Class")
        for bname in bn.bin_names_:
            header += f"  {bname:>8s}"
        print(header)
        for cls in ("rest", "task"):
            if cls not in table:
                continue
            row = "  {:>12s}".format(cls)
            for bname in bn.bin_names_:
                row += f"  {table[cls][bname]:>8.4f}"
            print(row)

    print(f"\n  Prior: P(rest)={bn.prior_[0]:.4f}  P(task)={bn.prior_[1]:.4f}")

    if sample_probs is not None:
        trace = bn.explain_sample(sample_probs)
        _print_bn_trace(trace, sample_probs)
    print()


def _print_bn_trace(trace: dict, sample_probs=None) -> None:
    """
    Print a BN inference trace dict in a mode-aware way.
    Handles both hard evidence (mode='hard') and soft evidence (mode='soft').
    """
    mode = trace.get("mode", "hard")
    print(f"\n--- BN Inference Trace (sample) ---")
    print(f"Mode: {'soft evidence' if mode == 'soft' else 'hard evidence'}")

    if sample_probs is not None:
        print("\nANFIS probabilities:")
        for s, name in enumerate(SIGNAL_NAMES):
            if s < len(sample_probs):
                print(f"  {name:<5s}: {float(sample_probs[s]):.4f}")

    if mode == "soft":
        print("\nDominant bins (visualization only - not the actual evidence):")
        for sig, b in trace.get("dominant_bins", trace.get("evidence_bins", {})).items():
            print(f"  {sig:<5s}: {b}")

        sm = trace.get("soft_memberships", {})
        if sm:
            print("\nSoft memberships:")
            for sig, bins in sm.items():
                parts = ", ".join(f"{k}={v:.4f}" for k, v in bins.items())
                print(f"  {sig:<5s}: {parts}")

        eff = trace.get("effective_likelihoods", {})
        if eff:
            print("\nEffective likelihoods P_eff(e_s|C) = sum_b mu_b * P(b|C):")
            for cls in ("rest", "task"):
                print(f"  Class={cls}:")
                for sig, val in eff.get(cls, {}).items():
                    print(f"    {sig:<5s}: {val:.6f}")

        llr = trace.get("log_likelihood_ratio", {})
        if llr:
            print("\nLog-likelihood ratio task/rest (+ supports TASK, - supports REST):")
            for sig, val in llr.items():
                print(f"  {sig:<5s}: {val:+.4f}")

        print(f"\nPosterior:")
        print(f"  P(task|soft evidence): {trace['posterior_task']:.4f}")
        print(f"  P(rest|soft evidence): {trace['posterior_rest']:.4f}")

    else:
        print("\nEvidence bins:")
        for sig, b in trace.get("evidence_bins", {}).items():
            print(f"  {sig:<5s}: {b}")

        print(f"\n  P(task|evidence): {trace['posterior_task']:.4f}")
        print(f"  P(rest|evidence): {trace['posterior_rest']:.4f}")
        print("  Log-likelihoods:")
        for c in (0, 1):
            cls = "rest" if c == 0 else "task"
            print(f"    Class={cls}: {trace['log_likelihoods'].get(c, {})}")
