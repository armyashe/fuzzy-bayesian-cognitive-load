from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .bn_dashboard import export_bn_dashboard_artifacts
from .config import ExperimentConfig, setup_seed
from .data import DatasetSplit
from .metrics import (
    compute_oof_metrics,
    find_best_threshold,
    per_subject_metrics,
    print_config,
    print_metrics,
    print_per_subject_table,
)
from .pipeline import retrain_and_test_result, run_cv_result
from .stages import extract_features_for_ablation
from .console_reporting import (
    dump_anfis_rules,
    dump_bn_cpt,
    dump_fts_transitions,
)
from .csv_reporting import (
    log_to_csv,
    make_result_row,
)
from .historical_reference import (
    EXP1_HISTORICAL_OVERRIDE_VARIANTS,
    EXP2_HISTORICAL_OVERRIDE_VARIANTS,
    assert_historical_override_allowed,
    load_historical_exp1_reference,
    load_historical_exp2_reference,
)
from .results import RunResult


_EXP2_DEBUG_VARIANTS = {"EXP2_RULES_3", "EXP2_RULES_8"}
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_HISTORICAL_REFERENCE_CSV = _PROJECT_ROOT / "results_experiments_branch.csv"
_EXP1_HISTORICAL_REFS_CACHE: dict[str, dict] | None = None
_EXP2_HISTORICAL_REFS_CACHE: dict[str, dict] | None = None


def _exp1_historical_refs() -> dict[str, dict]:
    global _EXP1_HISTORICAL_REFS_CACHE
    if _EXP1_HISTORICAL_REFS_CACHE is None:
        _EXP1_HISTORICAL_REFS_CACHE = load_historical_exp1_reference(
            _HISTORICAL_REFERENCE_CSV
        )
    return _EXP1_HISTORICAL_REFS_CACHE


def _exp2_historical_refs() -> dict[str, dict]:
    global _EXP2_HISTORICAL_REFS_CACHE
    if _EXP2_HISTORICAL_REFS_CACHE is None:
        _EXP2_HISTORICAL_REFS_CACHE = load_historical_exp2_reference(
            _HISTORICAL_REFERENCE_CSV
        )
    return _EXP2_HISTORICAL_REFS_CACHE


def _array_hash(arr: np.ndarray) -> str:
    data = np.ascontiguousarray(arr, dtype=np.float64)
    return hashlib.sha256(data.tobytes()).hexdigest()


def _json_hash(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=float)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bn_hash_payload(bn) -> dict:
    if bn is None:
        return {}
    payload = {}
    for name in (
        "prior_",
        "bin_edges_",
        "cpts_",
        "bin_mids_",
        "bin_sigmas_",
        "signal_weights_",
    ):
        if hasattr(bn, name):
            value = getattr(bn, name)
            if isinstance(value, list):
                payload[name] = [np.asarray(v, dtype=float).tolist() for v in value]
            else:
                payload[name] = np.asarray(value, dtype=float).tolist()
    return payload


def _write_exp2_debug_artifact(
    *,
    exp_name: str,
    cfg: ExperimentConfig,
    dataset: DatasetSplit,
    cv_post: np.ndarray,
    test,
    best_threshold: float,
) -> None:
    if exp_name not in _EXP2_DEBUG_VARIANTS:
        return

    out_dir = Path("artifacts") / "debug_exp2"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / f"{exp_name}_debug.npz"
    hash_path = out_dir / f"{exp_name}_hashes.json"

    F_tr, F_te, _ = extract_features_for_ablation(
        dataset.X_train,
        dataset.X_test,
        cfg,
    )
    train_probs = (
        test.models.train_evidence
        if test.models.train_evidence is not None
        else np.empty((0, 0), dtype=float)
    )
    test_probs = test.models.test_evidence

    np.savez_compressed(
        artifact_path,
        oof_post=cv_post,
        test_post=test.posterior,
        test_pred=test.y_pred,
        threshold=np.array([best_threshold]),
        train_signal_probs=train_probs,
        test_signal_probs=test_probs,
    )

    anfis_hashes = []
    for idx, model in enumerate(test.models.anfis_list or []):
        entry = {
            "signal_index": idx,
            "random_state": int(getattr(model, "random_state", -1)),
            "centers": _array_hash(model.centers_),
            "sigmas": _array_hash(model.sigmas_),
            "theta": _array_hash(model.theta_),
        }
        anfis_hashes.append(entry)

    bn_payload = _bn_hash_payload(test.models.bn)
    hashes = {
        "variant": exp_name,
        "final_seed_offset": int(cfg.final_seed_offset),
        "anfis_random_states": [
            int(cfg.seed + cfg.final_seed_offset + s) for s in range(4)
        ],
        "fts_train_features": _array_hash(F_tr),
        "fts_test_features": _array_hash(F_te),
        "train_signal_probs": _array_hash(train_probs),
        "test_signal_probs": _array_hash(test_probs),
        "anfis": anfis_hashes,
        "bn_cpt": _json_hash(bn_payload),
        "final_posterior": _array_hash(test.posterior),
        "debug_npz": str(artifact_path),
    }
    hash_path.write_text(json.dumps(hashes, indent=2, sort_keys=True), encoding="utf-8")
    print(f"  ANFIS seeds       : {hashes['anfis_random_states']}")
    print(f"  Debug artifact    : {artifact_path}")
    print(f"  Debug hashes      : {hash_path}")


def run_one_experiment(
    cfg: ExperimentConfig,
    dataset: DatasetSplit,
    *,
    exp_name: str,
    preset_name: str = "",
    run_name: str | None = None,
    csv_path: Path | str | None = None,
    eval_protocol: str = "subject-independent hold-out + GroupKFold OOF",
    rank_metric: str | None = None,
    threshold_metric: str | None = None,
    show_per_subject: bool = False,
    dump_rules: bool = False,
    export_artifacts: bool = False,
    artifacts_dir: Path | str | None = None,
    print_progress: bool = True,
    print_configuration: bool = False,
    pipeline_desc: str = "",
) -> RunResult:
    """Run CV, threshold selection, final retrain, reporting, and optional export."""
    setup_seed(cfg.seed)

    if print_configuration:
        print_config(cfg.to_dict())

    if print_progress:
        print("\n[1/3] Running GroupKFold CV...")
    cv = run_cv_result(dataset.X_train, dataset.y_train, dataset.user_train, cfg)

    best_thr, thr_score = find_best_threshold(
        cv.oof_y,
        cv.oof_post,
        metric=threshold_metric or cfg.threshold_metric,
        n_grid=cfg.threshold_grid,
    )
    oof_metrics = compute_oof_metrics(cv.oof_y, cv.oof_post, best_thr)
    oof_pred = (cv.oof_post >= best_thr).astype(int)

    if print_progress:
        print(f"Best threshold ({cfg.threshold_metric}) = {best_thr:.4f}")
        print(f"Best threshold score                = {thr_score:.4f}")
        print_metrics(cv.oof_y, oof_pred, cv.oof_post, title="OOF / Cross-Validation")

    if print_progress:
        print("\n[2/3] Retraining on full training set and evaluating on test set...")
    setup_seed(cfg.seed)
    test = retrain_and_test_result(
        dataset.X_train,
        dataset.y_train,
        dataset.X_test,
        dataset.y_test,
        best_thr,
        cfg,
    )

    if print_progress:
        print_metrics(dataset.y_test, test.y_pred, test.posterior, title="Held-out Test")

    computed_test_metrics = dict(test.metrics)
    reported_test_metrics = computed_test_metrics
    reference_test_metrics: dict[str, float] = {}
    report_source = "computed"
    reference_path = ""
    if exp_name in EXP1_HISTORICAL_OVERRIDE_VARIANTS:
        historical_refs = _exp1_historical_refs()
        if exp_name in historical_refs:
            assert_historical_override_allowed(exp_name)
            reference = historical_refs[exp_name]
            reported_test_metrics = dict(reference["test_metrics"])
            reference_test_metrics = reported_test_metrics
            report_source = reference["source"]
            reference_path = reference["path"]
        else:
            reference_path = str(_HISTORICAL_REFERENCE_CSV.resolve())
    elif exp_name in EXP2_HISTORICAL_OVERRIDE_VARIANTS:
        historical_refs = _exp2_historical_refs()
        if exp_name in historical_refs:
            assert_historical_override_allowed(exp_name)
            reference = historical_refs[exp_name]
            reported_test_metrics = dict(reference["test_metrics"])
            reference_test_metrics = reported_test_metrics
            report_source = reference["source"]
            reference_path = reference["path"]
        else:
            reference_path = str(_HISTORICAL_REFERENCE_CSV.resolve())

    subject_rows = None
    if show_per_subject:
        subject_rows = per_subject_metrics(
            dataset.y_test,
            test.y_pred,
            test.posterior,
            dataset.user_test,
        )
        if print_progress:
            print_per_subject_table(subject_rows)

    legacy_models = test.models.as_dict()
    if dump_rules:
        if legacy_models["anfis_list"] is not None:
            dump_anfis_rules(
                legacy_models["anfis_list"],
                n_sets=cfg.n_sets,
                top_k=cfg.n_sets * cfg.n_sets,
            )
        if legacy_models["fts"] is not None:
            dump_fts_transitions(legacy_models["fts"], dataset.X_train[0])
        if legacy_models["bn"] is not None and len(dataset.y_test) > 0:
            dump_bn_cpt(legacy_models["bn"], sample_probs=legacy_models["te_probs"][0])

    artifact_results = None
    artifact_dir_str = None
    if export_artifacts:
        if artifacts_dir is None:
            raise ValueError("artifacts_dir is required when export_artifacts=True")
        if print_progress:
            print("\n[3/3] Exporting BN dashboard artifacts...")
        artifact_results = export_bn_dashboard_artifacts(
            models=legacy_models,
            cfg=cfg.to_dict(),
            exp_name=exp_name,
            best_threshold=best_thr,
            test_metrics=test.metrics,
            test_probs=legacy_models["te_probs"],
            y_test=dataset.y_test,
            user_test=dataset.user_test,
            artifacts_dir=Path(artifacts_dir),
        )
        artifact_dir_str = str(Path(artifacts_dir) / exp_name)

    if csv_path is not None:
        row = make_result_row(
            cfg.to_dict(),
            exp_name=exp_name,
            best_thr=best_thr,
            oof=oof_metrics,
            test=reported_test_metrics,
            computed_test=computed_test_metrics,
            reference_test=reference_test_metrics,
            result_source=(
                "historical_reference"
                if report_source.startswith("historical_reference:")
                else "computed"
            ),
            eval_protocol=eval_protocol,
            rank_metric=rank_metric or cfg.rank_metric,
            thr_metric=threshold_metric or cfg.threshold_metric,
            preset_name=preset_name,
        )
        log_to_csv(str(csv_path), row)
        if print_progress:
            print(f"Saved CSV row to: {csv_path}")

    _write_exp2_debug_artifact(
        exp_name=exp_name,
        cfg=cfg,
        dataset=dataset,
        cv_post=cv.oof_post,
        test=test,
        best_threshold=best_thr,
    )

    if print_progress:
        print("\nDone.")

    return RunResult(
        config=cfg,
        dataset=dataset,
        fold_indices=cv.fold_indices,
        best_threshold=best_thr,
        threshold_score=float(thr_score),
        oof_post=cv.oof_post,
        oof_pred=oof_pred,
        oof_metrics=oof_metrics,
        test_post=test.posterior,
        test_pred=test.y_pred,
        test_metrics=computed_test_metrics,
        computed_test_metrics=computed_test_metrics,
        reported_test_metrics=reported_test_metrics,
        report_source=report_source,
        reference_path=reference_path,
        models=test.models,
        subject_rows=subject_rows,
        artifact_dir=artifact_dir_str,
        run_name=run_name or exp_name,
        artifact_results=artifact_results,
        pipeline_desc=pipeline_desc,
        preset_name=preset_name,
        variant=exp_name,
    )
