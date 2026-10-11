from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import os

import numpy as np
from threadpoolctl import threadpool_limits

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
    log_final_test_row,
    log_development_row,
    log_to_csv,
    make_development_result_row,
    make_final_test_result_row,
    make_result_row,
)
from .results import (
    DevelopmentResult,
    DevelopmentSelection,
    FinalEvaluationResult,
    RunResult,
)


_EXP2_DEBUG_VARIANTS = {"EXP2_RULES_3", "EXP2_RULES_8", "EXP2_RULES_12"}


def _array_hash(arr: np.ndarray) -> str:
    data = np.ascontiguousarray(arr, dtype=np.float64)
    return hashlib.sha256(data.tobytes()).hexdigest()


def _deterministic_thread_context():
    """Force deterministic linear algebra/KMeans reductions for experiments."""
    return threadpool_limits(limits=1)


def _json_hash(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=float)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _config_identifier(cfg: ExperimentConfig, configuration_id: str | None = None) -> str:
    if configuration_id:
        return configuration_id
    return cfg.variant_name or cfg.experiment_group or "config"


def _model_complexity(cfg: ExperimentConfig) -> tuple[int, int, int, str]:
    return (
        int(cfg.n_rules),
        int(cfg.n_sets),
        int(cfg.n_bins),
        _config_identifier(cfg),
    )


def _selection_metric_value(metrics: dict[str, float]) -> float:
    return float(metrics["bal_acc"])


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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
    cv_y: np.ndarray,
    test,
    best_threshold: float,
) -> None:
    if not cfg.dump_exp2_debug or exp_name not in _EXP2_DEBUG_VARIANTS:
        return

    out_dir = Path(cfg.exp2_debug_dir)
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
        oof_y=cv_y,
        test_post=test.posterior,
        test_pred=test.y_pred,
        test_y=dataset.y_test,
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


def run_development(
    X_train: np.ndarray,
    y_train: np.ndarray,
    user_train: np.ndarray,
    cfg: ExperimentConfig,
    *,
    configuration_id: str | None = None,
    threshold_metric: str | None = None,
    print_progress: bool = True,
    fold_indices: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> DevelopmentResult:
    """Run development-only GroupKFold OOF evaluation on training data."""
    setup_seed(cfg.seed)
    config_id = _config_identifier(cfg, configuration_id)

    if print_progress:
        print("\n[development] Running GroupKFold CV on training subjects only...")

    with _deterministic_thread_context():
        cv = run_cv_result(
            X_train,
            y_train,
            user_train,
            cfg,
            fold_indices=fold_indices,
        )

    threshold_metric_name = threshold_metric or cfg.threshold_metric
    best_thr, thr_score = find_best_threshold(
        cv.oof_y,
        cv.oof_post,
        metric=threshold_metric_name,
        n_grid=cfg.threshold_grid,
    )
    oof_metrics = compute_oof_metrics(cv.oof_y, cv.oof_post, best_thr)
    oof_pred = (cv.oof_post >= best_thr).astype(int)

    if print_progress:
        print(f"Selected OOF threshold ({threshold_metric_name}) = {best_thr:.4f}")
        print(f"OOF threshold score                        = {thr_score:.4f}")
        print_metrics(cv.oof_y, oof_pred, cv.oof_post, title="OOF / Development")

    return DevelopmentResult(
        config=cfg,
        configuration_id=config_id,
        fold_indices=cv.fold_indices,
        threshold=best_thr,
        threshold_metric=threshold_metric_name,
        threshold_score=float(thr_score),
        oof_post=cv.oof_post,
        oof_pred=oof_pred,
        oof_y=cv.oof_y,
        oof_metrics=dict(oof_metrics),
        selection_metric_value=_selection_metric_value(oof_metrics),
    )


def select_best_development_config(
    results: list[DevelopmentResult],
    *,
    output_path: Path | str | None = None,
) -> DevelopmentSelection:
    """Rank development candidates using OOF metrics only."""
    if not results:
        raise ValueError("results must contain at least one development result")

    ranked = sorted(
        results,
        key=lambda result: (
            -float(result.oof_metrics["bal_acc"]),
            -float(result.oof_metrics["macro_f1"]),
            -float(result.oof_metrics["auc"]),
            _model_complexity(result.config),
            result.configuration_id,
        ),
    )
    selected = ranked[0]
    selection = DevelopmentSelection(
        selected_result=selected,
        ranked_results=ranked,
        ranking_metric="oof_balanced_accuracy",
        tie_break_rules=[
            "oof_macro_f1",
            "oof_auc",
            "lower_model_complexity",
            "configuration_id_lexicographic",
        ],
        seed=int(selected.config.seed),
        k_folds=int(selected.config.k_folds),
        grouping_protocol="GroupKFold grouped by training subject IDs only",
        timestamp_utc=_utc_timestamp(),
    )

    if output_path is not None:
        payload = {
            "protocol_version": "publication_protocol_v2",
            "selected_using": "training_oof_only",
            "selection_metric": selection.ranking_metric,
            "tie_break_rules": selection.tie_break_rules,
            "selected_configuration_id": selected.configuration_id,
            "selected_threshold": selected.threshold,
            "seed": selection.seed,
            "k_folds": selection.k_folds,
            "grouping_protocol": selection.grouping_protocol,
            "timestamp_utc": selection.timestamp_utc,
            "ranked_results": [
                {
                    "configuration_id": result.configuration_id,
                    "oof_metrics": result.oof_metrics,
                    "selected_threshold": result.threshold,
                    "model_complexity": {
                        "n_rules": int(result.config.n_rules),
                        "n_sets": int(result.config.n_sets),
                        "n_bins": int(result.config.n_bins),
                    },
                }
                for result in ranked
            ],
        }
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    return selection


def lock_development_config(
    development_result: DevelopmentResult,
    selection: DevelopmentSelection,
    *,
    output_path: Path | str,
    extra_metadata: dict | None = None,
) -> dict:
    """Persist a versioned locked configuration selected from OOF only."""
    if development_result.configuration_id != selection.selected_result.configuration_id:
        raise ValueError("development_result must be the selected development result")

    config_payload = development_result.config.to_dict()
    lock_payload = {
        "protocol_version": "publication_protocol_v2",
        "schema_version": 1,
        "configuration_id": development_result.configuration_id,
        "selected_using": "training_oof_only",
        "selection_metric": selection.ranking_metric,
        "tie_break_rules": selection.tie_break_rules,
        "threshold_source": "training_oof_only",
        "threshold": development_result.threshold,
        "seed": int(development_result.config.seed),
        "k_folds": int(development_result.config.k_folds),
        "grouping_protocol": selection.grouping_protocol,
        "timestamp_utc": selection.timestamp_utc,
        "config": config_payload,
        "oof_metrics": dict(development_result.oof_metrics),
        "selection_metric_value": development_result.selection_metric_value,
        "final_test_evaluated": False,
    }
    if extra_metadata:
        lock_payload.update(extra_metadata)
    lock_payload["config_hash"] = _json_hash(config_payload)
    lock_payload["lock_hash"] = _json_hash(
        {k: v for k, v in lock_payload.items() if k != "lock_hash"}
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = output_file.with_name(f".{output_file.name}.tmp")
    tmp_file.write_text(
        json.dumps(lock_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp_file, output_file)
    return lock_payload


def _load_and_validate_lock(locked_config_path: Path | str) -> tuple[Path, dict]:
    lock_path = Path(locked_config_path)
    if not lock_path.exists():
        raise FileNotFoundError(f"Locked configuration not found: {lock_path}")

    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    required = {
        "protocol_version",
        "schema_version",
        "configuration_id",
        "selected_using",
        "selection_metric",
        "tie_break_rules",
        "threshold_source",
        "threshold",
        "seed",
        "k_folds",
        "grouping_protocol",
        "config",
        "oof_metrics",
        "final_test_evaluated",
        "config_hash",
        "lock_hash",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Locked configuration is missing required fields: {missing}")
    if payload["protocol_version"] != "publication_protocol_v2":
        raise ValueError("Locked configuration protocol_version is not supported")
    if payload["selected_using"] != "training_oof_only":
        raise ValueError("Locked configuration was not selected using training OOF only")
    if payload["threshold_source"] != "training_oof_only":
        raise ValueError("Locked configuration threshold source must be training OOF only")
    if payload["final_test_evaluated"] not in (False, 0):
        raise ValueError("Locked configuration already records final test evaluation")
    expected_config_hash = _json_hash(payload["config"])
    if payload["config_hash"] != expected_config_hash:
        raise ValueError("Locked configuration config_hash is invalid")
    expected_lock_hash = _json_hash(
        {k: v for k, v in payload.items() if k != "lock_hash"}
    )
    if payload["lock_hash"] != expected_lock_hash:
        raise ValueError("Locked configuration lock_hash is invalid")
    return lock_path, payload


def run_final_evaluation_from_lock(
    *,
    locked_config_path: Path | str,
    dataset: DatasetSplit,
    output_dir: Path | str | None = None,
    confirm_final_evaluation: bool = False,
    final_csv_path: Path | str | None = None,
) -> FinalEvaluationResult:
    """Run held-out evaluation only from a previously locked OOF-selected config."""
    if not confirm_final_evaluation:
        raise ValueError(
            "Final evaluation is guarded. Re-run with confirm_final_evaluation=True"
        )

    lock_path, lock_payload = _load_and_validate_lock(locked_config_path)
    cfg = ExperimentConfig(**lock_payload["config"])
    setup_seed(cfg.seed)

    with _deterministic_thread_context():
        test = retrain_and_test_result(
            dataset.X_train,
            dataset.y_train,
            dataset.X_test,
            dataset.y_test,
            float(lock_payload["threshold"]),
            cfg,
        )

    artifact_path = None
    if output_dir is not None:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        artifact_path = output_dir_path / f"{lock_payload['configuration_id']}_final_test.json"
        artifact_payload = {
            "protocol_version": lock_payload["protocol_version"],
            "configuration_id": lock_payload["configuration_id"],
            "locked_config_path": str(lock_path),
            "threshold": float(lock_payload["threshold"]),
            "test_metrics": dict(test.metrics),
            "final_test_evaluated": True,
            "timestamp_utc": _utc_timestamp(),
        }
        artifact_path.write_text(
            json.dumps(artifact_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    result = FinalEvaluationResult(
        locked_config_path=str(lock_path),
        final_artifact_path=str(artifact_path) if artifact_path is not None else None,
        threshold=float(lock_payload["threshold"]),
        test_pred=test.y_pred,
        test_post=test.posterior,
        test_metrics=dict(test.metrics),
        models=test.models,
    )

    if final_csv_path is not None:
        row = make_final_test_result_row(lock_payload["configuration_id"], result)
        log_final_test_row(str(final_csv_path), row)

    return result


def write_development_results_csv(
    results: list[DevelopmentResult],
    selection: DevelopmentSelection,
    *,
    csv_path: Path | str,
) -> None:
    csv_file = Path(csv_path)
    if csv_file.exists():
        csv_file.unlink()
    for rank, result in enumerate(selection.ranked_results, start=1):
        row = make_development_result_row(
            result,
            selection_rank=rank,
            selected=result.configuration_id == selection.selected_result.configuration_id,
            selection_metric=selection.ranking_metric,
            tie_break_rules=selection.tie_break_rules,
        )
        log_development_row(str(csv_file), row)


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
    with _deterministic_thread_context():
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
    with _deterministic_thread_context():
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
            test=test.metrics,
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
        cv_y=cv.oof_y,
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
        oof_metrics=dict(oof_metrics),
        test_post=test.posterior,
        test_pred=test.y_pred,
        test_metrics=dict(test.metrics),
        models=test.models,
        subject_rows=subject_rows,
        artifact_dir=artifact_dir_str,
        run_name=run_name or exp_name,
        artifact_results=artifact_results,
        pipeline_desc=pipeline_desc,
        preset_name=preset_name,
        variant=exp_name,
    )
