from __future__ import annotations

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
from .console_reporting import (
    dump_anfis_rules,
    dump_bn_cpt,
    dump_fts_transitions,
)
from .csv_reporting import (
    log_to_csv,
    make_result_row,
)
from .results import RunResult


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
        test_metrics=test.metrics,
        models=test.models,
        subject_rows=subject_rows,
        artifact_dir=artifact_dir_str,
        run_name=run_name or exp_name,
        artifact_results=artifact_results,
        pipeline_desc=pipeline_desc,
        preset_name=preset_name,
        variant=exp_name,
    )
