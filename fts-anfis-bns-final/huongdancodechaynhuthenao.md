# Huong dan code chay nhu the nao - thuc nghiem chinh

Tai lieu nay chi mo ta **thuc nghiem chinh** cua project: pipeline full

```text
FTS -> ANFIS -> Bayesian Network
```

Trong code, thuc nghiem chinh tuong ung voi:

```text
preset  = A2_soft
ablation = exp1
```

`A2_soft` la cau hinh chinh dang dung: Gaussian FTS, 3 fuzzy sets, 6 ANFIS rules, Naive BN, va BN soft evidence.

## 1. Chay tu dau?

Entry point chinh la:

```text
main.py
```

Chay bang terminal:

```bash
python main.py --preset A2_soft --ablation exp1
```

Hoac trong notebook:

```python
%run main.py
```

Neu khong truyen CLI args, `main.py` se dung cac hang so mac dinh gan o dau file:

```python
MODE = "single"
PRESET_NAME = "A2_soft"
ABLATION = "exp1"
DATA_FOLDER = "last_30s_segments"
CSV_PATH = "results_main.csv"
SAVE_CSV = True
DUMP_RULES = True
```

## 2. Luong goi ham tong quat

Call-flow ngan gon:

```text
main.py
  -> _parse_args()
  -> run_single(args)
       -> build_config(args)
       -> load_dataset(...)
       -> run_one_experiment(...)
            -> run_cv_result(...)
            -> find_best_threshold(...)
            -> compute_oof_metrics(...)
            -> retrain_and_test_result(...)
            -> per_subject_metrics(...)          optional
            -> dump_anfis_rules(...)             optional
            -> dump_fts_transitions(...)         optional
            -> dump_bn_cpt(...)                  optional
            -> export_bn_dashboard_artifacts(...)
            -> make_result_row(...)
            -> log_to_csv(...)
       -> result.as_dict()
  -> RUN_OUTPUT
```

Sau khi chay xong, `main.py` luu output vao bien global:

```python
RUN_OUTPUT
```

Bien nay huu ich trong notebook vi co the goi tiep:

```python
RUN_OUTPUT["test_metrics"]
RUN_OUTPUT["models"]
explain_test_sample(0)
print_bn_details()
```

## 3. Buoc 1 - Parse args va build config

File:

```text
main.py
```

Ham:

```python
_parse_args()
build_config(args)
```

`_parse_args()` doc CLI args nhu:

- `--preset`
- `--ablation`
- `--fusion`
- `--fts-summary-mode`
- `--dump-rules`
- `--show-per-subject`
- `--print-bn-details`
- `--explain-sample-index`

`build_config(args)` tao `ExperimentConfig` theo thu tu uu tien:

```text
ExperimentConfig default
  -> apply_preset(A2_soft)
  -> OVERRIDES trong main.py
  -> CLI args
```

Voi thuc nghiem chinh, config quan trong la:

```text
mf_type      = gaussian
anfis_mf_type = gaussian
n_sets       = 3
n_rules      = 6
n_bins       = 5
bn_structure = naive
bn_soft      = True
ablation     = exp1
```

Preset `A2_soft` nam o:

```text
src/fbn_pipeline/presets.py
```

Config dataclass nam o:

```text
src/fbn_pipeline/config.py
```

## 4. Buoc 2 - Load data

File:

```text
src/fbn_pipeline/data.py
```

Ham:

```python
load_dataset(project_root / cfg.data_path)
```

Du lieu mac dinh:

```text
last_30s_segments/
  temp.xlsx
  hr.xlsx
  gsr.xlsx
  rr.xlsx
  labels.xlsx
```

Output la `DatasetSplit`:

```python
dataset.X_train
dataset.y_train
dataset.user_train
dataset.X_test
dataset.y_test
dataset.user_test
```

Trong do:

```text
X shape = (N, 4, T)
4 signals = Temp, HR, GSR, RR
label rest = 0
label task = 1
```

Subject split da co san trong `data.py`:

- train subjects dung cho GroupKFold CV
- test subjects giu rieng lam held-out test

## 5. Buoc 3 - Goi runner chinh

File:

```text
src/fbn_pipeline/runner.py
```

Ham:

```python
run_one_experiment(...)
```

Trong `main.py`, ham nay duoc goi nhu sau:

```python
result = run_one_experiment(
    cfg,
    dataset,
    exp_name=exp_name,
    preset_name=preset_name or "",
    run_name=run_name,
    csv_path=(PROJECT_ROOT / csv_path_str) if save_csv else None,
    show_per_subject=show_per_subject,
    dump_rules=cfg.dump_rules,
    export_artifacts=True,
    artifacts_dir=artifacts_dir.parent,
    print_configuration=True,
    print_progress=True,
)
```

`run_one_experiment()` la noi dieu phoi toan bo thuc nghiem:

1. set seed
2. chay cross-validation
3. chon threshold
4. retrain tren full train
5. danh gia held-out test
6. in metrics
7. dump rules/CPT neu bat
8. export artifact cho dashboard
9. ghi CSV
10. tra ve `RunResult`

## 6. Buoc 4 - Cross-validation tren train subjects

File:

```text
src/fbn_pipeline/pipeline.py
```

Ham:

```python
run_cv_result(dataset.X_train, dataset.y_train, dataset.user_train, cfg)
```

Trong ham nay:

```python
gkf = GroupKFold(n_splits=cfg.k_folds)
```

Nghia la CV split theo subject, tranh viec sample cung user roi vao ca train va validation trong cung fold.

Moi fold chay theo flow:

```text
X_train_fold, X_valid_fold
  -> extract_features_for_ablation(...)
  -> fit_signal_models(...)
  -> calibrate_probs(...)
  -> make_bn(...)
  -> bn.fit(...)
  -> bn.predict_proba(...)
  -> ghi vao oof_post[valid_indices]
```

Voi thuc nghiem chinh `ablation = exp1`, nhanh code duoc chay la full pipeline:

```python
F_tr, F_va, _fts = extract_features_for_ablation(X_tr, X_va, cfg)

tr_probs, va_probs, _ = fit_signal_models(F_tr, F_va, y_tr, cfg)
tr_probs, va_probs = calibrate_probs(tr_probs, va_probs, y_tr, cfg.calibrate)

sw = compute_signal_weights(y_tr, tr_probs) if cfg.bn_reliability else None
bn = make_bn(cfg)
bn.fit(tr_probs, y_tr, signal_weights=sw)
oof_post[va_idx] = bn.predict_proba(va_probs)
```

Ket qua cua CV la:

```python
CvResult(
    oof_post=...,
    oof_y=...,
    fold_indices=...,
)
```

`oof_post` la posterior `P(task)` cua tung sample train, nhung duoc du doan khi sample do nam trong validation fold.

## 7. Ben trong Stage 1 - FTS

File:

```text
src/fbn_pipeline/stages.py
src/fbn_pipeline/models/fts_encoder.py
```

Ham/lop:

```python
extract_features_for_ablation(...)
FTSEncoder
```

Voi `exp1`, code tao `FTSEncoder`:

```python
fts = FTSEncoder(
    n_sets=cfg.n_sets,
    mf_type=cfg.mf_type,
    mf_sigma_scale=cfg.mf_sigma_scale,
    gbell_b=cfg.gbell_b,
    norm_mode=cfg.fts_norm,
    robust=cfg.robust_fts,
)
```

Sau do:

```python
F_train = fts.fit_transform(X_train)
F_valid = fts.transform(X_valid)
```

FTS bien raw window `(N, 4, T)` thanh transition features:

```text
Temp transitions
HR transitions
GSR transitions
RR transitions
```

Neu `n_sets = 3`, moi signal co `3 x 3 = 9` transition features.

Tong feature:

```text
4 signals * 9 = 36 features
```

## 8. Ben trong Stage 2 - ANFIS

File:

```text
src/fbn_pipeline/training.py
src/fbn_pipeline/models/anfis_tsk.py
```

Ham/lop:

```python
fit_signal_models(...)
ANFIS_TSK
```

`fit_signal_models()` chia FTS feature thanh 4 phan, moi phan ung voi 1 signal:

```text
Temp features -> ANFIS_TSK #1 -> P(task | Temp)
HR features   -> ANFIS_TSK #2 -> P(task | HR)
GSR features  -> ANFIS_TSK #3 -> P(task | GSR)
RR features   -> ANFIS_TSK #4 -> P(task | RR)
```

Output:

```python
train_probs shape = (N_train, 4)
valid_probs shape = (N_valid, 4)
```

Moi cot la probability cua mot signal:

```text
[p_temp, p_hr, p_gsr, p_rr]
```

Voi preset `A2_soft`, ANFIS dung:

```text
n_rules = 6
anfis_mf_type = gaussian
n_epochs = 50
lr = 0.003
ridge_lambda = 0.01
```

## 9. Ben trong Stage 3 - Bayesian Network

File:

```text
src/fbn_pipeline/fusion.py
src/fbn_pipeline/models/bn_fusion.py
```

Ham/lop:

```python
make_bn(cfg)
BNFusion
```

Voi thuc nghiem chinh:

```text
cfg.bn_structure = naive
cfg.bn_soft = True
```

Nen `make_bn(cfg)` tra ve:

```python
BNFusion(n_bins=cfg.n_bins, alpha=cfg.alpha, soft=cfg.bn_soft)
```

BN nhan 4 ANFIS probabilities lam evidence:

```text
P(task | Temp)
P(task | HR)
P(task | GSR)
P(task | RR)
```

Structure naive:

```text
Task -> E_Temp
Task -> E_HR
Task -> E_GSR
Task -> E_RR
```

BN fit tren train fold:

```python
bn.fit(tr_probs, y_tr, signal_weights=sw)
```

Sau do predict validation fold:

```python
bn.predict_proba(va_probs)
```

Output la posterior cuoi:

```text
P(task | evidence from 4 signals)
```

Vi `bn_soft = True`, BN khong ep moi signal vao 1 bin cung. No dung soft evidence membership tren cac evidence bins.

## 10. Buoc 5 - Chon threshold tu OOF

File:

```text
src/fbn_pipeline/metrics.py
```

Ham:

```python
find_best_threshold(cv.oof_y, cv.oof_post, metric=cfg.threshold_metric)
```

Sau CV, project co:

```text
oof_y    = label that cua train samples
oof_post = P(task) duoc du doan tu validation folds
```

Code sweep threshold tren OOF predictions de chon threshold tot nhat.

Mac dinh:

```text
threshold_metric = balanced_accuracy
threshold_grid = 181
```

Ket qua:

```python
best_thr, thr_score
```

Threshold nay moi duoc dung cho held-out test.

## 11. Buoc 6 - Retrain tren full train va danh gia test

File:

```text
src/fbn_pipeline/pipeline.py
```

Ham:

```python
retrain_and_test_result(
    dataset.X_train,
    dataset.y_train,
    dataset.X_test,
    dataset.y_test,
    best_thr,
    cfg,
)
```

Lan nay code train lai toan bo pipeline tren full train split:

```text
full X_train
  -> fit FTS
  -> fit 4 ANFIS models
  -> fit BN
```

Sau do apply len held-out test:

```text
X_test
  -> FTS transform
  -> ANFIS predict per signal
  -> BN predict posterior
  -> threshold best_thr
  -> y_pred
```

Output:

```python
TestResult(
    y_pred=...,
    posterior=...,
    metrics=...,
    models=...
)
```

`models` gom cac model da train tren full train:

```python
TrainedModels(
    fts=...,
    bn=...,
    anfis_list=[...],
    test_evidence=...
)
```

Khi convert sang legacy dict:

```python
{
    "fts": fts,
    "bn": bn,
    "anfis_list": anfis_list,
    "te_probs": test_evidence,
}
```

## 12. Buoc 7 - Metrics, CSV va artifacts

Trong `runner.py`, sau khi co CV va test result:

### Metrics

File:

```text
src/fbn_pipeline/metrics.py
```

Ham:

```python
compute_oof_metrics(...)
print_metrics(...)
per_subject_metrics(...)
```

Metrics gom:

- accuracy
- balanced accuracy
- precision
- recall
- f1
- macro_f1
- auc

### CSV

File:

```text
src/fbn_pipeline/csv_reporting.py
```

Ham:

```python
make_result_row(...)
log_to_csv(...)
```

Mac dinh ghi vao:

```text
results_main.csv
```

### Artifacts cho dashboard

File:

```text
src/fbn_pipeline/bn_dashboard.py
src/fbn_pipeline/artifacts/bn_serializer.py
src/fbn_pipeline/artifacts/anfis_serializer.py
```

Ham:

```python
export_bn_dashboard_artifacts(...)
```

Mac dinh ghi vao:

```text
artifacts/<run_name>/
  bn_details.json
  anfis_rules.json
```

`bn_details.json` chua:

- config
- threshold
- test metrics
- BN structure
- prior
- signal weights
- CPT tables
- sample inference trace

`anfis_rules.json` chua rule text cua 4 ANFIS models.

## 13. Output cuoi cung nam o dau?

Sau khi `main.py` chay xong:

```python
RUN_OUTPUT = run_single(_args)
```

`RUN_OUTPUT` la dict lay tu:

```python
RunResult.as_dict()
```

Cac key quan trong:

```python
RUN_OUTPUT["config"]
RUN_OUTPUT["dataset"]
RUN_OUTPUT["best_threshold"]
RUN_OUTPUT["oof_metrics"]
RUN_OUTPUT["test_metrics"]
RUN_OUTPUT["models"]
RUN_OUTPUT["test_post"]
RUN_OUTPUT["test_pred"]
RUN_OUTPUT["artifact_results"]
```

Neu can xem model:

```python
models = RUN_OUTPUT["models"]
models["fts"]
models["anfis_list"]
models["bn"]
models["te_probs"]
```

Neu can xem ket qua test:

```python
RUN_OUTPUT["test_metrics"]
RUN_OUTPUT["best_threshold"]
```

Neu can explain sample:

```python
explain_test_sample(0)
explain_sample_of_user("3caqi", 0)
print_bn_details()
```

## 14. Tom tat theo mot dong

Thuc nghiem chinh chay nhu sau:

```text
main.py
  -> build_config(A2_soft, exp1)
  -> load_dataset(last_30s_segments)
  -> run_one_experiment
       -> GroupKFold CV
            -> FTS fit/transform
            -> train 4 ANFIS
            -> fit BN soft evidence
            -> collect OOF P(task)
       -> choose best threshold from OOF
       -> retrain FTS + ANFIS + BN on full train
       -> evaluate held-out test
       -> save metrics CSV and dashboard artifacts
  -> RUN_OUTPUT
```

Neu nguoi moi chi can hieu code chay tu dau den dau, hay doc theo thu tu:

```text
main.py
src/fbn_pipeline/runner.py
src/fbn_pipeline/pipeline.py
src/fbn_pipeline/stages.py
src/fbn_pipeline/training.py
src/fbn_pipeline/fusion.py
src/fbn_pipeline/models/
```
