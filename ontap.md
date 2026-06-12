# On tap: Outputs cua pipeline FTS-ANFIS-BN

Tai lieu nay tom tat cac output duoc sinh ra khi chay pipeline trong
`fts-anfis-bns-final`, bam theo implementation hien tai trong `main.py` va
`src/fbn_pipeline/`.

## 1. Overview of Generated Outputs

Flow chay chinh:

```text
main.py
-> build_config()
-> load_dataset()
-> run_one_experiment()
   -> run_cv_result() / run_cv()
   -> find_best_threshold()
   -> retrain_and_test_result() / retrain_and_test()
   -> metrics/reporting/artifacts
-> RUN_OUTPUT
```

Entry point nam o `main.py`. Khi chay script truc tiep hoac `%run main.py`,
code goi:

```python
RUN_OUTPUT = run_single(_args)
```

`run_single()` tao config, load dataset, roi goi `run_one_experiment()` trong
`src/fbn_pipeline/runner.py`.

`build_config()` tao `ExperimentConfig` theo thu tu uu tien:

```text
default ExperimentConfig
-> apply_preset(cfg, PRESET_NAME hoac --preset)
-> OVERRIDES
-> script constants / CLI args
```

Preset chinh `A2_soft` trong `src/fbn_pipeline/presets.py` set cac tham so:

```text
n_sets=3
mf_type="gaussian"
mf_sigma_scale=0.4
fts_norm="row"
n_rules=6
sigma_scale=1.3
ridge_lambda=0.01
n_epochs=50
lr=0.003
n_bins=5
alpha=0.5
k_folds=5
bn_soft=True
```

Voi ablation `exp1`, pipeline la:

```text
FTS -> ANFIS -> Bayesian Network
```

## 2. Data Loading Outputs

`load_dataset()` nam trong `src/fbn_pipeline/data.py`. Ham doc:

```text
labels.xlsx
temp.xlsx
hr.xlsx
gsr.xlsx
rr.xlsx
```

va tra ve `DatasetSplit`:

| Output | Shape hien tai | Kieu | Y nghia | Dung o stage nao |
|---|---:|---|---|---|
| `dataset.X_train` | `(632, 4, 30)` | `float64 ndarray` | cua so tin hieu train | CV, retrain |
| `dataset.y_train` | `(632,)` | `int64 ndarray` | nhan train `{0,1}` | ANFIS, BN, threshold |
| `dataset.user_train` | `(632,)` | string ndarray | subject id train | `GroupKFold` |
| `dataset.X_test` | `(193, 4, 30)` | `float64 ndarray` | cua so tin hieu held-out test | final test |
| `dataset.y_test` | `(193,)` | `int64 ndarray` | nhan test `{0,1}` | tinh test metrics |
| `dataset.user_test` | `(193,)` | string ndarray | subject id test | per-subject report |

Dinh dang tong quat:

```text
X shape = (N, 4, T)
4 signals = Temp, HR, GSR, RR
T = 30 trong data hien tai
```

Thu tu signal trong code:

```python
("temp", "hr", "gsr", "rr")
```

Tuong ung ten bao cao:

```text
Temp, HR, GSR, RR
```

Label encode trong `_binarize_labels()`:

```text
"rest" -> -1 -> +1 => 0
cac task level khac -> +1, neu > 0 thi ep thanh 1
```

Do do:

```text
REST = 0
TASK = 1
```

`user_train` duoc dua vao `GroupKFold.split(X_train, y_train, user_train)` de
dam bao cac sample cung subject khong nam dong thoi trong train fold va
validation fold.

## 3. FTS Outputs

Stage 1 nam trong `FTSEncoder` tai
`src/fbn_pipeline/models/fts_encoder.py`, duoc goi qua
`extract_features_for_ablation()` trong `src/fbn_pipeline/stages.py`.

Input:

```text
X_train hoac X_valid/X_test: (N, 4, T)
```

Voi `exp1`, code tao:

```python
fts = FTSEncoder(
    n_sets=cfg.n_sets,
    mf_type=cfg.mf_type,
    mf_sigma_scale=cfg.mf_sigma_scale,
    gbell_b=cfg.gbell_b,
    norm_mode=cfg.fts_norm,
    robust=cfg.robust_fts,
)
F_train = fts.fit_transform(X_train)
F_valid = fts.transform(X_valid)
```

`fts.fit()` hoc tu training data:

```text
medians_      : median tung signal de impute NaN
n_signals     : so signal, o day la 4
grid_ hoac mf_params_:
  - A2_soft dung mf_type="gaussian"
  - gaussian dung grid centers va spacing theo min/max mo rong 5%
```

Voi `mf_type="gaussian"`:

```text
centers = linspace(vmin_extended, vmax_extended, n_sets)
sigma = spacing * mf_sigma_scale
```

Fuzzification tra ve:

```text
_memberships(values, signal_idx) -> mu shape (N, T, n_sets)
```

Tai moi thoi diem, membership duoc normalize de tong bang 1.

Transition matrix moi signal:

```python
mu_prev = mu[:, :-1, :]
mu_next = mu[:, 1:, :]
trans = np.einsum("nti,ntj->nij", mu_prev, mu_next)
```

Cong thuc:

```text
T_ij = sum_t mu_i(x_t) * mu_j(x_{t+1})
```

Output FTS:

```text
Raw window X: (N, 4, T)
-> Fuzzification: membership degrees (N, T, n_sets)
-> Transition matrix per signal: (N, n_sets, n_sets)
-> Flatten per signal: (N, n_sets^2)
-> Concatenate 4 signals: (N, 4 * n_sets^2)
```

Voi `n_sets=3`:

```text
moi signal: 3 * 3 = 9 features
4 signals: 4 * 9 = 36 features
F_train/F_valid/F_test: (N, 36)
```

Moi feature la mot fuzzy transition dang:

```text
P(to_state | from_state)
```

Khac biet `fts_norm`:

| Mode | Code | Y nghia |
|---|---|---|
| `row` | `trans / row_sum` | moi hang `from_state` cong lai bang 1, gan voi `P(to_state | from_state)` |
| `total_sum` | `trans / total_sum` | toan bo ma tran cong lai bang 1, bieu dien phan phoi chuyen tiep tong the |

`A2_soft` dung `fts_norm="row"`.

## 4. ANFIS Outputs

Stage 2 duoc implement boi `fit_signal_models()` trong
`src/fbn_pipeline/training.py`. Trong request co goi `_fit_signal_models()`,
nhung trong code hien tai ten ham la `fit_signal_models()`.

Trong CV:

```python
tr_probs, va_probs, anfis_models = fit_signal_models(F_tr, F_va, y_tr, cfg)
```

Trong retrain:

```python
tr_probs, te_probs, anfis_models = fit_signal_models(F_tr, F_te, y_train, cfg)
```

FTS output duoc chia theo signal:

```python
feat_per_sig = F_train.shape[1] // 4
signal s dung cot:
s * feat_per_sig : (s + 1) * feat_per_sig
```

Voi `A2_soft`, `F.shape[1]=36`, nen:

```text
feat_per_sig = 9
```

Mapping:

```text
Temp FTS features -> ANFIS_Temp -> p_temp
HR FTS features   -> ANFIS_HR   -> p_hr
GSR FTS features  -> ANFIS_GSR  -> p_gsr
RR FTS features   -> ANFIS_RR   -> p_rr
```

Co 4 signal nen co 4 model `ANFIS_TSK`. Moi ANFIS nhan:

```text
X_signal: (N, 9)
y_train : (N,)
```

Output:

```text
train_probs: (N_train_fold, 4)
valid_probs: (N_valid_fold, 4)
test_probs / te_probs: (N_test, 4)
```

Cot trong `probs`:

```text
column 0 = p_temp
column 1 = p_hr
column 2 = p_gsr
column 3 = p_rr
```

`predict_raw()` tra output Layer 5 truoc sigmoid. `predict_proba()` tra:

```text
sigmoid(raw) = p_task trong [0,1]
```

Output Stage 2:

```text
probs = [p_temp, p_hr, p_gsr, p_rr]
shape = (N, 4)
```

## 5. BN/CPT Outputs

Stage 3 nam trong `BNFusion` tai
`src/fbn_pipeline/models/bn_fusion.py`, duoc tao boi `make_bn(cfg)` trong
`src/fbn_pipeline/fusion.py`.

Voi `A2_soft`:

```text
make_bn() -> BNFusion(n_bins=5, alpha=0.5, soft=True)
```

BN nhan input tu ANFIS:

```python
bn.fit(tr_probs, y_train)
bn.predict_proba(va_probs hoac te_probs)
```

`probs` khong phai raw HR/RR/GSR/Temp. No la:

```text
probs = output ANFIS per signal
shape = (N, 4)
```

CPT khong hoc tu raw physiological signals truc tiep. CPT hoc tu output ANFIS
tren tap train cua fold hoac full training set.

BN hoc:

| Thuoc tinh | Y nghia |
|---|---|
| `bin_edges_` | quantile edges cho tung signal, fit tren `tr_probs` |
| `prior_` | `P(Y=rest)` va `P(Y=task)` co Laplace smoothing |
| `cpts_` | list CPT, moi signal co matrix `(2, n_bins)` |
| `bin_mids_` | midpoint cua bins, chi co khi `soft=True` |
| `bin_sigmas_` | sigma Gaussian soft membership, chi co khi `soft=True` |
| `signal_weights_` | trong so reliability neu bat `bn_reliability`; `A2_soft` mac dinh la 1 |

CPT:

```text
P(E_s = b | Y = c)
=
(count(E_s = b and Y = c) + alpha)
/
(count(Y = c) + alpha * n_bins)
```

Soft evidence trong `A2_soft`:

```text
P_eff(E_s | Y=c)
=
sum_b mu_s,b(p_s) * P(E_s=b | Y=c)
```

Sau do:

```text
P(Y=c | evidence)
proportional to
P(Y=c) * product_s P_eff(E_s | Y=c)
```

Output `bn.predict_proba()`:

```text
posterior = P(TASK | evidence)
shape = (N,)
```

`get_cpt_table()` tra CPT de hien thi. `explain_sample()` tra trace inference
cho mot sample: ANFIS probs, soft memberships, effective likelihoods,
log-likelihoods, posterior task/rest.

## 6. Cross-Validation Outputs

`run_cv_result()` nam trong `src/fbn_pipeline/pipeline.py`. Wrapper cu
`run_cv()` tra:

```python
oof_post, oof_y, fold_indices
```

Voi data hien tai:

```text
oof_post: (632,)
oof_y:    (632,)
fold_indices: list 5 phan tu, moi phan tu la (tr_idx, va_idx)
```

| Output | Tao tu dau | Dung de lam gi | Co phai model cuoi? | Co dung test? |
|---|---|---|---|---|
| `oof_post` | posterior tren validation fold cua tung fold | chon threshold, OOF metrics | khong | khong |
| `oof_y` | copy cua `y_train` | ground truth cho threshold/OOF metrics | khong | khong |
| `fold_indices` | `GroupKFold.split(...)` | audit split train/val theo subject | khong | khong |

Moi fold:

```text
1. Fit FTS tren X_tr
2. Transform X_tr va X_va
3. Train 4 ANFIS tren F_tr, y_tr
4. Predict ANFIS probs cho train/val
5. Fit BN/CPT tren tr_probs, y_tr
6. Predict va_probs -> posterior validation
7. Ghi vao oof_post[va_idx]
```

Moi fold train FTS, ANFIS, BN rieng. Moi fold co CPT rieng. CPT cua fold chi
dung de predict validation fold do.

`oof_post` la posterior cua toan bo training samples, nhung moi sample duoc du
doan khi no nam trong validation fold. `oof_post` dung de chon threshold, khong
phai output cua final model.

## 7. Threshold Selection Outputs va cach hoc nguong

`find_best_threshold()` nam trong `src/fbn_pipeline/metrics.py`. Trong
`runner.py`, ham duoc goi sau CV:

```python
best_thr, thr_score = find_best_threshold(
    cv.oof_y,
    cv.oof_post,
    metric=threshold_metric or cfg.threshold_metric,
    n_grid=cfg.threshold_grid,
)
```

Input:

```text
oof_y    = nhan that cua train samples
oof_post = OOF posterior P(TASK | evidence) cua train samples
```

Voi default config:

```text
metric = "balanced_accuracy"
threshold_grid = 181
low = 0.05
high = 0.95
```

Threshold duoc sweep:

```python
thresholds = np.linspace(0.05, 0.95, 181)
```

Moi threshold `t` duoc danh gia bang:

```text
oof_pred_t = (oof_post >= t).astype(int)
score_t = balanced_accuracy(oof_y, oof_pred_t)
```

Code giu threshold co score cao nhat:

```text
best_thr = argmax_t score_t
thr_score = max_t score_t
```

Noi cach khac, pipeline "hoc nguong" bang validation noi bo tren training set:

```text
Training subjects
-> GroupKFold OOF posterior
-> sweep threshold tren OOF
-> chon threshold toi uu theo balanced_accuracy
-> co dinh threshold nay
-> dung lai threshold do cho held-out test
```

Nguong khong duoc hoc bang gradient descent va khong la tham so cua ANFIS hay
BN. No la hyperparameter quyet dinh class label, duoc chon sau khi da co
`oof_post`.

Quan trong:

```text
best_thr duoc chon tu OOF prediction tren train.
best_thr khong duoc chon tu test.
test_post chi duoc so sanh voi best_thr sau khi nguong da co dinh.
```

Neu co nhieu threshold cung dat score cao nhat, implementation chon threshold
dau tien gap trong grid vi chi update khi `score > best_score`, khong update
khi bang nhau.

Output:

```text
best_thr   : threshold tot nhat tren OOF
thr_score  : score tuong ung
oof_pred   : (oof_post >= best_thr).astype(int)
oof_metrics: metrics tren OOF voi best_thr
```

## 8. Retrain Outputs

`retrain_and_test_result()` nam trong `src/fbn_pipeline/pipeline.py`.

Can retrain sau CV vi model trong CV chi hoc tren tung fold con. Final model
phai hoc tren toan bo training subjects truoc khi danh gia held-out test.

Retrain khong lay trung binh model tu cac fold:

```text
CV:
Fold 1 -> CPT_1
Fold 2 -> CPT_2
...
Fold 5 -> CPT_5

Retrain:
Full training set -> CPT_final
```

Trong retrain `exp1`:

```python
F_tr, F_te, fts = extract_features_for_ablation(X_train, X_test, cfg)
tr_probs, te_probs, anfis_models = fit_signal_models(F_tr, F_te, y_train, cfg)
bn = make_bn(cfg)
bn.fit(tr_probs, y_train)
te_post = bn.predict_proba(te_probs)
```

Ket luan:

```text
Retrain co hoc lai FTS: co, fit tren toan bo X_train.
Retrain co train lai 4 ANFIS: co, train tren toan bo F_tr.
Retrain co fit lai BN/CPT: co, fit tren tr_probs cua toan bo train.
CPT_final hoc tu output ANFIS tren toan bo training set.
CPT_final khong phai CPT cua mot fold.
CPT_final khong phai trung binh CPT_1...CPT_5.
CPT_final khong dung test de hoc.
```

Test chi duoc dung de transform/predict:

```text
X_test -> fts.transform()
F_te -> ANFIS predict_proba()
te_probs -> BN predict_proba()
```

## 9. Test Evaluation Outputs

Trong `retrain_and_test_result()`:

```python
y_pred = (te_post >= threshold).astype(int)
metrics = compute_oof_metrics(y_test, te_post, threshold)
```

| Output | Tao tu dau | Shape | Y nghia | Lien quan threshold |
|---|---|---:|---|---|
| `te_probs` | 4 ANFIS final tren `F_te` | `(193, 4)` | per-signal `p_task` tren test | chua threshold |
| `te_post` / `test_post` | BN final predict tren `te_probs` | `(193,)` | `P(TASK | evidence)` | dung de so voi threshold |
| `y_pred` / `test_pred` | `te_post >= best_thr` | `(193,)` | nhan cuoi `{0,1}` | truc tiep dung threshold |
| `metrics` / `test_metrics` | `compute_metrics(y_test, te_post, best_thr)` | dict | accuracy, precision, recall, F1, AUC, bal_acc, macro_f1 | metrics tru AUC phu thuoc threshold |
| `models` | `TrainedModels(...)` | object/dict | final FTS, BN, ANFIS list, test evidence | chua model sau retrain |

Cu the:

```text
te_probs = output cua 4 ANFIS tren test
te_post = posterior P(TASK | evidence) sau BN fusion
y_pred = nhan cuoi sau khi so sanh te_post voi best_thr
metrics = accuracy, balanced accuracy, F1, AUC,...
models = model cuoi cung sau retrain
```

## 10. RUN_OUTPUT Dictionary

`RUN_OUTPUT` la `result.as_dict()` tu `RunResult` trong
`src/fbn_pipeline/results.py`.

| Key | Y nghia |
|---|---|
| `RUN_OUTPUT["config"]` | object `ExperimentConfig` sau preset/override |
| `RUN_OUTPUT["dataset"]` | `DatasetSplit` chua train/test arrays |
| `RUN_OUTPUT["fold_indices"]` | 5 GroupKFold `(tr_idx, va_idx)` |
| `RUN_OUTPUT["best_threshold"]` | threshold chon tu OOF |
| `RUN_OUTPUT["threshold_score"]` | score tot nhat khi sweep threshold |
| `RUN_OUTPUT["oof_post"]` | OOF posterior tren train samples |
| `RUN_OUTPUT["oof_pred"]` | OOF label sau threshold |
| `RUN_OUTPUT["oof_metrics"]` | metrics tren OOF |
| `RUN_OUTPUT["test_post"]` | final posterior tren held-out test |
| `RUN_OUTPUT["test_pred"]` | final test labels |
| `RUN_OUTPUT["test_metrics"]` | final held-out test metrics |
| `RUN_OUTPUT["subject_rows"]` | per-subject test metrics neu bat |
| `RUN_OUTPUT["models"]` | dict model cuoi sau retrain |
| `RUN_OUTPUT["artifact_dir"]` | thu muc artifact neu export |
| `RUN_OUTPUT["run_name"]` | vi du `A2_soft_exp1` |
| `RUN_OUTPUT["artifact_results"]` | ket qua export dashboard artifacts |
| `RUN_OUTPUT["pipeline_desc"]` | mo ta pipeline neu truyen vao |
| `RUN_OUTPUT["preset_name"]` | preset name |
| `RUN_OUTPUT["variant"]` | experiment variant/name |

Ben trong `RUN_OUTPUT["models"]`:

| Key | Y nghia |
|---|---|
| `models["fts"]` | `FTSEncoder` final fit tren full train |
| `models["bn"]` | `BNFusion` final fit tren `tr_probs` full train |
| `models["anfis_list"]` | list 4 `ANFIS_TSK`, theo Temp/HR/GSR/RR |
| `models["te_probs"]` | test evidence, ANFIS probabilities tren test, shape `(N_test, 4)` |

## 11. Temporary vs Final Outputs

| Output | Sinh ra o dau | Dung de lam gi | Tam thoi hay cuoi cung |
|---|---|---|---|
| FTS features trong tung fold | `extract_features_for_ablation()` trong `run_cv_result()` | train/predict fold-local ANFIS | tam thoi |
| ANFIS probs trong tung fold | `fit_signal_models()` trong CV | evidence de fit fold BN va predict val | tam thoi |
| CPT trong tung fold | `bn.fit(tr_probs, y_tr)` trong CV | predict validation fold | tam thoi |
| OOF posterior | `oof_post[va_idx] = bn.predict_proba(va_probs)` | chon threshold, OOF metrics | trung gian |
| best threshold | `find_best_threshold(oof_y, oof_post)` | quyet dinh nhan OOF/test | final threshold |
| FTS sau retrain | `extract_features_for_ablation(X_train, X_test)` | transform final test | final model |
| ANFIS sau retrain | `fit_signal_models(F_tr, F_te, y_train)` | tao `te_probs` | final model |
| CPT_final sau retrain | `bn.fit(tr_probs, y_train)` | BN inference tren test | final model |
| `te_probs` | final ANFIS predict tren test | evidence cho BN, explain sample | final test evidence |
| test posterior | `bn.predict_proba(te_probs)` | `P(TASK|evidence)` test | final output |
| test prediction | `te_post >= best_thr` | nhan cuoi | final output |
| test metrics | `compute_oof_metrics(y_test, te_post, best_thr)` | bao cao hieu nang | final report |

## 12. Data Leakage Check

| Thanh phan | Fit tren train hay test? | Co leakage khong? | Giai thich |
|---|---|---|---|
| FTS membership parameters | train fold trong CV; full train trong retrain | Khong | `fit_transform()` chi tren train; validation/test dung `transform()` |
| ANFIS parameters | train fold trong CV; full train trong retrain | Khong | `model.fit(F_train_signal, y_train)` khong dung validation/test label |
| BN bin edges | `tr_probs` cua train fold hoac full train | Khong | `np.quantile(probs[:,s])` trong `bn.fit()` chi nhan train evidence |
| BN prior | `y_tr` hoac `y_train` | Khong | prior tinh tu label train |
| BN CPT | `tr_probs`, `y_tr` hoac full train | Khong | CPT count tu discretized ANFIS outputs cua train |
| soft evidence midpoints/sigmas | train `probs` trong `bn.fit()` | Khong | `_compute_soft_params(probs)` dung train evidence |
| threshold | OOF train predictions | Khong | `find_best_threshold()` nhan `cv.oof_y`, `cv.oof_post`, khong nhan test |
| final test metrics | test labels/probs | Khong de hoc | test dung sau khi model va threshold da co dinh |

Ket luan: voi `exp1` + `A2_soft`, test set chi di qua transform/predict va chi
dung de tinh report cuoi.

## 13. Thesis-Ready Explanation

Trong giai doan cross-validation, toan bo tap training subjects duoc chia bang
`GroupKFold` theo `user_train`, bao dam cac cua so tin hieu cua cung mot
subject khong xuat hien dong thoi trong train fold va validation fold. O moi
fold, pipeline FTS-ANFIS-BN duoc fit doc lap: `FTSEncoder` hoc membership
parameters tu train fold, bon mo hinh `ANFIS_TSK` hoc xac suat `p_task` rieng
cho Temp, HR, GSR va RR, sau do `BNFusion` hoc prior va CPT tu output ANFIS cua
train fold. Posterior tren validation fold duoc luu vao `oof_post`, nen moi
training sample chi co prediction khi no dong vai tro validation.

Nguong phan lop duoc chon sau cross-validation bang cach sweep cac gia tri
threshold tren `oof_post`. Voi moi threshold, code tao nhan tam thoi
`oof_pred = (oof_post >= threshold)` va tinh metric duoc cau hinh, mac dinh la
balanced accuracy. Threshold co score cao nhat duoc luu thanh `best_threshold`.
Nguong nay duoc hoc hoan toan tu OOF prediction cua training set, khong su dung
held-out test set.

Sau cross-validation, cac model fold-local khong duoc lay trung binh hay tai su
dung lam model cuoi. Implementation trong `retrain_and_test_result()` fit lai
toan bo pipeline tren full training set. Cu the, FTS duoc fit lai tren toan bo
`X_train`, bon ANFIS duoc train lai tren toan bo FTS features cua training set,
va Bayesian Network duoc fit lai tren `tr_probs`, tuc output ANFIS cua toan bo
training samples. Vi vay model cuoi phan anh toan bo training data nhung van
chua dung bat ky label hoac statistic nao tu held-out test subjects.

CPT cuoi cung cua Bayesian Network duoc uoc luong tu evidence da qua ANFIS,
khong phai tu raw Temp, HR, GSR, RR. Voi moi signal, output ANFIS lien tuc duoc
discretize bang quantile bin edges hoc tren training evidence. CPT uoc luong
`P(E_s=b|Y=c)` bang count co Laplace smoothing. Trong preset `A2_soft`,
inference khong dung mot bin cung duy nhat; moi xac suat ANFIS duoc anh xa
thanh soft membership tren cac bin, roi likelihood hieu dung duoc tinh bang tong
co trong so giua membership va CPT. Posterior cuoi la `P(TASK|evidence)` sau
khi ket hop prior va likelihood cua bon signal.

Held-out test evaluation chi duoc thuc hien sau khi threshold va model cuoi da
co dinh. Test windows duoc transform bang FTS final, dua qua bon ANFIS final de
tao `te_probs`, roi dua vao BN final de tao `test_post`. Nhan du doan cuoi cung
la `test_pred = test_post >= best_threshold`. Cac chi so nhu accuracy,
precision, recall, F1, balanced accuracy, macro-F1 va AUC duoc tinh tu
`y_test`, `test_post` va threshold da chon tu OOF. Do do test set dong vai tro
danh gia doc lap, khong tham gia hoc FTS, ANFIS, BN/CPT hoac threshold.
