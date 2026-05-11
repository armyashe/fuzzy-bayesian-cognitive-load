# FBN Pipeline Project Structure

## 1. Tổng quan

Project này triển khai pipeline **FTS → ANFIS → Bayesian Network** cho phân loại tải nhận thức (cognitive load) từ tín hiệu sinh lý.

Mục tiêu của cấu trúc:

- tách **model code** ra khỏi **logic chạy thí nghiệm**
- hỗ trợ 4 ablation experiments để đánh giá đóng góp từng stage
- dễ import lại trong notebook bằng `%run main.py`
- có Streamlit dashboard để trực quan hóa kết quả

---

## 2. Cấu trúc thư mục thực tế

```text
project/
├─ main.py
├─ app.py                  ← Streamlit dashboard
├─ README.md
├─ requirements.txt
├─ results_main.csv        ← CSV kết quả (tự sinh khi chạy)
├─ RUNSTREAMLIT.MD         ← hướng dẫn chạy dashboard
├─ artifacts/              ← JSON artifacts per run (tự sinh khi chạy)
│  └─ <preset>_<ablation>/
│     ├─ anfis_rules.json
│     └─ bn_details.json
├─ last_30s_segments/
│  ├─ gsr.xlsx
│  ├─ hr.xlsx
│  ├─ labels.xlsx
│  ├─ rr.xlsx
│  └─ temp.xlsx
├─ notebooks/
│  ├─ evaluate_ablation_experiments.ipynb
│  ├─ main.ipynb
│  └─ output.ipynb
└─ src/
   └─ fbn_pipeline/
      ├─ __init__.py
      ├─ bn_dashboard.py
      ├─ config.py
      ├─ data.py
      ├─ features.py
      ├─ metrics.py
      ├─ pipeline.py
      ├─ reporting.py
      └─ models/
         ├─ __init__.py
         ├─ anfis_tsk.py
         ├─ bn_fusion.py
         └─ fts_encoder.py
```

---

## 3. Mô tả từng file

### `main.py`

File chạy chính của project.

Nhiệm vụ:

- thêm `src/` vào `sys.path`
- import các module trong package `fbn_pipeline`
- parse CLI arguments (`--ablation`, `--preset`, `--fusion`, `--fts-summary-mode`, v.v.)
- chọn preset/config để chạy
- gọi hàm load dữ liệu
- gọi pipeline `run_cv()` và `retrain_and_test()`
- in kết quả ra màn hình
- ghi CSV và JSON artifacts nếu cần

Hằng số có thể chỉnh trực tiếp trong file (dùng khi chạy notebook):

```python
MODE            = "single"
PRESET_NAME     = "A2_soft"
ABLATION        = "exp1"
DATA_FOLDER     = "last_30s_segments"
CSV_PATH        = "results_main.csv"
SAVE_CSV        = True
SHOW_PER_SUBJECT = True
DUMP_RULES      = True
FUSION_MODE     = "mean"
FTS_SUMMARY_MODE = "diag_high"
```

Notebook có thể chạy bằng:

```python
%run main.py
```

---

### `app.py`

Streamlit dashboard để chạy và trực quan hóa pipeline trực tiếp trên trình duyệt.

Cho phép chọn preset và ablation experiment qua giao diện, xem kết quả metrics, BN CPT, ANFIS rules và per-subject breakdown.

Chạy bằng:

```bash
streamlit run app.py
```

Xem `RUNSTREAMLIT.MD` để biết hướng dẫn cài đặt môi trường chi tiết.

---

### `src/fbn_pipeline/__init__.py`

File đánh dấu `fbn_pipeline` là một Python package.

Export các thành phần quan trọng ở mức package:

- `ExperimentConfig`
- `apply_preset()`
- `setup_seed()`

---

### `src/fbn_pipeline/config.py`

Chứa toàn bộ phần cấu hình thí nghiệm.

Nội dung:

- `ExperimentConfig` — dataclass chứa toàn bộ hyperparameter
- `EXPERIMENT_PRESETS` — dict các preset đã định nghĩa
- `apply_preset()` — áp dụng preset vào config
- `setup_seed()` — set numpy và PYTHONHASHSEED

Các nhóm tham số trong `ExperimentConfig`:

- **FTS**: `n_sets`, `mf_type` (`triangular`/`gaussian`/`trapezoid`/`gbell`), `mf_sigma_scale`, `gbell_b`, `fts_norm` (`total_sum`/`row`)
- **ANFIS**: `n_rules`, `sigma_scale`, `ridge_lambda`, `n_epochs`, `lr`
- **BN**: `n_bins`, `alpha`, `bn_soft`, `bn_reliability`
- **Calibration**: `calibrate` (`none`/`platt`/`isotonic`), `scale_fts`, `robust_fts`
- **Evaluation**: `k_folds`, `threshold_metric`, `rank_metric`, `threshold_grid`
- **Ablation**: `ablation`, `fusion_mode_when_no_bn`, `no_stage1_feature_mode`, `fts_summary_mode`

Các preset có sẵn:

| Preset      | Điểm nổi bật                                            |
| ----------- | ----------------------------------------------------------- |
| `A1`      | triangular MF, 3 sets, BN hard                              |
| `A2`      | gaussian MF, 3 sets, BN hard                                |
| `A3`      | trapezoid MF, 5 sets                                        |
| `A4`      | gbell MF, 3 sets                                            |
| `A5`      | gaussian MF, 4 sets                                         |
| `A2_soft` | A2 + BN soft evidence                                       |
| `A2_plus` | A2 + soft + reliability + isotonic calibration + robust FTS |
| `A2_cal`  | A2 + isotonic calibration                                   |
| `A2_new`  | A2 với sigma_scale khác, 3 bins, alpha=2                  |

---

### `src/fbn_pipeline/data.py`

Chứa logic đọc dữ liệu từ thư mục dữ liệu gốc.

Nhiệm vụ:

- đọc các file Excel trong `last_30s_segments/`
- xếp tín hiệu theo thứ tự: `Temp`, `HR`, `GSR`, `RR`
- binarize nhãn: `rest` → 0, các task → 1
- tách train/test theo danh sách subject cố định

Hằng số quan trọng:

- `TRAIN_IDS`: 18 subject dùng để train + GroupKFold CV
- `TEST_IDS`: 5 subject held-out test

Đầu ra:

- `DatasetSplit` dataclass gồm `X_train`, `y_train`, `user_train`, `X_test`, `y_test`, `user_test`
- `X` có shape `(N, 4, T)` — N mẫu, 4 tín hiệu, T time steps

---

### `src/fbn_pipeline/features.py`

Chứa các hàm trích xuất đặc trưng thay thế dùng trong ablation experiments.

Nhiệm vụ:

- `extract_stat_features(X)` — trích 8 đặc trưng thống kê per signal từ raw windows `(N,4,T)` → `(N, 32)`, dùng trong **exp4** thay FTSEncoder
- `summarise_fts_per_signal(F, n_sets, mode)` — rút gọn FTS features `(N, 4*n_sets²)` → `(N, 4)`, dùng trong **exp3** để bỏ qua ANFIS
- `split_features_per_signal(F, n)` — tách feature matrix thành list per-signal
- `feature_names_stat(n_signals)` — tên cột tương ứng

Các đặc trưng thống kê (`STAT_FEATURE_NAMES`): `mean`, `std`, `min`, `max`, `range`, `slope`, `first_diff_mean`, `first_diff_std`.

Các mode tóm tắt FTS (`fts_summary_mode`):

- `"diag"` — tổng xác suất tự chuyển (self-transition)
- `"high"` — hoạt động trạng thái cao nhất
- `"diag_high"` — trung bình chuẩn hoá của diag + high *(mặc định)*
- `"entropy"` — 1 − normalized Shannon entropy

---

### `src/fbn_pipeline/metrics.py`

Chứa các hàm đánh giá mô hình.

Nhiệm vụ:

- tìm threshold tối ưu từ xác suất OOF
- tính accuracy, balanced accuracy, F1, macro-F1, AUC
- tính metric theo từng subject
- in bảng metric và config gọn gàng

Các hàm:

- `find_best_threshold(y_true, y_prob, metric, low, high, n_grid)` — sweep thresholds, hỗ trợ `accuracy`/`balanced_accuracy`/`f1`/`macro_f1`
- `print_metrics(y_true, y_pred, y_prob, title, compact)` — in metrics + confusion matrix
- `per_subject_metrics(y_true, y_pred, y_prob, subjects)` — metrics per subject
- `print_per_subject_table(rows)` — in bảng per-subject
- `print_config(cfg_dict)` — in cấu hình thí nghiệm

---

### `src/fbn_pipeline/pipeline.py`

Đây là file trung tâm của project.

Nhiệm vụ:

- xây dựng pipeline hoàn chỉnh theo `cfg.ablation`
- chạy GroupKFold CV (grouped theo subject)
- sinh OOF posterior
- retrain trên full train và đánh giá test
- xử lý calibration (Platt / isotonic), reliability weighting, feature scaling

Các hàm public:

- `run_cv(X_train, y_train, user_train, cfg)` → `(oof_post, oof_y, fold_indices)`
- `retrain_and_test(X_train, y_train, X_test, y_test, best_thr, cfg)` → `(pred, post, metrics, models)`
- `compute_oof_metrics(y_true, y_prob, threshold)` → dict metrics

Helper nội bộ:

- `_extract_features_for_ablation()` — dispatch Stage 1 theo ablation
- `_fit_signal_models()` — train một ANFIS per signal
- `_calibrate_probs()` — Platt hoặc isotonic calibration
- `_compute_signal_weights()` — tính AUC-based weights per signal
- `_fuse_without_bn()` — mean/weighted fusion cho exp2

---

### `src/fbn_pipeline/reporting.py`

Chứa các hàm xuất báo cáo và explainability.

Nhiệm vụ:

- ghi CSV kết quả
- tạo row log từ config + metrics
- in luật ANFIS
- in ma trận chuyển FTS
- in CPT của Bayesian Network

Các hàm:

- `log_to_csv(filepath, row)` — append row vào CSV (tự tạo header nếu file chưa tồn tại)
- `make_result_row(cfg, exp_name, best_thr, oof, test, ...)` → dict phù hợp với `CSV_COLUMNS`
- `dump_anfis_rules(anfis_models, n_sets, top_k)` — in IF-THEN rules per signal
- `dump_fts_transitions(fts, X_sample, top_k)` — in ma trận chuyển FTS
- `dump_bn_cpt(bn)` — in CPT của BNFusion
- `feature_names_for_fts(n_sets)` — tạo tên feature FTS

---

### `src/fbn_pipeline/bn_dashboard.py`

Chứa các hàm export artifacts cho Streamlit dashboard.

Nhiệm vụ:

- export JSON mô tả BN: prior, CPT, signal weights, sample inference trace, metrics
- export JSON luật ANFIS
- ghi toàn bộ artifacts vào thư mục `artifacts/<run_name>/`

Các hàm:

- `export_bn_details_json(bn, cfg, exp_name, best_threshold, test_metrics, ...)` → dict + file JSON
- `export_bn_dashboard_artifacts(models, cfg, exp_name, best_thr, test_metrics, ...)` — gọi tất cả export functions và ghi file

---

## 4. Thư mục `models/`

Thư mục này chỉ chứa **model/core algorithm**.

### `src/fbn_pipeline/models/__init__.py`

Export các model chính:

```python
from .fts_encoder import FTSEncoder
from .anfis_tsk import ANFIS_TSK
from .bn_fusion import BNFusion, SIGNAL_NAMES
```

---

### `src/fbn_pipeline/models/fts_encoder.py`

Chứa lớp `FTSEncoder`.

Nhiệm vụ:

- fit membership functions (triangular/gaussian/trapezoid/gbell) từ training data
- fuzzify mỗi time point → membership degrees
- xây dựng ma trận chuyển `T_ij = Σ μ_i(x_t) · μ_j(x_{t+1})`
- normalize theo `total_sum` hoặc `row`
- concatenate → `n_signals × n_sets²` features per window

Ý nghĩa trong pipeline: **Stage 1** — biến raw physiological windows `(N, 4, T)` thành FTS feature vector `(N, 4·n_sets²)`.

---

### `src/fbn_pipeline/models/anfis_tsk.py`

Chứa lớp `ANFIS_TSK`.

Kiến trúc 5 lớp TSK/Sugeno:

- **Layer 1** — Gaussian MFs (premise), khởi tạo tâm bằng KMeans
- **Layer 2** — Firing strength (tích trong log-space)
- **Layer 3** — Normalised firing strength (softmax)
- **Layer 4** — Linear consequent TSK: `f_r(x) = a_r^T x + b_r`
- **Layer 5** — Output: `ŷ = Σ w̄_r · f_r(x)`

Hybrid learning: forward (ridge least-squares cho consequent) + backward (gradient descent cho premise).

Đầu ra qua sigmoid → `p ∈ [0,1]`.

Ý nghĩa trong pipeline: **Stage 2** — một ANFIS riêng per signal, xuất `P(task | signal_s)`.

---

### `src/fbn_pipeline/models/bn_fusion.py`

Chứa lớp `BNFusion` và hằng `SIGNAL_NAMES = ["Temp", "HR", "GSR", "RR"]`.

DAG: `Task → E_Temp, E_HR, E_GSR, E_RR` (Naive Bayes structure).

Discretisation: quantile-based binning fitted on training fold only.

CPT estimation: counting + Laplace smoothing (`alpha`).

Inference modes:

- **Hard evidence** *(mặc định)*: `P(Task=c | e) ∝ P(c) · ∏_s P(E_s=e_s | c)`
- **Soft evidence** (`bn_soft=True`): dùng Gaussian kernel membership qua bin midpoints
- **Reliability weighting** (`bn_reliability=True`): log-weighted bởi per-signal AUC trên train fold

Ý nghĩa trong pipeline: **Stage 3** — hợp nhất evidence từ 4 tín hiệu → posterior `P(task | evidence)`.

---

## 5. Thư mục dữ liệu

### `last_30s_segments/`

Thư mục chứa dữ liệu đầu vào (30-second windows cuối mỗi segment).

Các file:

- `hr.xlsx`: Heart Rate
- `rr.xlsx`: Respiration Rate
- `gsr.xlsx`: Galvanic Skin Response
- `temp.xlsx`: Skin Temperature
- `labels.xlsx`: nhãn (`rest`/task) và `user_id` cho từng segment

Split cố định: 18 subjects train, 5 subjects held-out test.

---

## 6. Luồng chạy tổng quát

```
main.py
  │
  ├─ build_config()          → ExperimentConfig (preset + CLI overrides)
  ├─ load_dataset()          → DatasetSplit (X, y, user_ids)
  │
  ├─ run_cv()                GroupKFold CV
  │   ├─ _extract_features_for_ablation()   Stage 1
  │   ├─ _fit_signal_models()               Stage 2
  │   ├─ _calibrate_probs()                 (optional)
  │   └─ BNFusion / _fuse_without_bn()      Stage 3
  │
  ├─ find_best_threshold()   → best_thr
  │
  ├─ retrain_and_test()      retrain full + evaluate held-out test
  │
  ├─ print_metrics()         OOF + test metrics
  ├─ print_per_subject_table()
  ├─ dump_anfis_rules() / dump_fts_transitions()   (nếu --dump-rules)
  │
  └─ log_to_csv()            → results_main.csv
     export_bn_dashboard_artifacts()  → artifacts/<run>/
```

---

## 7. Ablation Experiments

| Ablation | Pipeline                             | Stage bỏ          |
| -------- | ------------------------------------ | ------------------ |
| `exp1` | FTS → ANFIS → BN                   | — (full pipeline) |
| `exp2` | FTS → ANFIS → Mean/Weighted fusion | Stage 3 (BN)       |
| `exp3` | FTS → Summary → BN                 | Stage 2 (ANFIS)    |
| `exp4` | Stat features → ANFIS → BN         | Stage 1 (FTS)      |

---

## 8. Cách chạy

### CLI

```bash
# exp1 = full pipeline: FTS → ANFIS → BN
python main.py --ablation exp1 --preset A2_soft

# exp2 = FTS → ANFIS → mean fusion (no BN)
python main.py --ablation exp2 --preset A2_soft --fusion mean

# exp2 = FTS → ANFIS → AUC-weighted fusion
python main.py --ablation exp2 --preset A2_soft --fusion weighted_mean

# exp3 = FTS → summary score → BN (no ANFIS)
python main.py --ablation exp3 --preset A2_soft --fts-summary-mode diag_high

# exp3 với entropy mode
python main.py --ablation exp3 --preset A2_soft --fts-summary-mode entropy

# exp4 = stat features → ANFIS → BN (no FTS)
python main.py --ablation exp4 --preset A2_soft
```

### Notebook

Đặt notebook ở thư mục gốc project, sau đó:

```python
%run main.py
```

### Streamlit Dashboard

```bash
streamlit run app.py
```

## 9. Kết luận

Project được chia thành 3 tầng:

- **data/config layer**: `config.py`, `data.py`
- **core model layer**: `models/` (`fts_encoder.py`, `anfis_tsk.py`, `bn_fusion.py`)
- **orchestration/evaluation layer**: `pipeline.py`, `features.py`, `metrics.py`, `reporting.py`, `bn_dashboard.py`, `main.py`
