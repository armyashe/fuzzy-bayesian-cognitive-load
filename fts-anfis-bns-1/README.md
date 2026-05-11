# FBN Pipeline Project Structure

## 1. Tổng quan

Project này tổ chức lại pipeline **FTS -> ANFIS -> Bayesian Network** theo hướng dễ bảo trì hơn.

Mục tiêu của cấu trúc mới là:
- tách **model code** ra khỏi **logic chạy thí nghiệm**
- dễ import lại trong notebook bằng `%run main.py`
- dễ mở rộng thêm `grid search`, `test`, `reporting`

---

## 2. Cấu trúc thư mục đề xuất

```text
project/
├─ main.py
├─ README.md
├─ last_30s_segments/
│  ├─ gsr.xlsx
│  ├─ hr.xlsx
│  ├─ labels.xlsx
│  ├─ rr.xlsx
│  └─ temp.xlsx
└─ src/
   └─ fbn_pipeline/
      ├─ __init__.py
      ├─ config.py
      ├─ data.py
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
- chọn preset/config để chạy
- gọi hàm load dữ liệu
- gọi pipeline huấn luyện và đánh giá
- in kết quả ra màn hình
- ghi CSV nếu cần

Notebook có thể chạy bằng:

```python
%run main.py
```

---

### `src/fbn_pipeline/__init__.py`
File đánh dấu `fbn_pipeline` là một Python package.

Thường dùng để export các thành phần quan trọng ở mức package, ví dụ:
- `ExperimentConfig`

Mục đích:
- giúp import gọn hơn
- giúp Python nhận diện đây là package hợp lệ

---

### `src/fbn_pipeline/config.py`
Chứa toàn bộ phần cấu hình thí nghiệm.

Nội dung nên đặt ở đây:
- `ExperimentConfig`
- `experiment_presets`
- `apply_preset()`
- các giá trị mặc định của FTS / ANFIS / BN

Mục đích:
- gom toàn bộ hyperparameter về một chỗ
- tránh hard-code rải rác trong file chạy
- dễ tạo preset như `A1`, `A2`, `A2_plus`

Ví dụ các nhóm tham số:
- FTS: `n_sets`, `mf_type`, `mf_sigma_scale`, `fts_norm`
- ANFIS: `n_rules`, `sigma_scale`, `ridge_lambda`, `n_epochs`, `lr`
- BN: `n_bins`, `alpha`, `bn_soft`, `bn_reliability`
- evaluation: `k_folds`, `threshold_metric`, `rank_metric`

---

### `src/fbn_pipeline/data.py`
Chứa logic đọc dữ liệu từ thư mục dữ liệu gốc.

Nhiệm vụ:
- đọc các file Excel trong `last_30s_segments/`
- ghép dữ liệu các tín hiệu: `HR`, `RR`, `GSR`, `Temp`
- đọc nhãn từ `labels.xlsx`
- xử lý shape về dạng phù hợp cho pipeline
- tách train/test hoặc trả ra các mảng cần dùng

Mục đích:
- tách phần I/O dữ liệu ra khỏi file chạy
- giúp sau này thay đổi nguồn dữ liệu mà không ảnh hưởng model

Đầu ra thường là:
- `X_train`, `y_train`, `user_train`
- `X_test`, `y_test`, `user_test`

hoặc một biến thể tương tự.

---

### `src/fbn_pipeline/metrics.py`
Chứa các hàm đánh giá mô hình.

Nhiệm vụ:
- tìm threshold tối ưu từ xác suất OOF
- tính accuracy, balanced accuracy, F1, macro-F1, AUC
- tính metric theo từng subject
- in bảng metric gọn gàng

Các hàm thường nằm ở đây:
- `find_best_threshold()`
- `print_metrics()`
- `per_subject_metrics()`
- `print_per_subject_table()`

Mục đích:
- gom toàn bộ phần evaluation vào một chỗ
- tránh lặp code tính metric giữa CV và final test

---

### `src/fbn_pipeline/pipeline.py`
Đây là file trung tâm của project sau khi refactor.

Nhiệm vụ:
- xây dựng pipeline hoàn chỉnh: **FTS -> ANFIS -> BN**
- chạy GroupKFold CV
- sinh OOF posterior
- retrain trên full train và đánh giá test
- chứa logic calibration, reliability weighting, scaling nếu có

Các hàm thường nằm ở đây:
- `run_cv()`
- `retrain_and_test()`
- helper nội bộ cho training/inference

Mục đích:
- tách orchestration ra khỏi model code
- giữ `main.py` mỏng hơn
- giúp tái sử dụng pipeline cho cả `single run` và `grid search`

---

### `src/fbn_pipeline/reporting.py`
Chứa các hàm xuất báo cáo và phần explainability.

Nhiệm vụ:
- ghi CSV kết quả
- tạo row log từ config + metrics
- in luật ANFIS
- in ma trận chuyển FTS
- in CPT của Bayesian Network

Các hàm thường nằm ở đây:
- `log_to_csv()`
- `make_row()`
- `dump_anfis_rules()`
- `dump_fts_transitions()`
- `dump_bn_cpt()`

Mục đích:
- tách riêng phần output/report khỏi pipeline tính toán
- giúp code rõ trách nhiệm hơn

---

## 4. Thư mục `models/`

Thư mục này chỉ nên chứa **model/core algorithm**, không nên chứa logic chạy thí nghiệm dài dòng.

### `src/fbn_pipeline/models/__init__.py`
Dùng để export các model chính:
- `FTSEncoder`
- `ANFIS_TSK`
- `BNFusion`

Ví dụ:

```python
from .fts_encoder import FTSEncoder
from .anfis_tsk import ANFIS_TSK
from .bn_fusion import BNFusion
```

---

### `src/fbn_pipeline/models/fts_encoder.py`
Chứa lớp `FTSEncoder`.

Nhiệm vụ:
- fuzzify chuỗi tín hiệu đầu vào
- xây dựng đặc trưng FTS transition
- biến đổi mỗi tín hiệu thành feature vector đầu vào cho ANFIS

Ý nghĩa trong pipeline:
- đây là **Stage 1**
- biến raw physiological windows thành đặc trưng mờ có tính diễn giải

Đầu vào thường là:
- `X` có shape gần dạng `(N, n_signals, T)`

Đầu ra:
- vector đặc trưng FTS cho từng mẫu

---

### `src/fbn_pipeline/models/anfis_tsk.py`
Chứa lớp `ANFIS_TSK`.

Nhiệm vụ:
- huấn luyện mô hình ANFIS 5 lớp kiểu TSK
- học luật mờ từ đặc trưng FTS
- xuất xác suất hoặc score cho từng tín hiệu

Ý nghĩa trong pipeline:
- đây là **Stage 2**
- mỗi tín hiệu thường có một ANFIS riêng

Đầu ra:
- xác suất `p(task | signal_features)` hoặc score tương đương

---

### `src/fbn_pipeline/models/bn_fusion.py`
Chứa lớp `BNFusion` và các hằng như `SIGNAL_NAMES`.

Nhiệm vụ:
- nhận đầu vào là xác suất từ các ANFIS theo từng tín hiệu
- lượng tử hóa hoặc dùng soft evidence
- ước lượng prior và CPT
- suy luận posterior cuối cùng của lớp đích

Ý nghĩa trong pipeline:
- đây là **Stage 3**
- hợp nhất evidence từ nhiều tín hiệu sinh lý theo hướng interpretable

Đầu ra:
- posterior cuối cùng, ví dụ `P(task | evidence)`

---

## 5. Thư mục dữ liệu

### `last_30s_segments/`
Thư mục chứa dữ liệu đầu vào theo từng tín hiệu.

Các file:
- `hr.xlsx`: Heart Rate
- `rr.xlsx`: Respiration Rate
- `gsr.xlsx`: Galvanic Skin Response
- `temp.xlsx`: Skin Temperature
- `labels.xlsx`: nhãn và/hoặc thông tin subject

Mục đích:
- tách dữ liệu khỏi source code
- giúp thay dataset mà không cần sửa model

---

## 6. Luồng chạy tổng quát

Pipeline tổng quát:

1. `main.py` đọc config
2. `data.py` load dữ liệu
3. `pipeline.py` gọi `FTSEncoder` để sinh FTS features
4. `pipeline.py` huấn luyện `ANFIS_TSK` cho từng tín hiệu
5. `pipeline.py` hợp nhất bằng `BNFusion`
6. `metrics.py` tính các chỉ số đánh giá
7. `reporting.py` ghi CSV hoặc in luật/CPT nếu cần

---

## 7. Cách chạy trong notebook

Đặt notebook ở thư mục gốc project, sau đó chạy:

```python
%run main.py
```

Nếu `main.py` được viết đúng, notebook không cần tự thêm `sys.path` nữa.

---

## 8. Nguyên tắc tổ chức code

### Nên để trong `models/`
- class mô hình
- hàm fit / predict / transform
- logic toán học cốt lõi

### Không nên để trong `models/`
- parse CLI
- load file Excel
- in report dài
- grid search dài
- logging CSV

### Nên để trong `pipeline.py`
- code ghép nhiều model lại với nhau
- cross-validation
- retrain/test
- calibration / thresholding / fusion flow

### Nên để trong `main.py`
- điểm bắt đầu chạy chương trình
- chọn preset
- gọi các module đã tách ra

---

## 9. Gợi ý mở rộng sau này

Bạn có thể bổ sung thêm:
- `search.py`: chứa grid search / nested CV
- `cli.py`: parse argparse riêng
- `tests/`: unit test cho từng module
- `notebooks/`: các file `.ipynb` chạy thí nghiệm

Ví dụ:

```text
project/
├─ notebooks/
├─ tests/
└─ src/
   └─ fbn_pipeline/
      ├─ search.py
      └─ cli.py
```

---

## 10. Kết luận

Sau khi refactor, project được chia thành 3 tầng rõ ràng:

- **data/config layer**: `config.py`, `data.py`
- **core model layer**: `models/`
- **orchestration/evaluation layer**: `pipeline.py`, `metrics.py`, `reporting.py`, `main.py`

Cách chia này giúp:
- dễ đọc hơn
- dễ debug hơn
- dễ tái sử dụng trong notebook
- dễ phát triển thành code nghiên cứu hoặc code package lâu dài
