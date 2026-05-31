# Dead Code Report — `fts-anfis-bns-final`

Phân tích toàn bộ `src/fbn_pipeline/` và các file top-level (`main.py`, `app.py`, `run_experiments.py`, `fix_nb.py`).  
Mỗi mục đều nêu rõ **nơi định nghĩa**, **lý do xác định là không dùng**, và **đề xuất xử lý**.

---

## 1. Hàm không bao giờ được gọi

### 1.1 `inner_cv_score()` — `src/fbn_pipeline/pipeline.py:453`

```python
def inner_cv_score(X, y, users, cfg) -> tuple[float, float, dict]:
    ...
```

**Trạng thái:** Chỉ xuất hiện đúng 1 lần (tại định nghĩa). Không có bất kỳ file nào trong codebase gọi hàm này.  
**Mục đích ban đầu:** Dùng cho nested cross-validation / hyperparameter search bên trong.  
**Đề xuất:** Xóa hoặc chuyển vào `notebooks/` nếu còn dùng cho thử nghiệm.

---

### 1.2 `compute_metrics()` — `src/fbn_pipeline/metrics.py:208`

```python
def compute_metrics(y_true, y_prob, threshold) -> dict[str, float]:
    ...
```

**Trạng thái:** Chỉ xuất hiện tại định nghĩa. Không được gọi từ bất cứ đâu.  
**Ghi chú:** Có vẻ là một phiên bản đơn giản hoá của `print_metrics()` trả về dict thay vì in ra. Chức năng của nó đã được tích hợp ngầm vào `compute_oof_metrics()` trong `pipeline.py`.  
**Đề xuất:** Xóa.

---

### 1.3 `feature_names_stat()` — `src/fbn_pipeline/features.py:98`

```python
def feature_names_stat(n_signals: int = 4) -> list[str]:
    ...
```

**Trạng thái:** Chỉ xuất hiện trong docstring module (dòng 17) và tại định nghĩa. Không được import hay gọi từ bất kỳ file nào khác.  
**Đề xuất:** Xóa, hoặc giữ lại nếu có kế hoạch dùng cho explainability khi `ablation=exp4`.

---

### 1.4 `split_features_per_signal()` — `src/fbn_pipeline/features.py:127`

```python
def split_features_per_signal(F, n_signals) -> list[np.ndarray]:
    ...
```

**Trạng thái:** Chỉ xuất hiện trong docstring module (dòng 18) và tại định nghĩa. Không được gọi từ bất kỳ file nào.  
**Đề xuất:** Xóa.

---

## 2. Biến toàn cục không được sử dụng

### 2.1 `BIN_NAMES` — `src/fbn_pipeline/models/bn_fusion.py:36`

```python
BIN_NAMES = ["low", "med", "high"]  # kept for backward compat
```

**Trạng thái:** Được định nghĩa với chú thích "backward compat" nhưng không được tham chiếu ở bất kỳ đâu trong toàn bộ codebase. Code hiện tại dùng `self.bin_names_` (sinh ra từ `_make_bin_names(n_bins)`) thay thế.  
**Đề xuất:** Xóa.

---

## 3. Import thừa trong một file

### 3.1 `print_config` — import thừa trong `run_experiments.py:85`

```python
from fbn_pipeline.metrics import (
    ...
    print_config,          # ← import nhưng không gọi
    ...
)
```

**Trạng thái:** `print_config` được import vào `run_experiments.py` nhưng không bao giờ được gọi bên trong file đó. Hàm này có được dùng trong `main.py` (dòng 313).  
**Đề xuất:** Xóa dòng import `print_config` khỏi `run_experiments.py`.

---

## 4. Script tiện ích độc lập

### 4.1 `fix_nb.py`

**Trạng thái:** File này là một script one-off dùng để sửa cell ID trong notebook JSON. Không được import hay gọi từ bất kỳ file nào. Không thuộc pipeline chính.  
**Đề xuất:** Chuyển sang thư mục `tools/` hoặc xóa nếu không còn cần thiết.

---

## 5. Preset cấu hình không được sử dụng trong `config.py`

Các preset sau được định nghĩa trong `EXPERIMENT_PRESETS` nhưng **không được tham chiếu trong bất kỳ file `.py` nào** (ngoài định nghĩa trong `config.py`).

| Preset | Ghi chú |
|---|---|
| `A1` | Chỉ xuất hiện trong chuỗi help text của argparse (`main.py:98`) |
| `A2` | Chỉ xuất hiện trong chuỗi help text — phiên bản hard evidence, đã được thay bằng `A2_soft` |
| `A3` | Không được tham chiếu ở đâu |
| `A4` | Không được tham chiếu ở đâu |
| `A5` | Không được tham chiếu ở đâu |
| `A2_plus` | Chỉ xuất hiện trong chuỗi help text |
| `A2_cal` | Không được tham chiếu ở đâu |
| `A2_new` | Không được tham chiếu ở đâu |
| `A2_hard_tan` | Không được tham chiếu ở đâu |
| `A2_soft_anfis_gbell` | Chỉ được dùng trong `notebooks/gbell-test.ipynb` (notebook thử nghiệm riêng) |

> **Lưu ý:** Các preset `EXP1_*`, `EXP2_*`, `EXP3_*`, `A2_soft`, `A2_gbell`, `A2_soft_bin3`, `A2_soft_nset4_bin3`, `A2_soft_tan` — tất cả đều **đang được dùng** trong `run_experiments.py` hoặc các notebook chính.

**Đề xuất:** Gom các preset thử nghiệm cũ (`A1`–`A5`, `A2_plus`, `A2_cal`, `A2_new`) vào một khối comment `# Legacy / unused presets` hoặc xóa nếu không cần backward compat.

---

## 6. Tóm tắt ưu tiên xử lý

| Ưu tiên | Thành phần | Hành động |
|---|---|---|
| 🔴 Cao | `BIN_NAMES` (bn_fusion.py) | Xóa ngay |
| 🔴 Cao | `compute_metrics()` (metrics.py) | Xóa ngay |
| 🔴 Cao | `inner_cv_score()` (pipeline.py) | Xóa ngay |
| 🔴 Cao | Import `print_config` trong run_experiments.py | Xóa import thừa |
| 🟡 Trung bình | `feature_names_stat()` (features.py) | Xóa hoặc ghi chú rõ mục đích |
| 🟡 Trung bình | `split_features_per_signal()` (features.py) | Xóa |
| 🟢 Thấp | Các preset cũ (A1–A5, A2_plus, A2_cal, v.v.) | Gom vào section `# Legacy` |
| 🟢 Thấp | `fix_nb.py` | Chuyển sang `tools/` hoặc xóa |

---

*Được tạo ngày 28/05/2026. Phân tích dựa trên static grep — không bao gồm các lời gọi động qua `importlib` hay chuỗi string.*
