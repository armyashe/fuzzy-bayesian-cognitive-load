# Hướng dẫn chạy Streamlit Dashboard

Tài liệu này hướng dẫn cách cài đặt môi trường và chạy dashboard Streamlit cho project FTS–ANFIS–Bayesian Network

---

## 2. Mở terminal tại thư mục gốc project

Mở terminal:

---

## 3. Tạo môi trường ảo

```powershell
python -m venv .venv
```

Kích hoạt môi trường ảo:

```powershell
.\.venv\Scripts\Activate.ps1
```

Nếu PowerShell báo lỗi policy, chạy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Khi kích hoạt thành công, terminal sẽ có dạng:

```text
(.venv) PS ...
```

---

## 4. Cài đặt thư viện

Nếu project đã có `requirements.txt`, chạy:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu chưa có `requirements.txt`, có thể cài thủ công:

---

## 5. Chạy Streamlit Dashboard

Chạy lệnh:

```powershell
python -m streamlit run app.py
```

Sau khi chạy, Streamlit sẽ mở trình duyệt tại:

```text
http://localhost:8501
```

Nếu trình duyệt không tự mở, copy link trong terminal và dán vào Chrome.

---

## 6. Cách sử dụng dashboard

Trong sidebar bên trái:

```text
Preset: A2_soft

Ablation experiment:
  exp1 — FTS → ANFIS → BN
  exp2 — FTS → ANFIS → Mean
  exp3 — FTS → Summary → BN
  exp4 — Stat → ANFIS → BN

Data path: last_30s_segments
```

Sau đó bấm:

```text
Run / refresh experiment
```

Dashboard sẽ chạy pipeline và hiển thị các tab:

```text
Proposed Method
Overview
Experiment Metrics
Stage-by-Stage Demo
Rules & CPT
```

---

## 7. Ý nghĩa các thực nghiệm

| Experiment | Pipeline             | Ý nghĩa            |
| ---------- | -------------------- | -------------------- |
| exp1       | FTS → ANFIS → BN   | Mô hình đầy đủ |
| exp2       | FTS → ANFIS → Mean | Bỏ Bayesian Network |
| exp3       | FTS → Summary → BN | Bỏ ANFIS            |
| exp4       | Stat → ANFIS → BN  | Bỏ FTS              |

Lưu ý: tab `Stage-by-Stage Demo` chỉ chạy đầy đủ với `exp1`, vì exp1 có đủ cả ba thành phần FTS, ANFIS và BN. Với exp2, exp3, exp4, một thành phần đã bị loại bỏ nên tab này sẽ hiển thị cảnh báo.
