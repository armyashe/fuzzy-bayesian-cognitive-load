# Pseudocode — Pipeline FTS–ANFIS–Bayesian Network

> Khôi phục từ source code thực tế. Mọi tên hàm, class, tham số đều bám sát code.
> Tài liệu này mô tả đúng 3 pipeline được báo cáo trong khóa luận.

---

## 1. Tổng quan pipeline trong code

Pipeline phân loại tải nhận thức (REST vs. TASK) từ 4 tín hiệu sinh lý (Temp, HR, GSR, RR):

```
Stage 0 : Tải dữ liệu + chia train/test theo subject ID (cố định trong data.py)
Stage 1 : FTS Encoding      — FTSEncoder.fit_transform() / transform()
Stage 2 : ANFIS (per-signal) — ANFIS_TSK.fit() / predict_proba()
Stage 3 : BN Fusion          — BNFusion.fit() / predict_proba()
Eval    : GroupKFold OOF → threshold selection → retrain → test
```

**Ba pipeline được báo cáo trong khóa luận:**

| Pipeline | Stage 1 | Stage 2 | Stage 3 | Vai trò trong khóa luận |
|---|---|---|---|---|
| FTS–ANFIS–BN | FTS Encoding | ANFIS | Bayesian Network | Mô hình đầy đủ (đề xuất) |
| FTS–ANFIS–Mean | FTS Encoding | ANFIS | Mean Fusion | Đánh giá vai trò của Bayesian Network |
| FTS Summary–BN | FTS Encoding → Summary Score | Bypass ANFIS | Bayesian Network | Đánh giá vai trò của ANFIS |

---

## 2. Call graph của chương trình

```
main.py → run_single()
 ├── build_config()                     # config.py: apply_preset, merge overrides
 ├── load_dataset()                     # data.py: đọc 4 xlsx, chia train/test
 │                                      #   train = 18 subjects, test = 5 subjects
 │
 ├── [1/3] run_cv(X_train, y_train, user_train, cfg)           # pipeline.py
 │     └── GroupKFold(n_splits=k_folds).split(X, y, user_ids)
 │           FOR each fold (tr_idx, va_idx):
 │             ├── FTSEncoder.fit_transform() / .transform()   # fts_encoder.py
 │             │
 │             ├── [FTS–ANFIS–BN]
 │             │     ├── _fit_signal_models()                  # pipeline.py
 │             │     │     FOR each signal s in {0,1,2,3}:
 │             │     │       ANFIS_TSK.fit() + .predict_proba() # anfis_tsk.py
 │             │     └── BNFusion.fit() + .predict_proba()     # bn_fusion.py
 │             │
 │             ├── [FTS–ANFIS–Mean]
 │             │     ├── _fit_signal_models()
 │             │     └── _fuse_without_bn() → mean(probs)      # pipeline.py
 │             │
 │             └── [FTS Summary–BN]
 │                   ├── summarise_fts_per_signal()             # features.py
 │                   └── BNFusion.fit() + .predict_proba()
 │
 ├── find_best_threshold(oof_post, oof_y)                       # metrics.py
 │
 ├── [2/3] retrain_and_test(X_train, y_train, X_test, y_test, τ, cfg)  # pipeline.py
 │     └── (quy trình giống run_cv nhưng fit trên toàn bộ X_train)
 │
 └── [3/3] Reporting
       ├── print_metrics()                                      # metrics.py
       ├── per_subject_metrics()                                # metrics.py
       ├── dump_anfis_rules() / dump_bn_cpt()                   # reporting.py
       └── log_to_csv()                                         # reporting.py
```

---

## 3. Algorithm 1: Full FTS–ANFIS–BN Pipeline

```
Algorithm 1: Full FTS–ANFIS–BN Pipeline

Source files:
    main.py        → run_single()
    pipeline.py    → run_cv(), retrain_and_test()
    data.py        → load_dataset()

Input:
    X      : physiological windows, shape (N, 4, T)
    y      : labels, y ∈ {REST=0, TASK=1}
    groups : subject IDs
    config : ExperimentConfig (preset, n_sets, n_rules, n_bins, bn_soft, ...)

Output:
    test_pred    : (N_test,) — binary predictions {REST=0, TASK=1}
    test_post    : (N_test,) — posterior P(TASK | evidence)
    test_metrics : dict      — accuracy, f1, auc, precision, recall, bal_acc

─────────────────────────────────────────────────────────────────
PHASE 0: Load Data
─────────────────────────────────────────────────────────────────
    dataset ← load_dataset(data_folder)
        X_train  : (N_tr, 4, T)   — 18 training subjects
        y_train  : (N_tr,)        — {0=REST, 1=TASK}
        user_train: (N_tr,)       — subject IDs
        X_test   : (N_te, 4, T)   — 5 held-out test subjects (never used until Phase 4)
        y_test   : (N_te,)

─────────────────────────────────────────────────────────────────
PHASE 1: GroupKFold Cross-Validation  →  OOF posteriors
─────────────────────────────────────────────────────────────────
    oof_post ← zeros(N_tr)

    FOR fold k = 1 … K  (K = cfg.k_folds, default 5):
        (tr_idx, va_idx) ← GroupKFold.split(X_train, y_train, user_train)
        // Guarantee: subjects in tr_idx and va_idx are disjoint
        X_tr, y_tr ← X_train[tr_idx], y_train[tr_idx]
        X_va, y_va ← X_train[va_idx], y_train[va_idx]

        // Stage 1: FTS Encoding (fit only on fold train)
        fts ← FTSEncoder(n_sets, mf_type, mf_sigma_scale, norm_mode="row")
        F_tr ← fts.fit_transform(X_tr)     // (N_tr_fold, 4*n_sets²) = (N, 36)
        F_va ← fts.transform(X_va)         // (N_va_fold, 36)

        // Stage 2: ANFIS per signal
        FOR s = 0, 1, 2, 3  (Temp, HR, GSR, RR):
            z_tr_s ← F_tr[:, s*n_sets² : (s+1)*n_sets²]   // (N_tr, 9)
            z_va_s ← F_va[:, s*n_sets² : (s+1)*n_sets²]   // (N_va, 9)

            anfis_s ← ANFIS_TSK(n_rules, sigma_scale, ridge_lambda, n_epochs, lr)
            anfis_s.fit(z_tr_s, y_tr)

            tr_probs[:, s] ← anfis_s.predict_proba(z_tr_s)   // P(TASK | signal_s)
            va_probs[:, s] ← anfis_s.predict_proba(z_va_s)   // (N_va,)

        // Stage 3: Bayesian Network Fusion
        bn ← BNFusion(n_bins=5, alpha, soft=True)
        bn.fit(tr_probs, y_tr)                         // tr_probs : (N_tr, 4)
        oof_post[va_idx] ← bn.predict_proba(va_probs)  // (N_va,)

─────────────────────────────────────────────────────────────────
PHASE 2: Threshold Selection (OOF only — test set not used)
─────────────────────────────────────────────────────────────────
    τ* ← find_best_threshold(y_train, oof_post,
                              metric="balanced_accuracy",
                              n_grid=181, low=0.05, high=0.95)
    // τ* = argmax_{τ} balanced_accuracy(y_train, oof_post ≥ τ)

─────────────────────────────────────────────────────────────────
PHASE 3: Retrain on Full Training Set
─────────────────────────────────────────────────────────────────
    fts_final ← FTSEncoder(...)
    F_tr_full ← fts_final.fit_transform(X_train)    // (N_tr, 36)
    F_te      ← fts_final.transform(X_test)         // (N_te, 36)

    FOR s = 0, 1, 2, 3:
        anfis_s_final.fit(F_tr_full[:, s*9:(s+1)*9], y_train)
        tr_probs_full[:, s] ← anfis_s_final.predict_proba(F_tr_full[:, s*9:(s+1)*9])
        te_probs[:, s]      ← anfis_s_final.predict_proba(F_te[:, s*9:(s+1)*9])

    bn_final ← BNFusion(n_bins=5, alpha, soft=True)
    bn_final.fit(tr_probs_full, y_train)
    test_post ← bn_final.predict_proba(te_probs)   // (N_te,)

─────────────────────────────────────────────────────────────────
PHASE 4: Test Evaluation (apply τ* from Phase 2 — no adjustment)
─────────────────────────────────────────────────────────────────
    test_pred ← (test_post ≥ τ*).astype(int)
    test_metrics ← compute_oof_metrics(y_test, test_post, τ*)
    // accuracy, f1, auc, precision, recall, balanced_accuracy
```

---

## 4. Algorithm 2: FTS Encoding

```
Algorithm 2: FTS Encoding

Source file:  fts_encoder.py → class FTSEncoder
              Methods: .fit(X), .transform(X), .fit_transform(X)

Input:
    X        : (N, S, T)   — N windows, S=4 signals, T time steps
    n_sets   : int         — số fuzzy sets (= 3 trong cấu hình chính)
    mf_type  : str         — "gaussian" (cấu hình chính A2_soft)
    norm_mode: str         — "row" (cấu hình chính A2_soft)

Output:
    F  : (N, S * n_sets²) = (N, 36)  — 9 transition features per signal

─────────────────────────────────────────────────────────────────
STEP 1: Fit — Học tham số membership function từ dữ liệu train
─────────────────────────────────────────────────────────────────
    medians ← nanmedian(X, axis=(N, T))   // (S,) dùng để impute NaN

    FOR each signal s = 0 … S-1:
        vals ← X[:, s, :].flatten()       // tất cả giá trị tín hiệu s trên train

        // Grid (even-spacing) partitioning cho Gaussian MF
        vmin, vmax ← min(vals), max(vals)
        margin ← (vmax - vmin) * 0.05
        centers[s] ← linspace(vmin-margin, vmax+margin, n_sets)  // (n_sets,)
        spacing[s] ← (vmax+margin - vmin-margin) / (n_sets - 1)
        sigma[s]   ← spacing[s] * mf_sigma_scale

─────────────────────────────────────────────────────────────────
STEP 2: Transform — Fuzzify + Build Transition Matrix
─────────────────────────────────────────────────────────────────
    X ← impute NaN with per-signal medians

    FOR each signal s = 0 … S-1:

        // Step 2a. Fuzzification
        FOR each time step t = 1 … T:
            FOR each fuzzy set k = 0 … n_sets-1:
                μ_k(x_t) ← exp(-0.5 · ((x_t − centers[s][k]) / sigma[s][k])²)
            // Result: μ(x_t) = [μ_Low, μ_Med, μ_High] ∈ R^{n_sets}

        // Step 2b. Build transition matrix
        //     T_ij = Σ_{t=1}^{T-1} μ_i(x_t) · μ_j(x_{t+1})
        mu_prev ← μ[1:T-1, :]     // (T-1, n_sets) — "from" memberships
        mu_next ← μ[2:T,   :]     // (T-1, n_sets) — "to" memberships
        trans ← einsum("ti,tj->ij", mu_prev, mu_next)   // (n_sets, n_sets)

        // Step 2c. Row-normalise transition matrix
        FOR each row i = 0 … n_sets-1:
            trans[i, :] ← trans[i, :] / sum(trans[i, :])
            // each row sums to 1: Σ_j T[i,j] = 1

        // Step 2d. Flatten
        z_s ← trans.flatten()    // (n_sets²,) = (9,) if n_sets=3

    RETURN F ← hstack([z_0, z_1, z_2, z_3])   // (N, 4 * n_sets²) = (N, 36)

─────────────────────────────────────────────────────────────────
Diễn giải kết quả (n_sets=3):
    F[i, 0:9]  = Temp transitions: [T_LL, T_LM, T_LH, T_ML, T_MM, T_MH, T_HL, T_HM, T_HH]
    F[i, 9:18] = HR   transitions
    F[i, 18:27]= GSR  transitions
    F[i, 27:36]= RR   transitions
    T_ij ≈ conditional probability: P(state j at t+1 | state i at t)
─────────────────────────────────────────────────────────────────
```

---

## 5. Algorithm 3: ANFIS Per-Signal Learning

```
Algorithm 3: ANFIS TSK per-signal classifier

Source file:  anfis_tsk.py → class ANFIS_TSK
              Methods: .fit(X, y), .predict_proba(X), .predict_raw(X)

Có 4 mô hình ANFIS riêng biệt:
    ANFIS_Temp, ANFIS_HR, ANFIS_GSR, ANFIS_RR
Mỗi ANFIS nhận đúng 9 đặc trưng FTS của tín hiệu tương ứng.

Input:
    z  : (N, d)   — d = n_sets² = 9 (FTS features of one signal)
    y  : (N,)     — binary labels {0, 1}
    n_rules       — số rule (thường 6–8)
    mf_type       — "gaussian" | "gbell" (chỉ ANFIS, độc lập với FTS)

Output:
    p_s : (N,)    — P(TASK | signal_s) ∈ (0, 1)

─────────────────────────────────────────────────────────────────
INITIALIZATION (trong .fit())
─────────────────────────────────────────────────────────────────
    actual_rules ← min(n_rules, max(N // 3, 2))   // tránh KMeans fail

    // Khởi tạo premise parameters bằng KMeans
    KMeans(n_clusters=actual_rules).fit(z)
    centers ← cluster_centers              // (R, d)
    sigmas  ← per-cluster std * sigma_scale // (R, d)

─────────────────────────────────────────────────────────────────
HYBRID LEARNING LOOP  (n_epochs iterations)
─────────────────────────────────────────────────────────────────
    FOR epoch = 1 … n_epochs:

        ── FORWARD PASS ──────────────────────────────────────────

        // Layer 1: Premise membership (Gaussian or Gbell)
        FOR each rule r = 1 … R:
            FOR each dimension j = 1 … d:
                IF mf_type == "gaussian":
                    μ_{r,j}(z_j) ← exp(-0.5 · ((z_j − c_{r,j}) / σ_{r,j})²)
                ELSE (gbell):
                    μ_{r,j}(z_j) ← 1 / (1 + |(z_j − c_{r,j}) / σ_{r,j}|^{2b})

        // Layer 2: Firing strength (product in log-space for stability)
        FOR each rule r:
            log w_r ← Σ_{j=1}^{d} log μ_{r,j}(z_j)

        // Layer 3: Normalised firing strength (softmax over log-firing)
        FOR each rule r:
            w̄_r ← exp(log w_r) / Σ_{k=1}^{R} exp(log w_k)
            // (numerically stable via log w_r -= max(log w))

        // Layer 4: TSK linear consequent (solved by Weighted Ridge LS)
        // Design matrix Φ[i, r*(d+1):(r+1)*(d+1)] = w̄_r(i) · [z_i; 1]
        Φ ← build_phi(z, W̄)    // (N, R*(d+1))

        // Ridge regression: (Φᵀ Φ + λI) θ = Φᵀ y
        θ ← solve( Φᵀ Φ + λ·I,  Φᵀ y )   // θ shape: (R*(d+1),)

        // Layer 5: Output aggregation
        y_hat ← Φ · θ                     // (N,)  — raw ANFIS output

        ── BACKWARD PASS (GD on premise params, skip on last epoch) ──

        residual ← (2/N) · (y_hat − y)    // dL/dy_hat

        // Per-rule consequent output
        FOR each rule r:
            f_r(z) ← [z; 1] · θ[r*(d+1):(r+1)*(d+1)]   // (N,)

        // Gradient w.r.t. normalised firing: dL/dW̄
        dW̄ ← residual ⊗ f_r     // (N, R)  outer product

        // Softmax backward: dL/d(log_w_s) = w̄_s · (dL/dw̄_s − Σ_r dL/dw̄_r · w̄_r)
        weighted ← (dW̄ ⊙ W̄).sum(axis=1)        // (N,)
        d_logw  ← W̄ ⊙ (dW̄ − weighted)           // (N, R)

        // Gradient w.r.t. centers and sigmas
        IF mf_type == "gaussian":
            dc ← (z - c) / σ²         // (N, R, d)
            dσ ← (z - c)² / σ³
        ELSE (gbell):
            dc ← 2b · (1−μ) / (z−c)  // with safe division
            dσ ← 2b · (1−μ) / σ

        grad_c ← einsum("nr, nrd → rd", d_logw, dc)
        grad_σ ← einsum("nr, nrd → rd", d_logw, dσ)

        // Gradient clipping (norm ≤ 10)
        IF ‖grad_c‖ > 10: grad_c ← 10 · grad_c / ‖grad_c‖

        // Parameter update
        c ← c − lr · grad_c
        σ ← max(σ − lr · grad_σ,  1e-4)

─────────────────────────────────────────────────────────────────
INFERENCE  (predict_proba)
─────────────────────────────────────────────────────────────────
    y_raw ← predict_raw(z)           // Layers 1-5 (no backward)
    p_s   ← sigmoid(y_raw)
           = 1 / (1 + exp(−y_raw))  // P(TASK | signal_s)

─────────────────────────────────────────────────────────────────
Kết quả từ 4 ANFIS:
    p_Temp, p_HR, p_GSR, p_RR ∈ (0,1)  — stacked thành (N, 4)
─────────────────────────────────────────────────────────────────
```

---

## 6. Algorithm 4: Bayesian Network Soft Evidence Fusion

```
Algorithm 4: Soft-Evidence Naive Bayesian Fusion

Source file:  bn_fusion.py → class BNFusion
              Methods: .fit(probs, y), .predict_proba(probs)

BN structure (Naive Bayes):
    CL ──→ E_Temp
    CL ──→ E_HR
    CL ──→ E_GSR
    CL ──→ E_RR

    CL ∈ {REST=0, TASK=1}
    E_s: discrete evidence, n_bins bins (default 5: very_low, low, med, high, very_high)

Input (fit):
    probs : (N, 4)   — ANFIS probabilities [p_Temp, p_HR, p_GSR, p_RR]
    y     : (N,)     — binary labels
    n_bins, alpha    — discretisation bins, Laplace smoothing
    soft             — True → soft evidence mode

Input (predict_proba):
    probs : (N, 4)   — evidence to evaluate
    threshold τ      — decision boundary from OOF selection

Output:
    posterior_task : (N,) — P(TASK | evidence)

─────────────────────────────────────────────────────────────────
FIT PHASE
─────────────────────────────────────────────────────────────────
    // Step 1: Quantile-based bin edges (fit ONLY on train fold)
    qs ← linspace(0, 1, n_bins+1)[1:-1]     // e.g. [0.2, 0.4, 0.6, 0.8] for 5 bins
    FOR each signal s:
        bin_edges[s] ← quantile(probs[:, s], qs)   // (n_bins-1,) boundaries

    // Step 2: Discretise training probabilities
    FOR each signal s:
        ev[:, s] ← digitize(probs[:, s], bin_edges[s])   // {0, 1, …, n_bins-1}

    // Step 3: Prior  P(CL = c)
    FOR c ∈ {0, 1}:
        prior[c] ← (count(y == c) + α) / (N + 2α)

    // Step 4: CPT  P(E_s = b | CL = c)
    FOR each signal s:
        FOR c ∈ {0, 1}:
            FOR b = 0 … n_bins-1:
                cpt[s][c, b] ← (count(ev[y==c, s] == b) + α)
            cpt[s][c] ← cpt[s][c] / sum(cpt[s][c])   // normalize to sum=1

    // Step 5 (soft=True): Soft evidence parameters
    FOR each signal s:
        ext ← [p_min, bin_edges[s][0], …, bin_edges[s][-1], p_max]   // n_bins+1 values
        bin_mids[s] ← 0.5 * (ext[:-1] + ext[1:])                      // (n_bins,)
        bin_sigmas[s] ← avg_bin_width * 0.5                            // Gaussian kernel σ

─────────────────────────────────────────────────────────────────
INFERENCE PHASE — Soft Evidence (soft=True)
─────────────────────────────────────────────────────────────────
    // Step 6: Gaussian kernel soft memberships
    FOR each sample i:
        FOR each signal s:
            FOR b = 0 … n_bins-1:
                μ_{s,b}(p_s) ← exp(-0.5 · ((p_s − mid_{s,b}) / σ_s)²)
            // Normalise: Σ_b μ_{s,b} = 1

    // Step 7: Effective likelihood using soft memberships
    FOR each sample i:
        FOR c ∈ {0, 1}:
            log_score[c] ← log(prior[c])
            FOR each signal s:
                P_eff(E_s | c) ← Σ_{b=0}^{n_bins-1}  μ_{s,b}(p_{i,s}) · cpt[s][c, b]
                log_score[c] ← log_score[c] + w_s · log(P_eff(E_s | c))

        // Step 8: Normalise to posterior (numerically stable softmax)
        log_score ← log_score − max(log_score)
        p_unnorm  ← exp(log_score)
        posterior_task[i] ← p_unnorm[1] / (p_unnorm[0] + p_unnorm[1])

─────────────────────────────────────────────────────────────────
DECISION
─────────────────────────────────────────────────────────────────
    IF posterior_task[i] ≥ τ:
        predict TASK (1)
    ELSE:
        predict REST (0)

─────────────────────────────────────────────────────────────────
Ghi chú:
    - Hard evidence (soft=False): dùng ev[:, s] = digitize(p_s) trực tiếp
      thay vì tính μ, rồi lookup cpt[s][c, ev[i,s]]
    - Cấu hình chính (A2_soft preset): soft=True, n_bins=5, alpha=0.5
─────────────────────────────────────────────────────────────────
```

---

## 7. Algorithm 5: Mean Fusion (FTS–ANFIS–Mean)

```
Algorithm 5: FTS–ANFIS–Mean Fusion

Source file:  pipeline.py → _fuse_without_bn()

Biến thể này thay BNFusion (Stage 3) bằng trung bình cộng đơn giản.
Stage 1 (FTSEncoder) và Stage 2 (ANFIS) giữ nguyên như Algorithm 1.

Input:
    va_probs : (N_va, 4)  — ANFIS probabilities [p_Temp, p_HR, p_GSR, p_RR]

Output:
    posterior : (N_va,)   — fused probability estimate

─────────────────────────────────────────────────────────────────
Fusion (thay thế Stage 3):
─────────────────────────────────────────────────────────────────
    // Trung bình đơn giản 4 tín hiệu
    p_final[i] ← (p_Temp[i] + p_HR[i] + p_GSR[i] + p_RR[i]) / 4
               = mean(va_probs[i, :])

Decision:
    IF p_final[i] ≥ τ:  predict TASK
    ELSE:               predict REST

Ghi chú:
    - BNFusion KHÔNG được khởi tạo hay fit trong biến thể này
    - Threshold τ vẫn được chọn từ OOF posteriors (Phase 2, Algorithm 1)
    - Algorithm 3 (ANFIS) giữ nguyên hoàn toàn
    - Mục đích: đánh giá vai trò của Bayesian Network trong Stage 3
```

---

## 8. Algorithm 6: FTS Summary Score–BN (FTS Summary–BN)

```
Algorithm 6: FTS Summary Score → BN Fusion

Source files:
    features.py  → summarise_fts_per_signal(F, n_sets, mode)
    pipeline.py  → run_cv() và retrain_and_test() với biến thể FTS Summary–BN

Biến thể này thay ANFIS_TSK (Stage 2) bằng heuristic summary score từ FTS.
Stage 1 (FTSEncoder) và Stage 3 (BNFusion) giữ nguyên.
ANFIS hoàn toàn không được sử dụng.

Input:
    F : (N, 4 * n_sets²)  — FTS transition features (output của FTSEncoder)
    n_sets               — số fuzzy sets (3)
    fts_summary_mode     — "diag_high" (default trong A2_soft preset)

Output:
    scores : (N, 4)       — per-signal summary score ∈ [0, 1]

─────────────────────────────────────────────────────────────────
FOR each signal s = 0, 1, 2, 3:
─────────────────────────────────────────────────────────────────
    tm ← F[:, s * n_sets² : (s+1) * n_sets²]     // (N, n_sets²)
    T  ← tm.reshape(N, n_sets, n_sets)             // (N, from_state, to_state)

    MODE "diag_high"  ← DEFAULT trong code (A2_soft preset)
        // Kết hợp trung bình chuẩn hóa của diag score và high score
        diag_score ← Σ_i T[:, i, i]               // trace: tổng self-transitions
        h ← n_sets − 1                             // index trạng thái cao nhất
        to_high   ← Σ_i T[:, i, h]                // chuyển ĐẾN trạng thái high
        from_high ← Σ_j T[:, h, j]                // chuyển TỪ trạng thái high
        high_raw  ← to_high + from_high − T[:, h, h]
        max_high  ← 2 − 1/n_sets                  // = 5/3 ≈ 1.667 khi n_sets=3
        high_norm ← high_raw / max_high
        score[:, s] ← 0.5 · diag_score + 0.5 · high_norm

    score[:, s] ← clip(score[:, s], 0.0, 1.0)

─────────────────────────────────────────────────────────────────
Sau khi có scores (N, 4):
    → Đưa trực tiếp vào BNFusion.fit(scores, y) và .predict_proba(scores)
    → Cùng quy trình inference như Algorithm 4
    → scores đóng vai trò evidence thay cho ANFIS probabilities
─────────────────────────────────────────────────────────────────

Ghi chú:
    - ANFIS_TSK KHÔNG được gọi trong biến thể này
    - FTSEncoder vẫn được fit và transform bình thường (Stage 1 giữ nguyên)
    - scores không phải là xác suất ANFIS; chúng là đặc trưng động học
      từ ma trận chuyển trạng thái FTS
    - Mục đích: đánh giá vai trò của ANFIS trong Stage 2
```
        // Trung bình chuẩn hóa của diag và high
        diag_score ← Σ_i T[:, i, i]
        h ← n_sets − 1
        to_high   ← Σ_i T[:, i, h]
        from_high ← Σ_j T[:, h, j]
        high_raw  ← to_high + from_high − T[:, h, h]
        max_high  ← 2 − 1/n_sets   // = 5/3 ≈ 1.667 khi n_sets=3
        high_norm ← high_raw / max_high
        score[:, s] ← 0.5 · diag_score + 0.5 · high_norm

    ELSE IF mode == "entropy":
        // 1 − entropy chuẩn hóa của phân phối chuyển trạng thái
        // Cao → phân phối tập trung hơn (ít entropy → dự đoán được)
        H ← −Σ_{k} tm_k · log(tm_k)              // Shannon entropy
        H_max ← log(n_sets²)                       // = log(9) ≈ 2.197 khi n_sets=3
        score[:, s] ← 1 − H / H_max

    score[:, s] ← clip(score[:, s], 0.0, 1.0)

─────────────────────────────────────────────────────────────────
Sau khi có scores (N, 4):
    → Đưa trực tiếp vào BNFusion.fit(scores, y) và .predict_proba(scores)
    → Cùng quy trình inference như Algorithm 4
    → scores đóng vai trò evidence thay cho ANFIS probabilities
─────────────────────────────────────────────────────────────────

Ghi chú:
    - ANFIS_TSK KHÔNG được gọi trong biến thể FTS Summary–BN
    - FTSEncoder vẫn được fit và transform bình thường (Stage 1 giữ nguyên)
    - scores không phải là xác suất ANFIS; chúng là đặc trưng động học
      từ ma trận chuyển trạng thái FTS
    - BNFusion sử dụng quantile binning trên scores (không yêu cầu đầu vào ∈ [0,1])
```

---

## 9. Algorithm 7: Evaluation Protocol

```
Algorithm 7: Subject-Independent Evaluation Protocol

Source files:
    pipeline.py → run_cv(), retrain_and_test(), compute_oof_metrics()
    metrics.py  → find_best_threshold(), per_subject_metrics(), print_metrics()
    data.py     → load_dataset() (phân chia cố định theo TRAIN_IDS / TEST_IDS)

Nguyên tắc cốt lõi:
    - Test set (5 subjects) KHÔNG được nhìn trong bất kỳ bước nào trước đánh giá cuối
    - Threshold được chọn HOÀN TOÀN từ OOF trên tập train
    - GroupKFold đảm bảo subject không bị rò rỉ giữa train fold và val fold

─────────────────────────────────────────────────────────────────
PHASE 1: GroupKFold Cross-Validation
─────────────────────────────────────────────────────────────────
    oof_post ← array of NaN, shape (N_train,)
    oof_y    ← copy of y_train

    gkf ← GroupKFold(n_splits=K)

    FOR fold k = 1 … K:
        (tr_idx, va_idx) ← gkf.split(X_train, y_train, user_train)
        // Đảm bảo: user_train[tr_idx] ∩ user_train[va_idx] = ∅

        X_tr, y_tr ← X_train[tr_idx], y_train[tr_idx]
        X_va, y_va ← X_train[va_idx], y_train[va_idx]

        // Toàn bộ pipeline (theo ablation variant) fit HOÀN TOÀN trên fold train
        model_k ← fit_pipeline(X_tr, y_tr, cfg)

        // Dự đoán trên fold validation (chưa được thấy trong quá trình fit)
        oof_post[va_idx] ← model_k.predict_proba(X_va)

        // In per-fold metrics tại threshold = 0.5 (chỉ để debug, không dùng chọn model)
        tr_post_fold ← model_k.predict_proba(X_tr)
        PRINT Fold k:
            Train: Acc, AUC, F1 tại t=0.5
            Val  : Acc, AUC, F1 tại t=0.5

─────────────────────────────────────────────────────────────────
PHASE 2: OOF Threshold Selection
─────────────────────────────────────────────────────────────────
    // Chọn ngưỡng quyết định tối ưu dựa trên OOF posteriors
    // (KHÔNG dùng test set)
    τ*, score* ← find_best_threshold(
        y_true = oof_y,
        y_prob = oof_post,
        metric = "balanced_accuracy",
        n_grid = 181,
        low = 0.05, high = 0.95
    )
    // τ* = argmax_{τ ∈ [0.05, 0.95]} balanced_accuracy(oof_y, oof_post ≥ τ)

─────────────────────────────────────────────────────────────────
PHASE 3: Retrain on Full Training Set
─────────────────────────────────────────────────────────────────
    // Retrain một lần duy nhất trên toàn bộ X_train với τ* đã xác định
    model_final ← fit_pipeline(X_train, y_train, cfg)

─────────────────────────────────────────────────────────────────
PHASE 4: Evaluate on Held-out Test Set
─────────────────────────────────────────────────────────────────
    test_post ← model_final.predict_proba(X_test)   // (N_test,)
    test_pred ← (test_post ≥ τ*).astype(int)        // dùng τ* từ Phase 2

    test_metrics ← compute_oof_metrics(y_test, test_post, τ*):
        accuracy          ← accuracy_score(y_test, test_pred)
        precision         ← precision_score(y_test, test_pred)
        recall            ← recall_score(y_test, test_pred)
        f1                ← f1_score(y_test, test_pred)
        roc_auc           ← roc_auc_score(y_test, test_post)
        balanced_accuracy ← balanced_accuracy_score(y_test, test_pred)

─────────────────────────────────────────────────────────────────
PHASE 5: Per-Subject Analysis (optional, cfg.show_per_subject)
─────────────────────────────────────────────────────────────────
    FOR each subject u ∈ unique(user_test):
        mask ← (user_test == u)
        compute per-subject: Acc, Balanced-Acc, Macro-F1, F1, AUC

─────────────────────────────────────────────────────────────────
Ghi chú quan trọng:
    - "Train accuracy" in fold = accuracy TẠI THRESHOLD 0.5 trên tập train fold
      (chỉ dùng để detect overfitting, KHÔNG ảnh hưởng đến model selection)
    - "OOF accuracy" = accuracy trên toàn bộ tập train với τ* OOF
    - "Test accuracy" = accuracy trên 5 subjects chưa từng được dùng
    - Thứ tự đúng: fit → OOF evaluate → select τ → retrain → test
                   (test set chỉ được dùng MỘT LẦN duy nhất ở bước cuối)
─────────────────────────────────────────────────────────────────
```

---

## 10. Bảng ánh xạ code ↔ thuật toán

| Bước thuật toán | File | Class / Function | Input shape | Output shape | Dùng trong |
|---|---|---|---|---|---|
| Tải dữ liệu | `data.py` | `load_dataset()` | folder path | `DatasetSplit` | Tất cả |
| Cấu hình pipeline | `config.py` | `ExperimentConfig`, `apply_preset()` | dict | `ExperimentConfig` | Tất cả |
| GroupKFold CV | `pipeline.py` | `run_cv()` | `(N,4,T), (N,), (N,)` | `oof_post (N,)` | Tất cả |
| FTS Fit | `fts_encoder.py` | `FTSEncoder.fit()` | `(N,4,T)` | `FTSEncoder` | Tất cả |
| FTS Transform | `fts_encoder.py` | `FTSEncoder.transform()` | `(N,4,T)` | `(N, 4*n_sets²)` | Tất cả |
| ANFIS fit (per signal) | `anfis_tsk.py` | `ANFIS_TSK.fit()` | `(N, n_sets²), (N,)` | `ANFIS_TSK` | FTS–ANFIS–BN, FTS–ANFIS–Mean |
| ANFIS inference | `anfis_tsk.py` | `ANFIS_TSK.predict_proba()` | `(N, n_sets²)` | `(N,)` | FTS–ANFIS–BN, FTS–ANFIS–Mean |
| 4×ANFIS dispatch | `pipeline.py` | `_fit_signal_models()` | `(N, 4*n_sets²)` | `tr_probs, va_probs (N,4)` | FTS–ANFIS–BN, FTS–ANFIS–Mean |
| FTS summary score | `features.py` | `summarise_fts_per_signal()` | `(N, 4*n_sets²)` | `(N, 4)` | FTS Summary–BN |
| BN Fusion fit | `bn_fusion.py` | `BNFusion.fit()` | `(N,4), (N,)` | `BNFusion` | FTS–ANFIS–BN, FTS Summary–BN |
| BN Inference | `bn_fusion.py` | `BNFusion.predict_proba()` | `(N,4)` | `(N,)` | FTS–ANFIS–BN, FTS Summary–BN |
| Mean fusion (no BN) | `pipeline.py` | `_fuse_without_bn()` | `(N,4)` | `(N,)` | FTS–ANFIS–Mean |
| Threshold selection | `metrics.py` | `find_best_threshold()` | `(N,), (N,)` | `τ*, score*` | Tất cả |
| Retrain + test | `pipeline.py` | `retrain_and_test()` | `(N,4,T)*2` | `pred, post, metrics` | Tất cả |
| Tính metrics | `pipeline.py` | `compute_oof_metrics()` | `(N,), (N,), τ` | `dict` | Tất cả |
| Per-subject metrics | `metrics.py` | `per_subject_metrics()` | `(N,)*3, subjects` | `list[dict]` | Tất cả |
| In metrics | `metrics.py` | `print_metrics()` | `y_true, y_pred, y_prob` | stdout | Tất cả |
| Dump ANFIS rules | `reporting.py` | `dump_anfis_rules()` | `list[ANFIS_TSK]` | JSON / stdout | Tùy cờ |
| Dump BN CPT | `reporting.py` | `dump_bn_cpt()` | `BNFusion` | JSON / stdout | Tùy cờ |
| Ghi CSV | `reporting.py` | `log_to_csv()` | `dict` | CSV row | Tùy cờ |

---

## 11. Gợi ý đưa vào khóa luận

Dưới đây là phiên bản rút gọn, học thuật, phù hợp với Chương 3 (Phương pháp đề xuất) hoặc Chương 4 (Thực nghiệm).

---

### Algorithm 1 (Academic) — Full FTS–ANFIS–BN Pipeline

```
Algorithm 1: FTS–ANFIS–BN Cognitive Load Classification

Require: Physiological windows X ∈ ℝ^{N×4×T}, labels y ∈ {0,1}^N,
         subject IDs G ∈ ℤ^N, configuration Θ
Ensure: Predicted labels ŷ_test, posterior P(TASK|e), metrics M

1:  {X_tr, y_tr, G_tr, X_te, y_te} ← SubjectHoldOut(X, y, G)
2:  {p̂_oof, τ*} ← GroupKFold_CV(X_tr, y_tr, G_tr, Θ)          ▷ Alg. 2–4
3:  τ* ← argmax_τ BalancedAccuracy(y_tr, p̂_oof ≥ τ)
4:  {fts*, {ANFIS_s*}, BN*} ← RetainFullModel(X_tr, y_tr, Θ)  ▷ Alg. 2–4
5:  P_test ← [ANFIS_s*(FTS*(X_te[s])) for s ∈ {T,H,G,R}]
6:  p̂_test ← BN*.predict_proba(P_test)
7:  ŷ_test ← 𝟙[p̂_test ≥ τ*]
8:  M ← EvaluateMetrics(y_te, ŷ_test, p̂_test)
9:  return ŷ_test, p̂_test, M
```

---

### Algorithm 2 (Academic) — FTS Encoding

```
Algorithm 2: Fuzzy Transition Matrix Encoding

Require: Signal window x_s ∈ ℝ^T, n fuzzy sets A = {A_1, …, A_n}
         fitted on training data
Ensure: Feature vector z_s ∈ ℝ^{n²}

1:  FOR t = 1 to T DO
2:      μ(x_t) ← [μ_{A_k}(x_t)]_{k=1}^{n}        ▷ Gaussian/triangular MF
3:  END FOR
4:  FOR i,j = 1 to n DO
5:      T_{ij} ← Σ_{t=1}^{T-1} μ_{A_i}(x_t) · μ_{A_j}(x_{t+1})
6:  END FOR
7:  FOR i = 1 to n DO
8:      T_{i,:} ← T_{i,:} / Σ_j T_{ij}            ▷ row normalisation: each row sums to 1
9:  END FOR
10: z_s ← vec(T) ∈ ℝ^{n²}                         ▷ flatten, 9 dims if n=3
11: return z_s
```

---

### Algorithm 3 (Academic) — ANFIS TSK

```
Algorithm 3: ANFIS-TSK Per-Signal Classifier (Hybrid Learning)

Require: Training pairs {z_i, y_i}_{i=1}^N, z ∈ ℝ^d, y ∈ {0,1}
         hyperparameters: R rules, λ ridge, η learning rate, E epochs
Ensure: Model parameters {c_r, σ_r, θ_r}_{r=1}^R

1:  {c_r}_{r=1}^R ← KMeans(z, R).cluster_centers_
2:  FOR epoch = 1 to E DO
    ── Forward (WLS for consequent) ──
3:      w̄_r ← softmax(Σ_j log μ_{r,j}(z_j))      ▷ Layers 1–3 (Gaussian/Gbell)
4:      Φ_{i,r} ← w̄_r(z_i) · [z_i; 1]            ▷ Layer 4 design matrix
5:      θ ← (ΦᵀΦ + λI)⁻¹ Φᵀ y                    ▷ Ridge LS
6:      ŷ ← Φ θ                                    ▷ Layer 5
    ── Backward (GD for premise) ──
7:      Compute ∂L/∂c_r, ∂L/∂σ_r via chain rule through softmax
8:      c_r ← c_r − η ∂L/∂c_r  ;  σ_r ← max(σ_r − η ∂L/∂σ_r, ε)
9:  END FOR
10: p_s(z) ← σ(ŷ) = 1/(1+e^{−ŷ})                 ▷ sigmoid → P(TASK)
11: return {c_r, σ_r, θ_r}
```

---

### Algorithm 4 (Academic) — Soft-Evidence BN Fusion

```
Algorithm 4: Soft-Evidence Naive Bayes Fusion

Require: Training evidence P_tr ∈ [0,1]^{N×4}, labels y ∈ {0,1}^N
         Test evidence p = [p_T, p_H, p_G, p_R] ∈ [0,1]^4
         n_bins B, Laplace α, decision threshold τ
Ensure: Posterior P̂(TASK|p)

1:  Fit quantile bin edges on P_tr: edges_s = quantile(P_tr[:,s], linspace(0,1,B+1)[1:-1])
2:  π_c ← (count(y=c) + α) / (N + 2α),  c ∈ {0,1}
3:  FOR s = 0 to 3 DO
4:      CPT_s[c,b] ← (count(y=c, bin(p_s)=b) + α) / (N_c + B·α)
5:  END FOR
6:  Compute bin midpoints mid_{s,b} and Gaussian kernel σ_s
7:  FOR c ∈ {0, 1} DO
8:      ℓ_c ← log π_c + Σ_s log( Σ_b  Gauss(p_s; mid_{s,b}, σ_s) · CPT_s[c,b] )
9:  END FOR
10: P̂(TASK|p) ← exp(ℓ_1) / (exp(ℓ_0) + exp(ℓ_1))
11: return P̂(TASK|p)
```

---

> **Lưu ý cho khóa luận:**
> - Algorithms 1–4 mô tả full pipeline (FTS–ANFIS–BN). Algorithm 5 mô tả FTS–ANFIS–Mean, Algorithm 6 mô tả FTS Summary–BN.
> - Khi viết Section "Experimental Setup", làm rõ: train = 18 subjects, test = 5 subjects,
>   threshold selection dùng balanced accuracy trên OOF (không dùng test set).
> - Phân biệt rõ "cross-validation accuracy" (OOF, dùng chọn threshold) và
>   "test accuracy" (held-out, báo cáo cuối cùng).
