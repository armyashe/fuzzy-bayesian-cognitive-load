import streamlit as st

from fbn_pipeline.config import ExperimentConfig


def render_method_diagram(cfg: ExperimentConfig, timesteps: int):
    bn_mode = "Soft Evidence" if cfg.bn_soft else "Hard Evidence"
    html = f"""
    <style>
    .method-wrap {{margin-top: 0.25rem;}}
    .method-title {{font-size: 1.05rem; font-weight: 700; margin-bottom: 0.5rem;}}
    .method-note {{font-size: 0.92rem; color: #666; margin: 0.35rem 0 0.8rem 0;}}
    .method-box {{
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 10px;
        background: rgba(250, 250, 250, 0.85);
        color: #111;
    }}
    .input-box {{background: #eef2f7;}}
    .layer1 {{background: #e7f1dc;}}
    .layer2 {{background: #f8edcf;}}
    .layer3 {{background: #dfeaf8;}}
    .final-box {{background: #f2f2f2;}}
    .box-head {{font-size: 1rem; font-weight: 800; margin-bottom: 8px;}}
    .chip-row {{display: flex; gap: 8px; flex-wrap: wrap; margin-top: 6px;}}
    .chip {{
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(255,255,255,0.9);
        border: 1px solid rgba(49, 51, 63, 0.18);
        font-size: 0.86rem;
        font-weight: 600;
    }}
    .stage-grid {{display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px;}}
    .stage-grid-3 {{display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px;}}
    .mini {{
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(49, 51, 63, 0.18);
        border-radius: 12px;
        padding: 10px 12px;
        min-height: 82px;
    }}
    .mini-head {{font-size: 0.9rem; font-weight: 800; margin-bottom: 4px;}}
    .mini-body {{font-size: 0.83rem; line-height: 1.35;}}
    .arrow {{text-align: center; font-size: 1.4rem; font-weight: 800; color: #555; margin: 0.1rem 0 0.5rem 0;}}
    .caption {{font-size: 0.85rem; color: #444; margin-top: 8px;}}
    </style>

    <div class="method-wrap">
      <div class="method-title">Proposed Method</div>

      <div class="method-box input-box">
        <div class="box-head">Input Data</div>
        <div class="chip-row">
          <span class="chip">Channel 0: Temp x₀(t)</span>
          <span class="chip">Channel 1: HR x₁(t)</span>
          <span class="chip">Channel 2: GSR x₂(t)</span>
          <span class="chip">Channel 3: RR x₃(t)</span>
          <span class="chip">Window T = {timesteps}</span>
        </div>
      </div>

      <div class="arrow">↓</div>

      <div class="method-box layer1">
        <div class="box-head">Layer 1 — FTS Encoding</div>
        <div class="stage-grid">
          <div class="mini">
            <div class="mini-head">Gaussian / Fuzzy Partition</div>
            <div class="mini-body">{cfg.mf_type} membership functions with K = {cfg.n_sets} fuzzy sets per signal.</div>
          </div>
          <div class="mini">
            <div class="mini-head">Membership Sequence</div>
            <div class="mini-body">Convert each raw time point into a normalized fuzzy membership vector.</div>
          </div>
          <div class="mini">
            <div class="mini-head">Transition Extraction</div>
            <div class="mini-body">Compute fuzzy transition counts from consecutive time steps for each channel.</div>
          </div>
          <div class="mini">
            <div class="mini-head">Transition Vector</div>
            <div class="mini-body">Normalize transitions with mode = {cfg.fts_norm} and flatten to a {cfg.n_sets * cfg.n_sets}-D vector per signal.</div>
          </div>
        </div>
        <div class="caption">Each channel is encoded independently, then concatenated into one FTS feature representation.</div>
      </div>

      <div class="arrow">↓</div>

      <div class="method-box layer2">
        <div class="box-head">Layer 2 — Per-signal ANFIS-TSK</div>
        <div class="stage-grid">
          <div class="mini">
            <div class="mini-head">Rule Firing</div>
            <div class="mini-body">Premise layer computes normalized firing strengths for R = {cfg.n_rules} fuzzy rules.</div>
          </div>
          <div class="mini">
            <div class="mini-head">TSK Consequent</div>
            <div class="mini-body">Each rule maps the FTS vector to a local linear consequent.</div>
          </div>
          <div class="mini">
            <div class="mini-head">Aggregate Output</div>
            <div class="mini-body">Weighted sum across active rules yields a raw per-signal score.</div>
          </div>
          <div class="mini">
            <div class="mini-head">Sigmoid Probability</div>
            <div class="mini-body">Transform the raw score into one probability pₛ = P(Y = 1) for each signal.</div>
          </div>
        </div>
        <div class="caption">Outputs a 4-D evidence vector: [p₀, p₁, p₂, p₃] for Temp, HR, GSR, and RR.</div>
      </div>

      <div class="arrow">↓</div>

      <div class="method-box layer3">
        <div class="box-head">Layer 3 — Bayesian Network Fusion</div>
        <div class="stage-grid-3">
          <div class="mini">
            <div class="mini-head">Evidence Modeling</div>
            <div class="mini-body">Discretize or softly distribute each ANFIS probability into B = {cfg.n_bins} evidence bins using {bn_mode.lower()}.</div>
          </div>
          <div class="mini">
            <div class="mini-head">Naive Bayes DAG</div>
            <div class="mini-body">Task node Y points to evidence nodes E<sub>Temp</sub>, E<sub>HR</sub>, E<sub>GSR</sub>, E<sub>RR</sub>.</div>
          </div>
          <div class="mini">
            <div class="mini-head">Posterior Inference</div>
            <div class="mini-body">Combine prior and CPT likelihoods to estimate P(Y = 1 | evidence).</div>
          </div>
        </div>
      </div>

      <div class="arrow">↓</div>

      <div class="method-box final-box">
        <div class="box-head">Final Output</div>
        <div class="chip-row">
          <span class="chip">Posterior: P(Y = 1 | p)</span>
          <span class="chip">Binary decision: Task (1) vs Rest (0)</span>
        </div>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
