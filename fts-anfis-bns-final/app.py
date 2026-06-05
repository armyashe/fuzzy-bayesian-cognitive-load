from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from streamlit_app.sidebar import render_sidebar
from streamlit_app.tabs.bn_rules_tab import render_bn_rules_tab
from streamlit_app.tabs.overview_tab import render_overview_tab
from streamlit_app.tabs.results_tab import render_results_tab
from streamlit_app.tabs.sample_explanation_tab import render_sample_explanation_tab


st.set_page_config(page_title="Fuzzy–ANFIS–BN Dashboard", layout="wide")
st.title("Fuzzy–ANFIS–Bayesian Network Dashboard")

render_sidebar(ROOT)

if "results" not in st.session_state:
    st.info("Select an experiment group and run the pipeline from the sidebar.")
    st.stop()

results = st.session_state["results"]

tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Results", "Sample Explanation", "BN / Rules"]
)

with tab1:
    render_overview_tab(results)

with tab2:
    render_results_tab(results)

with tab3:
    render_sample_explanation_tab(results)

with tab4:
    render_bn_rules_tab(results, root=ROOT)
