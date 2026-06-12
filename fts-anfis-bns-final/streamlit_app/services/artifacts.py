import json
from pathlib import Path

import streamlit as st


def load_json_artifact(exp_name: str, filename: str, artifacts_root: Path | str = "artifacts") -> dict | None:
    artifact_root = Path(artifacts_root) if isinstance(artifacts_root, str) else artifacts_root
    artifact_path = artifact_root / exp_name / filename
    if artifact_path.exists():
        try:
            with open(artifact_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Failed to load {filename}: {e}")
            return None
    return None


def load_bn_artifact_json(exp_name: str, artifacts_root: Path | str = "artifacts") -> dict | None:
    """Load BN details from JSON artifact if it exists."""
    return load_json_artifact(exp_name, "bn_details.json", artifacts_root)


def load_anfis_rules_json(exp_name: str, artifacts_root: Path | str = "artifacts") -> dict | None:
    """Load ANFIS rules from JSON artifact if it exists."""
    return load_json_artifact(exp_name, "anfis_rules.json", artifacts_root)


def get_bn_graph_image_path(exp_name: str, artifacts_root: Path | str = "artifacts") -> Path | None:
    """Get BN graph image path if it exists."""
    artifact_root = Path(artifacts_root) if isinstance(artifacts_root, str) else artifacts_root
    img_path = artifact_root / exp_name / "bn_graph.png"
    if img_path.exists():
        return img_path
    return None
