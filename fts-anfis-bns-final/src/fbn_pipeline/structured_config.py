from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


VALID_FTS_MF = frozenset({"triangular", "gaussian", "trapezoid", "gbell"})
VALID_FTS_NORM = frozenset({"total_sum", "row"})
VALID_ANFIS_MF = frozenset({"gaussian", "gbell"})
VALID_CALIBRATION = frozenset({"none", "platt", "isotonic"})
VALID_OUTER_PROTOCOLS = frozenset({"loso"})
VALID_SELECTION_METRICS = frozenset({"balanced_accuracy", "auc", "roc_auc"})
VALID_THRESHOLD_METRICS = frozenset({"accuracy", "balanced_accuracy", "f1", "macro_f1"})


@dataclass(slots=True)
class FTSConfig:
    n_sets: int
    mf_type: str
    mf_sigma_scale: float
    gbell_b: float
    norm_mode: str
    robust: bool
    trap_plateau_scale: float = 0.15
    trap_width_scale: float = 1.0
    grid_margin_scale: float = 0.05

    def __post_init__(self) -> None:
        if self.n_sets < 2:
            raise ValueError("FTS n_sets must be >= 2")
        if self.mf_type not in VALID_FTS_MF:
            raise ValueError(f"Unsupported FTS membership function: {self.mf_type!r}")
        if self.norm_mode not in VALID_FTS_NORM:
            raise ValueError(f"Unsupported FTS norm mode: {self.norm_mode!r}")


@dataclass(slots=True)
class ANFISConfig:
    n_rules: int
    sigma_scale: float
    ridge_lambda: float
    n_epochs: int
    learning_rate: float
    random_state: int
    mf_type: str = "gaussian"
    gbell_b: float = 2.0

    def __post_init__(self) -> None:
        if self.n_rules < 1:
            raise ValueError("ANFIS n_rules must be >= 1")
        if self.n_epochs < 1:
            raise ValueError("ANFIS n_epochs must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("ANFIS learning_rate must be > 0")
        if self.random_state is None:
            raise ValueError("ANFIS random_state is required")
        if self.mf_type not in VALID_ANFIS_MF:
            raise ValueError(f"Unsupported ANFIS membership function: {self.mf_type!r}")


@dataclass(slots=True)
class BNConfig:
    n_bins: int
    alpha: float
    soft_evidence: bool
    reliability_weighting: bool
    calibration: str

    def __post_init__(self) -> None:
        if self.n_bins < 2:
            raise ValueError("BN n_bins must be >= 2")
        if self.alpha < 0:
            raise ValueError("BN alpha must be >= 0")
        if self.calibration not in VALID_CALIBRATION:
            raise ValueError(f"Unsupported BN calibration: {self.calibration!r}")
        if self.calibration != "none" and not self.soft_evidence and self.reliability_weighting:
            raise ValueError("Calibration with reliability weighting must use compatible evidence settings.")


@dataclass(slots=True)
class EvaluationConfig:
    outer_protocol: str
    inner_folds: int
    selection_metric: str
    threshold_metric: str
    threshold_grid: tuple[float, ...]
    seed: int
    deterministic_threads: int
    output_directory: str

    def __post_init__(self) -> None:
        if self.outer_protocol not in VALID_OUTER_PROTOCOLS:
            raise ValueError(f"Unsupported outer_protocol: {self.outer_protocol!r}")
        if self.inner_folds < 2:
            raise ValueError("inner_folds must be >= 2")
        if self.selection_metric not in VALID_SELECTION_METRICS:
            raise ValueError(f"Unsupported selection_metric: {self.selection_metric!r}")
        if self.threshold_metric not in VALID_THRESHOLD_METRICS:
            raise ValueError(f"Unsupported threshold_metric: {self.threshold_metric!r}")
        if self.seed is None:
            raise ValueError("seed is required")
        if self.deterministic_threads < 1:
            raise ValueError("deterministic_threads must be >= 1")
        if not self.output_directory:
            raise ValueError("output_directory is required")
        self._validate_threshold_grid(self.threshold_grid)

    @staticmethod
    def _validate_threshold_grid(grid: tuple[float, ...]) -> None:
        if len(grid) < 2:
            raise ValueError("threshold_grid must contain at least two values")
        prev = None
        for value in grid:
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError("threshold_grid values must be within [0, 1]")
            if prev is not None and float(value) <= prev:
                raise ValueError("threshold_grid must be strictly increasing")
            prev = float(value)

    def validate_available_groups(self, groups: list[str] | tuple[str, ...]) -> None:
        unique_groups = list(dict.fromkeys(groups))
        if len(unique_groups) < 2:
            raise ValueError("At least two subject groups are required")
        if self.inner_folds > len(unique_groups):
            raise ValueError("inner_folds cannot exceed the available outer-training groups")

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["output_directory"] = str(Path(self.output_directory))
        return data


@dataclass(slots=True)
class NestedEvaluationConfig:
    fts: FTSConfig
    anfis: ANFISConfig
    bn: BNConfig
    evaluation: EvaluationConfig
    candidate_ids: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        validate_candidate_ids(self.candidate_ids)

    def as_dict(self) -> dict[str, object]:
        return {
            "fts": asdict(self.fts),
            "anfis": asdict(self.anfis),
            "bn": asdict(self.bn),
            "evaluation": self.evaluation.as_dict(),
            "candidate_ids": list(self.candidate_ids),
        }


def validate_candidate_ids(candidate_ids: tuple[str, ...] | list[str]) -> None:
    seen: set[str] = set()
    for candidate_id in candidate_ids:
        if candidate_id in seen:
            raise ValueError(f"Duplicate candidate ID: {candidate_id}")
        seen.add(candidate_id)
