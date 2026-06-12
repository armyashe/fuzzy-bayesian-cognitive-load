from __future__ import annotations

from enum import Enum


class Ablation(str, Enum):
    """Canonical names for pipeline variants."""

    FULL = "exp1"
    NO_BN = "exp2"
    NO_ANFIS = "exp3"
    NO_FTS = "exp4"


ABLATION_DESCRIPTIONS: dict[Ablation, str] = {
    Ablation.FULL: "FTS -> ANFIS -> BN",
    Ablation.NO_BN: "FTS -> ANFIS -> Mean/Weighted Mean",
    Ablation.NO_ANFIS: "FTS -> Summary -> BN",
    Ablation.NO_FTS: "Stat Features -> ANFIS -> BN",
}

ABLATION_CHOICES: tuple[str, ...] = tuple(item.value for item in Ablation)


def describe_ablation(value: str) -> str:
    """Return a human-readable pipeline description for an ablation value."""
    return ABLATION_DESCRIPTIONS[Ablation(value)]
