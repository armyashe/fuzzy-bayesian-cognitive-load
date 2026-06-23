from __future__ import annotations

import os
import platform
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

import numpy as np


THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
)


@dataclass(slots=True)
class DeterminismMetadata:
    seed: int
    deterministic_threads: int
    timestamp_utc: str
    python_version: str
    platform: str
    package_versions: dict[str, str]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def configure_determinism(
    seed: int,
    *,
    deterministic_threads: int = 1,
    respect_existing_env: bool = True,
) -> DeterminismMetadata:
    if seed is None:
        raise ValueError("seed is required")
    if deterministic_threads < 1:
        raise ValueError("deterministic_threads must be >= 1")

    os.environ["PYTHONHASHSEED"] = str(seed)
    for name in THREAD_ENV_VARS:
        if respect_existing_env:
            os.environ.setdefault(name, str(deterministic_threads))
        else:
            os.environ[name] = str(deterministic_threads)

    random.seed(seed)
    np.random.seed(seed)

    return DeterminismMetadata(
        seed=seed,
        deterministic_threads=deterministic_threads,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        python_version=sys.version.replace("\n", " "),
        platform=platform.platform(),
        package_versions=_package_versions(["numpy", "pandas", "scikit-learn", "scipy", "threadpoolctl"]),
    )


def _package_versions(names: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not-installed"
    return versions
