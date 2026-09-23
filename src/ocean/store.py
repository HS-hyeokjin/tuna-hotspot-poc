from __future__ import annotations

from pathlib import Path

import pandas as pd

from .extractor import (
    feature_coverage,
    merge_ocean_features,
)


DEFAULT_FEATURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "external"
    / "processed"
    / "fishing_ocean_features.parquet"
)

DEFAULT_CACHE_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "external"
    / "cache"
)


def load_feature_store(
    path: str | Path = DEFAULT_FEATURE_PATH,
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def attach_feature_store(
    fishing_df: pd.DataFrame,
    path: str | Path = DEFAULT_FEATURE_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = load_feature_store(path)
    merged = merge_ocean_features(
        fishing_df,
        features,
    )
    coverage = feature_coverage(
        features
    )
    return merged, coverage
