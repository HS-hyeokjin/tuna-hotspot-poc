from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .config import DATASETS, OCEAN_FEATURE_COLUMNS


ProgressCallback = Callable[[int, int, str], None]


def _cache_matches(
    cached: pd.DataFrame,
    requested: pd.DataFrame,
) -> bool:
    keys = [
        "source_row_id",
        "date",
        "lat",
        "lon",
    ]
    if any(
        col not in cached.columns
        for col in keys
    ):
        return False

    left = (
        cached[keys]
        .copy()
        .sort_values("source_row_id")
        .reset_index(drop=True)
    )
    right = (
        requested[keys]
        .copy()
        .sort_values("source_row_id")
        .reset_index(drop=True)
    )

    if len(left) != len(right):
        return False

    left["date"] = pd.to_datetime(
        left["date"],
        errors="coerce",
    )
    right["date"] = pd.to_datetime(
        right["date"],
        errors="coerce",
    )

    return (
        left["source_row_id"].equals(
            right["source_row_id"]
        )
        and left["date"].equals(
            right["date"]
        )
        and np.allclose(
            left["lat"].to_numpy(float),
            right["lat"].to_numpy(float),
            equal_nan=True,
        )
        and np.allclose(
            left["lon"].to_numpy(float),
            right["lon"].to_numpy(float),
            equal_nan=True,
        )
    )


def _import_copernicusmarine():
    try:
        import copernicusmarine
    except ImportError as exc:
        raise RuntimeError(
            "copernicusmarine 패키지가 없습니다. "
            "pip install -r requirements.txt 를 실행하세요."
        ) from exc
    return copernicusmarine


def _coord_name(
    ds: xr.Dataset | xr.DataArray,
    candidates: tuple[str, ...],
) -> str:
    for candidate in candidates:
        if candidate in ds.coords or candidate in ds.dims:
            return candidate
    raise KeyError(
        f"좌표를 찾지 못했습니다. 후보={candidates}, "
        f"coords={list(ds.coords)}"
    )


def _normalise_points(
    fishing_df: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "source_row_id",
        "date",
        "lat",
        "lon",
    }
    missing = required - set(fishing_df.columns)
    if missing:
        raise ValueError(
            "외부 데이터 결합에 필요한 컬럼이 없습니다: "
            + ", ".join(sorted(missing))
        )

    points = fishing_df[
        ["source_row_id", "date", "lat", "lon"]
    ].copy()
    points["date"] = pd.to_datetime(
        points["date"],
        errors="coerce",
    )
    points["lat"] = pd.to_numeric(
        points["lat"],
        errors="coerce",
    )
    points["lon"] = pd.to_numeric(
        points["lon"],
        errors="coerce",
    )
    points = points.dropna(
        subset=["date", "lat", "lon"]
    )
    points = points[
        points["lat"].between(-90, 90)
        & points["lon"].between(-180, 180)
    ].copy()

    # 월 + 20도 공간 블록. 날짜변경선을 가로질러 거대한 bbox가
    # 생성되는 것을 방지하고 원격 요청 크기도 줄인다.
    points["month_key"] = (
        points["date"]
        .dt.to_period("M")
        .astype(str)
    )
    points["lon_band"] = np.floor(
        (points["lon"] + 180.0) / 20.0
    ).astype(int)
    points["lat_band"] = np.floor(
        (points["lat"] + 90.0) / 20.0
    ).astype(int)

    return points.reset_index(drop=True)


def _open_subset(
    dataset_key: str,
    points: pd.DataFrame,
    margin_deg: float = 0.6,
) -> xr.Dataset:
    cm = _import_copernicusmarine()
    config = DATASETS[dataset_key]

    start = points["date"].min().normalize()
    end = (
        points["date"].max().normalize()
        + pd.Timedelta(hours=23, minutes=59)
    )

    min_lon = max(
        -180.0,
        float(points["lon"].min()) - margin_deg,
    )
    max_lon = min(
        179.92,
        float(points["lon"].max()) + margin_deg,
    )
    min_lat = max(
        -80.0,
        float(points["lat"].min()) - margin_deg,
    )
    max_lat = min(
        90.0,
        float(points["lat"].max()) + margin_deg,
    )

    kwargs = dict(
        dataset_id=config.dataset_id,
        variables=list(config.variables),
        minimum_longitude=min_lon,
        maximum_longitude=max_lon,
        minimum_latitude=min_lat,
        maximum_latitude=max_lat,
        start_datetime=start.isoformat(),
        end_datetime=end.isoformat(),
    )

    if config.surface_only:
        kwargs.update(
            minimum_depth=0.0,
            maximum_depth=1.0,
        )

    return cm.open_dataset(**kwargs)


def _select_surface(
    array: xr.DataArray,
) -> xr.DataArray:
    for depth_name in (
        "depth",
        "elevation",
        "deptht",
        "depthu",
        "depthv",
    ):
        if (
            depth_name in array.coords
            or depth_name in array.dims
        ):
            try:
                return array.sel(
                    {depth_name: 0.0},
                    method="nearest",
                )
            except Exception:
                return array.isel(
                    {depth_name: 0}
                )
    return array


def _point_values(
    ds: xr.Dataset,
    variable: str,
    points: pd.DataFrame,
    lat_values: np.ndarray | None = None,
    lon_values: np.ndarray | None = None,
) -> np.ndarray:
    if variable not in ds:
        raise KeyError(
            f"{variable} 변수가 dataset에 없습니다: "
            f"{list(ds.data_vars)}"
        )

    arr = _select_surface(ds[variable])
    lat_name = _coord_name(
        arr,
        ("latitude", "lat", "nav_lat"),
    )
    lon_name = _coord_name(
        arr,
        ("longitude", "lon", "nav_lon"),
    )
    time_name = _coord_name(
        arr,
        ("time",),
    )

    lats = (
        points["lat"].to_numpy(dtype=float)
        if lat_values is None
        else np.asarray(lat_values, dtype=float)
    )
    lons = (
        points["lon"].to_numpy(dtype=float)
        if lon_values is None
        else np.asarray(lon_values, dtype=float)
    )

    idx = xr.DataArray(
        np.arange(len(points)),
        dims="point",
    )
    selected = arr.sel(
        {
            time_name: xr.DataArray(
                points["date"]
                .to_numpy(dtype="datetime64[ns]"),
                dims="point",
            ),
            lat_name: xr.DataArray(
                lats,
                dims="point",
            ),
            lon_name: xr.DataArray(
                lons,
                dims="point",
            ),
        },
        method="nearest",
    )

    values = np.asarray(
        selected.compute().values
    )
    return values.reshape(-1).astype(float)


def _sst_features(
    ds: xr.Dataset,
    points: pd.DataFrame,
    neighbour_deg: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    center = _point_values(
        ds,
        "thetao",
        points,
    )

    gradient = np.full(
        len(points),
        np.nan,
        dtype=float,
    )

    safe = (
        points["lon"].abs() <= 179.5
    ) & (
        points["lat"].abs() <= 89.5
    )
    if not safe.any():
        return center, gradient

    subset = points.loc[safe].copy()
    lat = subset["lat"].to_numpy(float)
    lon = subset["lon"].to_numpy(float)

    north = _point_values(
        ds,
        "thetao",
        subset,
        lat_values=lat + neighbour_deg,
        lon_values=lon,
    )
    south = _point_values(
        ds,
        "thetao",
        subset,
        lat_values=lat - neighbour_deg,
        lon_values=lon,
    )
    east = _point_values(
        ds,
        "thetao",
        subset,
        lat_values=lat,
        lon_values=lon + neighbour_deg,
    )
    west = _point_values(
        ds,
        "thetao",
        subset,
        lat_values=lat,
        lon_values=lon - neighbour_deg,
    )

    dy_km = 2.0 * neighbour_deg * 111.32
    dx_km = (
        2.0
        * neighbour_deg
        * 111.32
        * np.cos(np.deg2rad(lat))
    )
    dx_km = np.where(
        np.abs(dx_km) < 1e-6,
        np.nan,
        dx_km,
    )

    dtdy = (north - south) / dy_km
    dtdx = (east - west) / dx_km

    local_gradient = (
        np.sqrt(
            np.square(dtdy)
            + np.square(dtdx)
        )
        * 100.0
    )
    gradient[
        np.flatnonzero(safe.to_numpy())
    ] = local_gradient

    return center, gradient


def extract_chunk(
    points: pd.DataFrame,
) -> pd.DataFrame:
    out = points[
        ["source_row_id", "date", "lat", "lon"]
    ].copy()

    for col in OCEAN_FEATURE_COLUMNS:
        out[col] = np.nan

    errors: list[str] = []

    try:
        ds_temp = _open_subset(
            "temperature",
            points,
        )
        try:
            sst, gradient = _sst_features(
                ds_temp,
                points,
            )
            out["ocean_sst"] = sst
            out[
                "ocean_sst_gradient_c_per_100km"
            ] = gradient
        finally:
            ds_temp.close()
    except Exception as exc:
        errors.append(
            f"temperature={type(exc).__name__}: {exc}"
        )

    try:
        ds_cur = _open_subset(
            "current",
            points,
        )
        try:
            u = _point_values(
                ds_cur,
                "uo",
                points,
            )
            v = _point_values(
                ds_cur,
                "vo",
                points,
            )
            out["ocean_current_u"] = u
            out["ocean_current_v"] = v
            speed = np.sqrt(
                np.square(u)
                + np.square(v)
            )
            out[
                "ocean_current_speed"
            ] = speed
            out[
                "ocean_current_dir_deg"
            ] = (
                np.degrees(
                    np.arctan2(u, v)
                )
                + 360.0
            ) % 360.0
        finally:
            ds_cur.close()
    except Exception as exc:
        errors.append(
            f"current={type(exc).__name__}: {exc}"
        )

    try:
        ds_ssh = _open_subset(
            "sea_level",
            points,
        )
        try:
            out["ocean_ssh"] = (
                _point_values(
                    ds_ssh,
                    "zos",
                    points,
                )
            )
        finally:
            ds_ssh.close()
    except Exception as exc:
        errors.append(
            f"sea_level={type(exc).__name__}: {exc}"
        )

    try:
        ds_chl = _open_subset(
            "chlorophyll",
            points,
        )
        try:
            out["ocean_chl"] = (
                _point_values(
                    ds_chl,
                    "chl",
                    points,
                )
            )
        finally:
            ds_chl.close()
    except Exception as exc:
        errors.append(
            f"chlorophyll={type(exc).__name__}: {exc}"
        )

    out["ocean_error"] = (
        " | ".join(errors)
        if errors
        else ""
    )
    out["ocean_complete"] = (
        out[OCEAN_FEATURE_COLUMNS]
        .notna()
        .sum(axis=1)
        >= 5
    )
    return out


def build_ocean_feature_store(
    fishing_df: pd.DataFrame,
    output_path: str | Path,
    cache_dir: str | Path,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> pd.DataFrame:
    points = _normalise_points(
        fishing_df
    )
    if points.empty:
        raise ValueError(
            "외부 해양 데이터를 조회할 유효 날짜/좌표가 없습니다."
        )

    output_path = Path(output_path)
    cache_dir = Path(cache_dir)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    groups = list(
        points.groupby(
            [
                "month_key",
                "lon_band",
                "lat_band",
            ],
            sort=True,
        )
    )
    parts: list[pd.DataFrame] = []
    total = len(groups)

    for index, (group_key, chunk) in enumerate(
        groups,
        start=1,
    ):
        month_key, lon_band, lat_band = group_key
        cache_path = cache_dir / (
            f"{month_key}_lon{lon_band:02d}_lat{lat_band:02d}.parquet"
        )

        label = (
            f"{month_key} / lon-band {lon_band} / "
            f"lat-band {lat_band} / {len(chunk):,} rows"
        )
        if progress:
            progress(index, total, label)

        use_cache = False
        part = None

        if cache_path.exists() and not force:
            cached = pd.read_parquet(
                cache_path
            )
            if _cache_matches(
                cached,
                chunk,
            ):
                part = cached
                use_cache = True

        if not use_cache:
            part = extract_chunk(
                chunk.reset_index(drop=True)
            )
            part.to_parquet(
                cache_path,
                index=False,
            )

        parts.append(part)

    features = pd.concat(
        parts,
        ignore_index=True,
    )
    features = (
        features.sort_values(
            "source_row_id"
        )
        .drop_duplicates(
            subset=["source_row_id"],
            keep="last",
        )
        .reset_index(drop=True)
    )
    features.to_parquet(
        output_path,
        index=False,
    )
    return features


def feature_coverage(
    features: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    total = max(len(features), 1)
    for col in OCEAN_FEATURE_COLUMNS:
        if col not in features:
            continue
        count = int(
            features[col].notna().sum()
        )
        rows.append(
            {
                "feature": col,
                "non_null": count,
                "coverage_pct": (
                    count / total * 100.0
                ),
            }
        )
    return pd.DataFrame(rows)


def merge_ocean_features(
    fishing_df: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    if "source_row_id" not in fishing_df:
        raise ValueError(
            "조업데이터에 source_row_id가 없습니다."
        )
    if "source_row_id" not in features:
        raise ValueError(
            "외부 특징 파일에 source_row_id가 없습니다."
        )

    ocean_cols = [
        "source_row_id",
        *[
            col
            for col in OCEAN_FEATURE_COLUMNS
            if col in features.columns
        ],
    ]
    optional = [
        col
        for col in [
            "ocean_error",
            "ocean_complete",
        ]
        if col in features.columns
    ]

    return fishing_df.merge(
        features[
            ocean_cols + optional
        ].drop_duplicates(
            subset=["source_row_id"]
        ),
        on="source_row_id",
        how="left",
        validate="one_to_one",
    )
