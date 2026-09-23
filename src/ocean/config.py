from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OceanDataset:
    key: str
    dataset_id: str
    variables: tuple[str, ...]
    surface_only: bool
    resolution: str
    description: str


# V3는 조업데이터 기간(2023~2026)을 한 소스 체계로 맞추기 위해
# Global Analysis & Forecast 일자료를 우선 사용한다.
# Dataset ID는 Copernicus Marine 2026-09 공개 카탈로그 기준.
DATASETS: dict[str, OceanDataset] = {
    "temperature": OceanDataset(
        key="temperature",
        dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
        variables=("thetao",),
        surface_only=True,
        resolution="0.083° daily",
        description="Sea water potential temperature",
    ),
    "current": OceanDataset(
        key="current",
        dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
        variables=("uo", "vo"),
        surface_only=True,
        resolution="0.083° daily",
        description="Eastward / northward sea water velocity",
    ),
    "sea_level": OceanDataset(
        key="sea_level",
        dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        variables=("zos",),
        surface_only=False,
        resolution="0.083° daily",
        description="Sea surface height above geoid",
    ),
    "chlorophyll": OceanDataset(
        key="chlorophyll",
        dataset_id="cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m",
        variables=("chl",),
        surface_only=True,
        resolution="0.25° daily",
        description="Total chlorophyll concentration",
    ),
}


OCEAN_FEATURE_COLUMNS = [
    "ocean_sst",
    "ocean_current_u",
    "ocean_current_v",
    "ocean_current_speed",
    "ocean_current_dir_deg",
    "ocean_ssh",
    "ocean_chl",
    "ocean_sst_gradient_c_per_100km",
]


def dataset_catalog_rows() -> list[dict]:
    return [
        {
            "key": item.key,
            "dataset_id": item.dataset_id,
            "variables": ", ".join(item.variables),
            "resolution": item.resolution,
            "description": item.description,
        }
        for item in DATASETS.values()
    ]
