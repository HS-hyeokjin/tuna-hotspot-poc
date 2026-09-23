from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "선망_조업보고 데이터.xlsx"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

RAW_COLUMNS = [
    "date_raw",
    "vessel",
    "captain",
    "fishing_ground",
    "lat_raw",
    "lon_raw",
    "school_fish",
    "log_fish",
    "pa",
    "prev_cum",
    "daily",
    "cum",
    "sj_75",
    "sj_4",
    "sj_3",
    "sj_m3",
    "sh_ps",
    "yf_20",
    "yf_75",
    "yf_ps",
    "yf_gg",
    "be_20",
    "water_temp",
    "current",
]

CATCH_COLUMNS = [
    "sj_75", "sj_4", "sj_3", "sj_m3", "sh_ps",
    "yf_20", "yf_75", "yf_ps", "yf_gg", "be_20",
]
SJ_COLUMNS = ["sj_75", "sj_4", "sj_3", "sj_m3"]
YF_COLUMNS = ["yf_20", "yf_75", "yf_ps", "yf_gg"]
BE_COLUMNS = ["be_20"]
METHOD_COLUMNS = ["school_fish", "log_fish", "pa"]
