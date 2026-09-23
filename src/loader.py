from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pandas as pd

from .config import RAW_COLUMNS


def load_fishing_excel(source: str | Path | BinaryIO) -> pd.DataFrame:
    """사조 선망 조업보고 엑셀(A:X)을 읽어 평탄한 DataFrame으로 반환한다."""
    df = pd.read_excel(
        source,
        sheet_name="조업정보",
        header=None,
        skiprows=3,
        usecols="A:X",
    )
    if df.shape[1] != len(RAW_COLUMNS):
        raise ValueError(
            f"예상 컬럼 수는 {len(RAW_COLUMNS)}개인데 {df.shape[1]}개를 읽었습니다. "
            "원본 엑셀 형식이 변경되었는지 확인하세요."
        )
    df.columns = RAW_COLUMNS
    return df.dropna(how="all").reset_index(drop=True)
