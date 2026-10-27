"""Step 1: download the raw World Bank indicator files into data/raw/.

Usage:  python -m src.download_data
"""
import io
import urllib.request

import pandas as pd

from src.config import INDICATORS, MIRROR_URL, RAW_DIR


def download_indicator(name: str, code: str) -> pd.DataFrame:
    url = MIRROR_URL.format(code=code)
    with urllib.request.urlopen(url, timeout=60) as resp:
        df = pd.read_csv(io.BytesIO(resp.read()))
    expected = {"Country Name", "Country Code", "Year", "Value"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{code}: unexpected file layout, missing columns {missing}")
    return df


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, code in INDICATORS.items():
        df = download_indicator(name, code)
        out = RAW_DIR / f"{name}.csv"
        df.to_csv(out, index=False)
        print(f"{name:<14} {code:<20} {len(df):>6} rows  years {df.Year.min()}-{df.Year.max()}")
    print(f"Saved to {RAW_DIR}")


if __name__ == "__main__":
    main()
