"""Step 2: turn raw indicator files into one country-year table with features and a target.

Each row = one country in one year t.
  * Features use only information available at the end of year t.
  * The target is inflation in year t+1 (next year).

Usage:  python -m src.features
"""
import numpy as np
import pandas as pd

from src.config import AGGREGATE_CODES, INDICATORS, PROCESSED_DIR, RAW_DIR, START_YEAR


def slog(x):
    """Signed log: sign(x) * log(1 + |x|).

    Inflation ranges from about -5% to tens of thousands of percent (Venezuela, Zimbabwe).
    On a raw scale those few hyperinflation years dominate everything. The signed log
    squeezes the extremes (1% -> 0.69, 10% -> 2.4, 1000% -> 6.9) while still handling
    negative values (deflation), which a plain log cannot.
    """
    return np.sign(x) * np.log1p(np.abs(x))


def inv_slog(z):
    """Undo slog, to turn model outputs back into percent."""
    return np.sign(z) * np.expm1(np.abs(z))


def load_raw() -> pd.DataFrame:
    """Read every raw file and join them into one wide table: country, year, indicator columns."""
    frames = []
    for name in INDICATORS:
        df = pd.read_csv(RAW_DIR / f"{name}.csv")
        df = df.rename(columns={"Country Name": "country", "Country Code": "iso3",
                                "Year": "year", "Value": name})
        frames.append(df.set_index(["iso3", "year"])[[name, "country"]])

    names = pd.concat([f["country"] for f in frames]).groupby(level=0).last()
    wide = pd.concat([f.drop(columns="country") for f in frames], axis=1)
    wide = wide.reset_index()
    wide = wide[~wide["iso3"].isin(AGGREGATE_CODES)]
    wide = wide[wide["year"] >= START_YEAR - 3]   # keep 3 extra years so early lags exist
    wide["country"] = wide["iso3"].map(names)
    return wide


def complete_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Give every country a row for every year, so 'previous row' always means 'previous year'.

    Without this, a country missing 2003 would have its 2004 row compared to 2002
    when computing lags, silently mixing up time steps.
    """
    years = range(df["year"].min(), df["year"].max() + 1)
    grid = pd.MultiIndex.from_product([df["iso3"].unique(), years], names=["iso3", "year"])
    out = df.set_index(["iso3", "year"]).reindex(grid).reset_index()
    out["country"] = out.groupby("iso3")["country"].transform(lambda s: s.ffill().bfill())
    return out.sort_values(["iso3", "year"]).reset_index(drop=True)


FEATURES = [
    # inflation's own history (strongest signal: inflation is persistent)
    "infl_t", "infl_lag1", "infl_lag2", "infl_mean3", "infl_change",
    # other price measure
    "deflator_t",
    # money and currency (classic drivers of high inflation)
    "money_growth_t", "money_growth_lag1", "fx_depr_t", "fx_depr_lag1",
    # real economy
    "gdp_growth_t", "unemployment_t", "unemployment_change",
    # interest rates
    "lending_rate_t", "real_rate_t",
    # openness to imported price shocks
    "imports_gdp_t",
    # the global cycle (e.g., the 2021-22 worldwide surge)
    "global_infl_t", "global_infl_change",
]

INFLATION_HISTORY_FEATURES = FEATURES[:5]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = complete_grid(df)
    g = df.groupby("iso3")

    infl = slog(df["inflation"])
    df["infl_t"] = infl
    df["infl_lag1"] = infl.groupby(df["iso3"]).shift(1)
    df["infl_lag2"] = infl.groupby(df["iso3"]).shift(2)
    df["infl_mean3"] = df[["infl_t", "infl_lag1", "infl_lag2"]].mean(axis=1, skipna=False)
    df["infl_change"] = df["infl_t"] - df["infl_lag1"]

    df["deflator_t"] = slog(df["deflator"])

    money = slog(df["money_growth"])
    df["money_growth_t"] = money
    df["money_growth_lag1"] = money.groupby(df["iso3"]).shift(1)

    # Currency depreciation: % change in local-currency units per US dollar.
    fx_prev = g["fx_rate"].shift(1)
    depr = (df["fx_rate"] / fx_prev - 1) * 100
    df["fx_depr_t"] = slog(depr)
    df["fx_depr_lag1"] = df["fx_depr_t"].groupby(df["iso3"]).shift(1)

    df["gdp_growth_t"] = df["gdp_growth"]
    df["unemployment_t"] = df["unemployment"]
    df["unemployment_change"] = df["unemployment"] - g["unemployment"].shift(1)
    df["lending_rate_t"] = slog(df["lending_rate"])
    df["real_rate_t"] = slog(df["real_rate"])
    df["imports_gdp_t"] = df["imports_gdp"]

    # Median inflation across all countries in year t. Uses only year-t data,
    # so it is known at forecast time (no peeking at the future).
    global_med = df.groupby("year")["inflation"].median()
    df["global_infl_t"] = slog(df["year"].map(global_med))
    df["global_infl_change"] = df["global_infl_t"] - slog(df["year"].map(global_med.shift(1)))

    # Target: next year's inflation, on the slog scale.
    df["target_year"] = df["year"] + 1
    df["inflation_next"] = g["inflation"].shift(-1)
    df["y"] = slog(df["inflation_next"])
    return df


def build_dataset() -> pd.DataFrame:
    df = build_features(load_raw())
    df = df[df["year"] >= START_YEAR]
    # A row is usable only if we know this year's inflation (the minimum to forecast from).
    df = df[df["inflation"].notna()]
    cols = ["iso3", "country", "year", "target_year", "inflation", "inflation_next", "y"] + FEATURES
    return df[cols].reset_index(drop=True)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data = build_dataset()
    out = PROCESSED_DIR / "panel.csv"
    data.to_csv(out, index=False)
    labeled = data["y"].notna()
    print(f"{len(data)} country-years, {data.iso3.nunique()} countries, "
          f"years {data.year.min()}-{data.year.max()}")
    print(f"{labeled.sum()} rows have a known next-year target; "
          f"{(~labeled).sum()} are latest-year rows used only for the live forecast")
    print("Share of missing values per feature:")
    print(data[FEATURES].isna().mean().round(2).to_string())
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
