"""Step 3: walk-forward backtest. Pretend it is the end of each year from 2005 to 2024,
train only on what was known then, forecast the next year, and score the forecasts.

Usage:  python -m src.backtest
"""
import numpy as np
import pandas as pd

from src.config import FIRST_TEST_YEAR, PROCESSED_DIR, REPORTS_DIR
from src.features import inv_slog
from src.models import fit_predict, interval_models, model_zoo

INTERVAL_ALPHA = 0.20      # target: 80% of outcomes inside the range
CALIBRATION_YEARS = 5      # widen ranges using the previous 5 years of forecast errors


def walk_forward(data: pd.DataFrame) -> pd.DataFrame:
    """For each target year Y:
        train on rows whose target year is <= Y-1 (outcome already known by end of Y-1),
        test on rows whose target year is Y.
    A random train/test split would let the model learn from 2022 to 'predict' 2021,
    which is impossible in real life. Splitting by time avoids that leak.
    """
    labeled = data[data["y"].notna()]
    last_year = int(labeled["target_year"].max())
    results = []
    for Y in range(FIRST_TEST_YEAR, last_year + 1):
        train = labeled[labeled["target_year"] <= Y - 1]
        test = labeled[labeled["target_year"] == Y]
        out = test[["iso3", "country", "year", "target_year", "inflation", "inflation_next"]].copy()
        for name, model in model_zoo().items():
            out[name] = inv_slog(fit_predict(model, train, test))
        for name, model in interval_models().items():
            out[f"{name}_raw"] = fit_predict(model, train, test)   # kept on slog scale
        out["y"] = test["y"]
        results.append(out)
        print(f"  target year {Y}: trained on {len(train):>5} rows, tested on {len(test):>3}")
    preds = pd.concat(results, ignore_index=True)
    return calibrate_ranges(preds)


def interval_widening(past: pd.DataFrame) -> float:
    """How much to widen raw 10-90% ranges so ~80% of past outcomes would have landed inside.

    For each past forecast, the score is how far the actual value fell outside its range
    (negative if it was inside). The 80th percentile of those scores is the widening needed.
    This is 'conformalized quantile regression' (Romano et al., 2019).
    """
    scores = np.maximum(past["q10_raw"] - past["y"], past["y"] - past["q90_raw"])
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - INTERVAL_ALPHA)) / n)
    return float(np.quantile(scores, level))


def calibrate_ranges(preds: pd.DataFrame) -> pd.DataFrame:
    """Widen each year's ranges using only forecast errors from the previous CALIBRATION_YEARS."""
    preds = preds.copy()
    preds["q10"] = np.nan
    preds["q90"] = np.nan
    for Y in sorted(preds["target_year"].unique()):
        past = preds[(preds["target_year"] < Y) & (preds["target_year"] >= Y - CALIBRATION_YEARS)]
        if past["target_year"].nunique() < CALIBRATION_YEARS:
            continue   # not enough history yet to calibrate; leave this year out of coverage stats
        w = interval_widening(past)
        rows = preds["target_year"] == Y
        preds.loc[rows, "q10"] = inv_slog(preds.loc[rows, "q10_raw"] - w)
        preds.loc[rows, "q90"] = inv_slog(preds.loc[rows, "q90_raw"] + w)
    return preds


def score(actual, pred) -> dict:
    err = np.abs(actual - pred)
    return {
        "Median abs. error (pp)": np.median(err),
        "Within 2 pp": np.mean(err <= 2),
        "Mean abs. error, log scale": np.mean(np.abs(np.sign(actual) * np.log1p(np.abs(actual))
                                                     - np.sign(pred) * np.log1p(np.abs(pred)))),
        "Mean abs. error (pp)": np.mean(err),
    }


def summarize(preds: pd.DataFrame, model_names) -> pd.DataFrame:
    rows = {m: score(preds["inflation_next"], preds[m]) for m in model_names}
    table = pd.DataFrame(rows).T
    base = table.loc["Persistence (baseline)"]
    table["Error cut vs. baseline"] = 1 - table["Mean abs. error, log scale"] / base["Mean abs. error, log scale"]
    return table


def by_regime(preds: pd.DataFrame, model_names) -> pd.DataFrame:
    bins = [-np.inf, 5, 20, np.inf]
    labels = ["Low (<5%)", "Elevated (5-20%)", "High (>20%)"]
    preds = preds.assign(regime=pd.cut(preds["inflation"], bins=bins, labels=labels))
    rows = []
    for regime, grp in preds.groupby("regime", observed=True):
        for m in model_names:
            rows.append({"regime": regime, "model": m, "n": len(grp),
                         "Median abs. error (pp)": np.median(np.abs(grp["inflation_next"] - grp[m]))})
    return pd.DataFrame(rows).pivot(index="model", columns="regime", values="Median abs. error (pp)")


def main() -> None:
    data = pd.read_csv(PROCESSED_DIR / "panel.csv")
    print("Running walk-forward backtest...")
    preds = walk_forward(data)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_csv(REPORTS_DIR / "backtest_predictions.csv", index=False)

    names = list(model_zoo())
    table = summarize(preds, names)
    table.to_csv(REPORTS_DIR / "backtest_summary.csv")
    regimes = by_regime(preds, names)
    regimes.to_csv(REPORTS_DIR / "backtest_by_regime.csv")

    cal = preds[preds["q10"].notna()]
    raw_inside = ((cal["y"] >= cal["q10_raw"]) & (cal["y"] <= cal["q90_raw"])).mean()
    inside = ((cal["inflation_next"] >= cal["q10"]) & (cal["inflation_next"] <= cal["q90"])).mean()
    pd.Series({"raw_coverage": raw_inside, "calibrated_coverage": inside, "n": len(cal),
               "first_year": int(cal.target_year.min())}).to_csv(
        REPORTS_DIR / "interval_coverage.csv", header=False)

    pd.set_option("display.width", 160)
    print(f"\n{len(preds)} forecasts, target years "
          f"{preds.target_year.min()}-{preds.target_year.max()}\n")
    print(table.round(3).to_string())
    print("\nMedian abs. error (pp) by current inflation level:")
    print(regimes.round(2).to_string())
    print(f"\n80% prediction range, target years {cal.target_year.min()}-{cal.target_year.max()} "
          f"({len(cal)} forecasts): raw {raw_inside:.1%}, calibrated {inside:.1%}")


if __name__ == "__main__":
    main()
