"""Step 5: train the final model on all known outcomes and forecast each country's next year.

Usage:  python -m src.train_final   (run after src.backtest; it reuses the backtest's recent
                                     errors to set the width of the 80% range)
"""
import joblib
import pandas as pd

from src.backtest import CALIBRATION_YEARS, interval_widening
from src.config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from src.features import FEATURES, inv_slog
from src.models import BEST_MODEL, gbm, interval_models


def main() -> None:
    data = pd.read_csv(PROCESSED_DIR / "panel.csv")
    labeled = data[data["y"].notna()]

    point = gbm(FEATURES).fit(labeled[FEATURES], labeled["y"])
    q = {name: m.fit(labeled[FEATURES], labeled["y"]) for name, m in interval_models().items()}

    # Range widening from the most recent backtest years (same method as the backtest).
    backtest = pd.read_csv(REPORTS_DIR / "backtest_predictions.csv")
    recent = backtest[backtest["target_year"] > backtest["target_year"].max() - CALIBRATION_YEARS]
    widen = interval_widening(recent)

    # Each country's most recent year with data is the starting point for its forecast.
    latest = data.sort_values("year").groupby("iso3").tail(1)
    latest = latest[latest["year"] >= data["year"].max() - 1]   # skip countries with stale data
    X = latest[FEATURES]
    out = latest[["iso3", "country", "year", "inflation"]].rename(
        columns={"year": "latest_data_year", "inflation": "latest_inflation"})
    out["forecast_year"] = out["latest_data_year"] + 1
    out["forecast"] = inv_slog(point.predict(X))
    out["low_80"] = inv_slog(q["q10"].predict(X) - widen)
    out["high_80"] = inv_slog(q["q90"].predict(X) + widen)
    out = out.sort_values("country").round(2)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"point": point, "q10": q["q10"], "q90": q["q90"], "widen": widen,
                 "features": FEATURES, "name": BEST_MODEL}, MODELS_DIR / "inflation_model.joblib")
    out.to_csv(REPORTS_DIR / "latest_forecasts.csv", index=False)

    print(f"Trained on {len(labeled)} country-years. Range widening: {widen:.3f} (log scale)")
    print(f"Forecasts for {len(out)} countries:")
    print(out["forecast_year"].value_counts().sort_index().to_string())
    print(out[out["country"].isin(["United States", "Germany", "Turkiye", "Argentina",
                                   "Japan", "India", "Nigeria"])].to_string(index=False))


if __name__ == "__main__":
    main()
