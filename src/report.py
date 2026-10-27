"""Step 4: charts and supporting numbers for the README.

Usage:  python -m src.report   (run after src.backtest)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import FIGURES_DIR, PROCESSED_DIR, RANDOM_STATE, REPORTS_DIR
from src.features import FEATURES, slog
from src.models import BEST_MODEL, gbm

BASELINE = "Persistence (baseline)"

# Colors (validated categorical palette, light mode) and neutral inks.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"

FEATURE_LABELS = {
    "infl_t": "Inflation this year", "infl_lag1": "Inflation last year",
    "infl_lag2": "Inflation 2 years ago", "infl_mean3": "3-year average inflation",
    "infl_change": "Change in inflation", "deflator_t": "GDP deflator inflation",
    "money_growth_t": "Money supply growth", "money_growth_lag1": "Money supply growth, last year",
    "fx_depr_t": "Currency depreciation vs US$", "fx_depr_lag1": "Currency depreciation, last year",
    "gdp_growth_t": "Real GDP growth", "unemployment_t": "Unemployment rate",
    "unemployment_change": "Change in unemployment", "lending_rate_t": "Bank lending rate",
    "real_rate_t": "Real interest rate", "imports_gdp_t": "Imports, % of GDP",
    "global_infl_t": "Global median inflation", "global_infl_change": "Change in global inflation",
}


def style(ax, title, subtitle=None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold",
                 pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=INK_2)


def new_fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, facecolor=SURFACE)
    plt.close(fig)


def per_year(preds):
    rows = []
    for year, grp in preds.groupby("target_year"):
        rows.append({
            "target_year": year,
            "baseline": np.median(np.abs(grp["inflation_next"] - grp[BASELINE])),
            "model": np.median(np.abs(grp["inflation_next"] - grp[BEST_MODEL])),
            "baseline_log": np.mean(np.abs(slog(grp["inflation_next"]) - slog(grp[BASELINE]))),
            "model_log": np.mean(np.abs(slog(grp["inflation_next"]) - slog(grp[BEST_MODEL]))),
        })
    return pd.DataFrame(rows)


def chart_error_by_year(yearly):
    fig, ax = new_fig(8, 4.2)
    x = yearly["target_year"]
    ax.plot(x, yearly["baseline"], color=GRAY, linewidth=2, marker="o", markersize=4,
            label="Persistence baseline")
    ax.plot(x, yearly["model"], color=BLUE, linewidth=2, marker="o", markersize=4,
            label="Boosted trees")
    events = {2009: "2009: inflation collapses\nafter the financial crisis",
              2022: "2022: worldwide surge;\nmodel expected a reversal"}
    for year, text in events.items():
        row = yearly[yearly["target_year"] == year]
        if row.empty:
            continue
        peak = row[["baseline", "model"]].max(axis=1).iloc[0]
        ax.annotate(text, (year, peak), xytext=(12, -4), textcoords="offset points",
                    fontsize=8.5, color=INK_2, va="top")
    ax.set_ylabel("Median forecast error (percentage points)", color=INK_2, fontsize=9)
    ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
    ax.set_ylim(0, yearly[["baseline", "model"]].max().max() * 1.12)
    ax.set_xticks(range(int(x.min()), int(x.max()) + 1, 2))
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper center", ncol=2)
    style(ax, "Forecast error by year", "Each point: typical miss across ~175 countries forecasting that year")
    save(fig, "error_by_year.png")


def chart_model_comparison(summary):
    order = [m for m in summary.index if m != BASELINE]
    vals = summary.loc[order, "Error cut vs. baseline"] * 100
    fig, ax = new_fig(8, 3.2)
    y = np.arange(len(order))[::-1]
    ax.barh(y, vals, height=0.55, color=BLUE)
    for yi, v in zip(y, vals):
        ax.text(v + 0.2, yi, f"{v:.1f}%", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, order, fontsize=9, color=INK)
    ax.set_xlabel("Reduction in mean absolute error vs. 'same as last year' (log scale)",
                  color=INK_2, fontsize=9)
    ax.set_xlim(0, max(vals) * 1.25)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    style(ax, "How much each model beats the baseline", "Walk-forward backtest, forecasts for 2006-2025")
    save(fig, "model_comparison.png")


def feature_importance(data):
    """Permutation importance: shuffle one feature at a time and measure how much worse
    the forecasts get. Model trained on targets up to 2018, scored on 2019-2025."""
    labeled = data[data["y"].notna()]
    train = labeled[labeled["target_year"] <= 2018]
    test = labeled[labeled["target_year"] >= 2019]
    model = gbm(FEATURES).fit(train[FEATURES], train["y"])
    result = permutation_importance(model, test[FEATURES], test["y"], scoring="neg_mean_absolute_error",
                                    n_repeats=10, random_state=RANDOM_STATE)
    imp = pd.Series(result.importances_mean, index=FEATURES).sort_values(ascending=False)
    imp.to_csv(REPORTS_DIR / "feature_importance.csv", header=["importance"])
    return imp


def chart_importance(imp, top=10):
    imp = imp.head(top)
    share = imp / imp.clip(lower=0).sum() * 100
    fig, ax = new_fig(8, 4.2)
    y = np.arange(len(imp))[::-1]
    ax.barh(y, share, height=0.6, color=BLUE)
    for yi, v in zip(y, share):
        ax.text(v + 0.5, yi, f"{v:.0f}%", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [FEATURE_LABELS[f] for f in imp.index], fontsize=9, color=INK)
    ax.set_xlabel("Share of total importance (top 10 shown)", color=INK_2, fontsize=9)
    ax.set_xlim(0, share.max() * 1.15)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    style(ax, "What the model relies on", "Permutation importance on 2019-2025 forecasts")
    save(fig, "feature_importance.png")


def chart_examples(preds, countries):
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), dpi=150, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, name in zip(axes.flat, countries):
        c = preds[preds["country"] == name].sort_values("target_year")
        band = c[c["q10"].notna()]
        ax.fill_between(band["target_year"], band["q10"], band["q90"], color=BLUE, alpha=0.15,
                        linewidth=0, label="80% range")
        ax.plot(c["target_year"], c["inflation_next"], color=INK, linewidth=2, label="Actual")
        ax.plot(c["target_year"], c[BEST_MODEL], color=BLUE, linewidth=2, linestyle="--",
                label="Forecast (made a year ahead)")
        style(ax, name)
        ax.title.set_fontsize(10)
        ax.set_ylim(bottom=min(0, c["inflation_next"].min() * 1.1))
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9,
               labelcolor=INK_2)
    fig.suptitle("Forecasts vs. what happened (inflation, %)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "country_examples.png", facecolor=SURFACE)
    plt.close(fig)


EXAMPLE_COUNTRIES = ["United States", "Germany", "Brazil", "Turkiye", "Nigeria", "Japan"]


def main():
    preds = pd.read_csv(REPORTS_DIR / "backtest_predictions.csv")
    summary = pd.read_csv(REPORTS_DIR / "backtest_summary.csv", index_col=0)
    data = pd.read_csv(PROCESSED_DIR / "panel.csv")

    yearly = per_year(preds)
    yearly.to_csv(REPORTS_DIR / "error_by_year.csv", index=False)
    wins = int((yearly["model_log"] < yearly["baseline_log"]).sum())
    print(f"Boosted trees beat the baseline in {wins} of {len(yearly)} test years (log-scale MAE)")
    print(yearly.round(3).to_string(index=False))

    imp = feature_importance(data)
    print("\nPermutation importance (increase in log-scale MAE when shuffled):")
    print(imp.round(4).to_string())

    chart_error_by_year(yearly)
    chart_model_comparison(summary)
    chart_importance(imp)
    chart_examples(preds, EXAMPLE_COUNTRIES)
    print(f"\nFigures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
