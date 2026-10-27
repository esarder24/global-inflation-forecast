"""Interactive explorer for the inflation forecasts.

Run:  streamlit run app/streamlit_app.py
Reads only the precomputed files in data/ and reports/ (run the pipeline first).
"""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
BLUE, INK, GRAY = "#2a78d6", "#0b0b0b", "#8a8984"
BEST = "Boosted trees, all features"
BASELINE = "Persistence (baseline)"

st.set_page_config(page_title="Next-Year Inflation Forecaster", layout="wide")


@st.cache_data
def load():
    panel = pd.read_csv(ROOT / "data" / "processed" / "panel.csv")
    backtest = pd.read_csv(REPORTS / "backtest_predictions.csv")
    latest = pd.read_csv(REPORTS / "latest_forecasts.csv")
    summary = pd.read_csv(REPORTS / "backtest_summary.csv", index_col=0)
    coverage = pd.read_csv(REPORTS / "interval_coverage.csv", header=None, index_col=0)[1]
    return panel, backtest, latest, summary, coverage


panel, backtest, latest, summary, coverage = load()

st.title("Next-Year Inflation Forecaster")
st.caption("Forecasts next year's consumer-price inflation for 170+ countries from World Bank data. "
           "Every past forecast shown here was made using only data available at the time.")

tab_country, tab_accuracy, tab_all = st.tabs(["Country forecast", "How accurate is it?", "All countries"])

with tab_country:
    countries = sorted(latest["country"])
    default = countries.index("United States") if "United States" in countries else 0
    country = st.selectbox("Country", countries, index=default)
    f = latest[latest["country"] == country].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Forecast for {int(f.forecast_year)}", f"{f.forecast:.1f}%",
              delta=f"{f.forecast - f.latest_inflation:+.1f} pp vs {int(f.latest_data_year)}",
              delta_color="off")
    c2.metric("80% range", f"{f.low_80:.1f}% to {f.high_80:.1f}%")
    c3.metric(f"Actual inflation, {int(f.latest_data_year)}", f"{f.latest_inflation:.1f}%")
    if f.forecast_year <= panel["year"].max():
        st.info(f"The World Bank has not yet published {int(f.forecast_year)} inflation for {country}, "
                f"so this forecasts {int(f.forecast_year)} from {int(f.latest_data_year)} data.")

    hist = panel[panel["country"] == country][["year", "inflation"]].rename(
        columns={"year": "Year", "inflation": "Inflation (%)"})
    past = backtest[backtest["country"] == country].rename(columns={"target_year": "Year"})
    future = pd.DataFrame({"Year": [f.forecast_year], "Forecast": [f.forecast],
                           "low": [f.low_80], "high": [f.high_80]})
    band = pd.concat([past[["Year", "q10", "q90"]].rename(columns={"q10": "low", "q90": "high"}),
                      future[["Year", "low", "high"]]]).dropna()
    fc = pd.concat([past[["Year", BEST]].rename(columns={BEST: "Forecast"}), future[["Year", "Forecast"]]])

    x = alt.X("Year:Q", axis=alt.Axis(format="d", title=None))
    tooltip_fmt = ".1f"
    layers = [
        alt.Chart(band).mark_area(color=BLUE, opacity=0.15).encode(
            x=x, y=alt.Y("low:Q", title="Inflation (%)"), y2="high:Q",
            tooltip=[alt.Tooltip("Year:Q", format="d"), alt.Tooltip("low:Q", format=tooltip_fmt, title="80% low"),
                     alt.Tooltip("high:Q", format=tooltip_fmt, title="80% high")]),
        alt.Chart(hist).mark_line(color=INK, strokeWidth=2).encode(
            x=x, y="Inflation (%):Q",
            tooltip=[alt.Tooltip("Year:Q", format="d"), alt.Tooltip("Inflation (%):Q", format=tooltip_fmt)]),
        alt.Chart(fc).mark_line(color=BLUE, strokeWidth=2, strokeDash=[6, 4]).encode(
            x=x, y="Forecast:Q",
            tooltip=[alt.Tooltip("Year:Q", format="d"), alt.Tooltip("Forecast:Q", format=tooltip_fmt)]),
        alt.Chart(future).mark_point(color=BLUE, filled=True, size=90).encode(
            x=x, y="Forecast:Q",
            tooltip=[alt.Tooltip("Year:Q", format="d"), alt.Tooltip("Forecast:Q", format=tooltip_fmt)]),
    ]
    st.altair_chart(alt.layer(*layers).properties(height=380), width="stretch")
    st.caption("Black: actual inflation. Blue dashed: forecast made one year ahead. "
               "Shaded: 80% range. Dot: latest forecast.")

    if len(past):
        err = (past["inflation_next"] - past[BEST]).abs().median()
        base = (past["inflation_next"] - past[BASELINE]).abs().median()
        st.write(f"For {country}, the model's typical miss since {int(past.Year.min())} was "
                 f"**{err:.1f} pp**, versus **{base:.1f} pp** for 'same as last year'.")

with tab_accuracy:
    s = summary.loc[BEST]
    b = summary.loc[BASELINE]
    c1, c2, c3 = st.columns(3)
    c1.metric("Error cut vs. 'same as last year'", f"{s['Error cut vs. baseline']:.0%}")
    c2.metric("Typical miss", f"{s['Median abs. error (pp)']:.2f} pp",
              delta=f"{s['Median abs. error (pp)'] - b['Median abs. error (pp)']:+.2f} pp vs baseline",
              delta_color="inverse")
    c3.metric("80% range actually caught", f"{coverage['calibrated_coverage']:.0%}")
    st.markdown(
        "**How this was tested.** For every year from 2006 to 2025 the model was retrained using only "
        "earlier data, then asked to forecast that year for every country. Model settings were chosen "
        "on 1996-2005, which the test never touches.")
    st.image(str(REPORTS / "figures" / "error_by_year.png"))
    st.image(str(REPORTS / "figures" / "model_comparison.png"))
    st.image(str(REPORTS / "figures" / "feature_importance.png"))
    table = summary.copy()
    table["Within 2 pp"] = (table["Within 2 pp"] * 100).round(1).astype(str) + "%"
    table["Error cut vs. baseline"] = (table["Error cut vs. baseline"] * 100).round(1).astype(str) + "%"
    st.dataframe(table.round(3), width="stretch")

with tab_all:
    show = latest.rename(columns={
        "country": "Country", "latest_data_year": "Latest data", "latest_inflation": "Latest inflation (%)",
        "forecast_year": "Forecast year", "forecast": "Forecast (%)", "low_80": "80% low", "high_80": "80% high"})
    st.dataframe(show.drop(columns="iso3"), hide_index=True, width="stretch", height=600)
    st.download_button("Download forecasts (CSV)", latest.to_csv(index=False), "latest_forecasts.csv")
