"""Run the whole pipeline in order. Works the same on Windows, macOS and Linux.

    python run_all.py              # use the data snapshot already in data/raw/
    python run_all.py --download   # fetch fresh World Bank data first
"""
import sys

from src import backtest, download_data, features, report, train_final, tune

if __name__ == "__main__":
    steps = [
        ("Build country-year dataset", features.main),
        ("Choose model settings on 1996-2005 (validation)", tune.main),
        ("Walk-forward backtest on 2006-latest (test)", backtest.main),
        ("Charts and supporting numbers", report.main),
        ("Train final model and forecast next year", train_final.main),
    ]
    if "--download" in sys.argv:
        steps.insert(0, ("Download World Bank data", download_data.main))
    for i, (title, step) in enumerate(steps, 1):
        print(f"\n=== Step {i}/{len(steps)}: {title} ===")
        step()
    print("\nDone. Launch the app with:  streamlit run app/streamlit_app.py")
