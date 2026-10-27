"""Choose the boosted-tree settings using ONLY the validation years (targets 1996-2005).

The test years (2006 onward) are never touched here. Tuning on the same years you report
results for would make the results look better than they really are.

Usage:  python -m src.tune
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.config import FIRST_TEST_YEAR, PROCESSED_DIR, RANDOM_STATE
from src.features import FEATURES

VALIDATION_YEARS = range(1996, FIRST_TEST_YEAR)

CANDIDATES = [
    # (loss, max_leaf_nodes, min_samples_leaf, learning_rate, max_iter)
    ("squared_error", 15, 40, 0.05, 400),
    ("squared_error", 15, 40, 0.03, 200),
    ("squared_error", 7, 80, 0.03, 150),
    ("absolute_error", 15, 40, 0.05, 400),
    ("absolute_error", 15, 40, 0.03, 200),
    ("absolute_error", 7, 80, 0.03, 300),
    ("absolute_error", 31, 20, 0.05, 400),
]


def validation_error(make_model, data) -> float:
    errors = []
    for Y in VALIDATION_YEARS:
        train = data[data["target_year"] <= Y - 1]
        test = data[data["target_year"] == Y]
        model = make_model().fit(train[FEATURES], train["y"])
        errors.append(np.abs(test["y"] - model.predict(test[FEATURES])))
    return float(np.concatenate(errors).mean())


def main() -> None:
    data = pd.read_csv(PROCESSED_DIR / "panel.csv")
    data = data[data["y"].notna()]
    persistence = np.mean([np.abs(data.loc[data.target_year == Y, "y"]
                                  - data.loc[data.target_year == Y, "infl_t"]).mean()
                           for Y in VALIDATION_YEARS])
    print(f"Validation target years {VALIDATION_YEARS.start}-{VALIDATION_YEARS.stop - 1}")
    print(f"Persistence baseline, mean abs. error (log scale): {persistence:.4f}\n")
    rows = []
    for loss, leaves, min_leaf, lr, iters in CANDIDATES:
        def make(loss=loss, leaves=leaves, min_leaf=min_leaf, lr=lr, iters=iters):
            return HistGradientBoostingRegressor(
                loss=loss, learning_rate=lr, max_iter=iters, max_leaf_nodes=leaves,
                min_samples_leaf=min_leaf, l2_regularization=1.0, early_stopping=False,
                random_state=RANDOM_STATE)
        err = validation_error(make, data)
        rows.append((loss, leaves, min_leaf, lr, iters, round(err, 4)))
        print(f"{loss:<15} leaves={leaves:<3} min_leaf={min_leaf:<3} lr={lr:<5} trees={iters:<4} -> {err:.4f}")
    best = min(rows, key=lambda r: r[-1])
    print(f"\nBest: {best}")


if __name__ == "__main__":
    main()
