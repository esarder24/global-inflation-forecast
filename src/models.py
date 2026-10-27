"""The models being compared. Each one is fit on the slog scale and predicts slog(next-year inflation)."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE
from src.features import FEATURES, INFLATION_HISTORY_FEATURES


class Persistence:
    """Baseline: 'next year's inflation will equal this year's'. No learning at all.

    Any model worth using has to beat this.
    """
    features = ["infl_t"]

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["infl_t"].to_numpy()


def ridge(features):
    # Linear regression with a small penalty. Linear models cannot handle missing values,
    # so we fill gaps with the training median and add 0/1 flags saying "this was missing".
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        Ridge(alpha=1.0),
    )
    model.features = features
    return model


def gbm(features, loss="absolute_error", quantile=None):
    # Gradient-boosted trees: many small decision trees, each correcting the last.
    # Handles missing values natively and captures non-linear effects, e.g. money growth
    # matters a lot when inflation is already high and very little when it is low.
    #
    # loss="absolute_error" makes the model predict the *median* outcome instead of the mean,
    # so a handful of hyperinflation years cannot drag every forecast around.
    # These settings were chosen in src/tune.py using only target years 1996-2005,
    # which never overlap the 2006+ test years.
    model = HistGradientBoostingRegressor(
        loss=loss,
        quantile=quantile,
        learning_rate=0.03,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )
    model.features = features
    return model


def model_zoo():
    """Point-forecast models, in the order they appear in results tables."""
    return {
        "Persistence (baseline)": Persistence(),
        "Ridge, inflation history only": ridge(INFLATION_HISTORY_FEATURES),
        "Ridge, all features": ridge(FEATURES),
        "Boosted trees, inflation history only": gbm(INFLATION_HISTORY_FEATURES),
        "Boosted trees, all features": gbm(FEATURES),
    }


BEST_MODEL = "Boosted trees, all features"


def interval_models():
    """10th and 90th percentile forecasts -> an 80% prediction range."""
    return {
        "q10": gbm(FEATURES, loss="quantile", quantile=0.10),
        "q90": gbm(FEATURES, loss="quantile", quantile=0.90),
    }


def fit_predict(model, train, test):
    model.fit(train[model.features], train["y"])
    return np.asarray(model.predict(test[model.features]))
