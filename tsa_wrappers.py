######################################
##             Wrappers             ##
######################################
from typing import Callable
import pandas as pd
import numpy as np
from pandas.core.series import Series
from tsa_preprocessing import TimeseriesGenerator
from IPython.display import clear_output


class BaseFuncModel:
    """
    Wraps a base function in a class to be able to use the 
    `fit` and `forecast` methods.
    """

    def __init__(self, func: Callable, **kwargs) -> None:
        self.func = func
        self.kwargs = kwargs

    def fit(self, ts: Series) -> None:
        self.ts = ts
        return self

    def forecast(self, h: int) -> Series:
        return self.func(self.ts, h, **self.kwargs)


class StatsModelsWrapper:
    """
    Wraps a statsmodels function in a class to be able to use the 
    a reusable `fit` and `forecast` methods.
    """

    def __init__(self, model, **kwargs) -> None:
        self.model = model
        self.kwargs = kwargs

    def fit(self, ts: Series) -> None:
        self.fitted_model = self.model(ts, **self.kwargs).fit()
        return self

    def forecast(self, h: int) -> Series:
        return self.fitted_model.forecast(h)


class RecursiveRegressor:
    """
    Wraps an `sklearn` model in a class that is able to forecast recursively.

    For univariate data, set `X = None` and `y = ts` on fit.

    TODO@adrianemikko: enable usability for exogenous variables
    """

    def __init__(self, estimator) -> None:
        self.estimator = estimator
        self.__dict__.update(estimator.get_params())

    def fit(self, X: Series, y: Series):
        X_train, _, y_train, _ = TimeseriesGenerator(
            y, None, self.w, h=1)
        self.fitted_model = self.estimator.fit(X_train, y_train)
        clear_output()
        return self

    def predict(self, X: Series):
        forecasts = []
        X_train = list(X)
        for _ in range(self.h):
            y_pred = self.fitted_model.predict([X_train[-self.w:]])
            forecasts.extend(y_pred)
            X_train.extend(y_pred)
        return pd.Series(forecasts)
