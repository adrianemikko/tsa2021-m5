######################################
##             Wrappers             ##
######################################
from typing import Callable
import pandas as pd
import numpy as np
from IPython.display import clear_output
from pandas.core.series import Series
from tsa_preprocessing import TimeseriesGenerator


class BaseFuncModel:
    def __init__(self, func: Callable, **kwargs) -> None:
        self.func = func
        self.kwargs = kwargs

    def fit(self, ts: Series) -> None:
        self.ts = ts
        return self

    def forecast(self, h: int) -> Series:
        return self.func(self.ts, h, **self.kwargs)


class StatsModelsWrapper:
    def __init__(self, model, **kwargs) -> None:
        self.model = model
        self.kwargs = kwargs

    def fit(self, ts: Series) -> None:
        self.fitted_model = self.model(ts, **self.kwargs).fit()
        return self

    def forecast(self, h: int) -> Series:
        return self.fitted_model.forecast(h)


class RecursiveRegressor:
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
