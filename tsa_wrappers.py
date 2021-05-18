######################################
##             Wrappers             ##
######################################
from typing import Callable
from pandas.core.series import Series


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
