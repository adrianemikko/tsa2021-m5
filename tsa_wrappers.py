######################################
##             Wrappers             ##
######################################
from pandas.core.series import Series


class BaseFuncModel:
    def __init__(self, func: function, **kwargs) -> None:
        self.func = func
        self.kwargs = kwargs

    def fit(self, ts: Series) -> BaseFuncModel:
        self.ts = ts
        return self

    def forecast(self, h: int) -> Series:
        return self.func(self.ts, h, **self.kwargs)


class StatsModelsWrapper:
    def __init__(self, model, **kwargs) -> None:
        self.model = model
        self.kwargs = kwargs

    def fit(self, ts: Series) -> StatsModelsWrapper:
        self.model.fit(ts, **self.kwargs)
        return self

    def forecast(self, h: int) -> Series:
        return self.model.forecast(h)
