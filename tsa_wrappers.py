######################################
##             Wrappers             ##
######################################
import numpy as np


class TSABenchmarkModel:
    def __init__(self, func, **kwargs):
        self.func = func
        self.kwargs = kwargs

    def fit(self, ts):
        self.ts = ts

    def forecast(self, h):
        return self.func(self.ts, h, **self.kwargs)
