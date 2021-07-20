######################################
##             Wrappers             ##
######################################
from typing import Callable
import pandas as pd
import numpy as np
from IPython.display import clear_output
from pandas.core.series import Series
from beyondpdm.tsa_preprocessing import TimeseriesGenerator


class BaseFuncModel:
    """
    Wraps a function into a model `statsmodels`-like class that can use
    `fit` and `forecast` methods.

    Parameters
    ----------
    func : Callable
        A function that can be called with the `ts`, and `h` arguments.

    """

    def __init__(self, func: Callable, **kwargs) -> None:
        self.func = func
        self.kwargs = kwargs

    def fit(self, ts: Series) -> None:
        """
        Saves the time series data as an attribute.

        Parameters
        ----------
        ts : Series
            Time series to apply function to.

        Returns
        -------
        self : BaseFuncModel
            The fitted model

        """
        self.ts = ts
        return self

    def forecast(self, h: int) -> Series:
        """
        Calls the `func` to output the predictions.

        Parameters
        ----------
        h : int
            The forecast horizon to be used.

        Returns
        -------
        predictions : Series
            The predictions on `ts` using `func` for `h` periods.

        """
        return self.func(self.ts, h, **self.kwargs)


class StatsModelsWrapper:
    """
    Wraps a `statsmodels` class into a model class that can use
    `fit` and `forecast` methods.

    Parameters
    ----------
    model : model
        A `statsmodels` model class to be adapted.

    """

    def __init__(self, model, **kwargs) -> None:
        self.model = model
        self.kwargs = kwargs

    def fit(self, ts: Series) -> None:
        """
        Instantiates a `statsmodels` model using the `ts` provided
        and calls the fit method of that model.

        Parameters
        ----------
        ts : Series
            Time series fit the model to.

        Returns
        -------
        self : StatsModelsWrapper
            The fitted model

        """
        self.fitted_model = self.model(ts, **self.kwargs).fit()
        return self

    def forecast(self, h: int) -> Series:
        """
        Calls the `forecast` method of the `fitted_model` to output
        the predictions.

        Parameters
        ----------
        h : int
            The forecast horizon to be used.

        Returns
        -------
        predictions : Series
            The predictions on `ts` using `fitted_model`'s 
            `forecast` method for `h` periods.

        """
        return self.fitted_model.forecast(h)


class RecursiveRegressor:
    """
    Wraps an `sklearn` estimator into a model class that forecasts
    recursively on the fitted data.

    Parameters
    ----------
    model : model
        An `slkearn` model instance to be adapted.

    """

    def __init__(self, estimator) -> None:
        self.estimator = estimator
        self.__dict__.update(estimator.get_params())

    def fit(self, X: Series, y: Series):
        """
        Transforms the data into a time series array with a horizon of 1
        and fits the estimator on it.

        Parameters
        ----------
        X : Series
            Time series fit the model to as exogenous features.
        y : Series
            Time series fit the model to as endogenous features
            and targets.

        Returns
        -------
        self : RecursiveRegressor
            The fitted model

        """
        X_train, _, y_train, _ = TimeseriesGenerator(
            y, None, self.w, h=1)
        self.fitted_model = self.estimator.fit(X_train, y_train)
        clear_output()
        return self

    def predict(self, X: Series):
        """
        Calls the `predict` method of the fitted `estimator` to output
        the predictions.

        Parameters
        ----------
        X : Series
            The window features on which to predict.

        Returns
        -------
        predictions : Series
            The predictions on `X` using `fitted_model`'s 
            `predict` method for `h` periods.

        """
        forecasts = []
        X_train = list(X)
        for _ in range(self.h):
            y_pred = self.fitted_model.predict([X_train[-self.w:]])
            forecasts.extend(y_pred)
            X_train.extend(y_pred)
        return pd.Series(forecasts)
