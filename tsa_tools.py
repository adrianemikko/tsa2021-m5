# Authors
# * Adriane Mikko A. Amorado
# * Nika Karen O. Espiritu
# * Joanna Bebe G. Quinto

#############
## Imports ##
#############
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.seasonal import STL
from IPython.display import clear_output
from matplotlib import pyplot as plt
from IPython.display import display
from itertools import product
from datetime import datetime
from tsa_benchmarks import *
from tsa_metrics import *
import lightgbm as lgb
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import pickle
import tqdm
import json


warnings.filterwarnings("ignore")
register_matplotlib_converters()
sns.set_style('darkgrid')
np.set_printoptions(precision=2)
pd.set_option('precision', 2)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_colwidth', None)


###############
## Functions ##
###############

def timeSeriesFiltering(ts, lower=np.NINF, upper=np.inf, plot=False):
    """
    Replace values lower than `lower` and higher that `upper
    with interpolated values.
    """
    if plot:
        fig, axes = plt.subplots(figsize=(12, 5), nrows=2, sharex=True)
        ts.plot(ax=axes[0], title='Original', c='k')
        axes[0].axhline(lower, c='r', lw=0.5, ls=':')
        axes[0].axhline(upper, c='r', lw=0.5, ls=':')

    ts = (ts
          .where((ts >= lower) & (ts <= upper))
          .interpolate(method='time'))

    if plot:
        ts.plot(ax=axes[1], title='Interpolated', c='k')
        plt.tight_layout()

    return ts


def mslt(ts, s=[12], plot=False):
    components = {'Data': ts}
    series = ts.copy()
    for t in s:
        res = STL(
            series, period=t, seasonal=t if t % 2 else t+1, robust=True).fit()
        components[f'Trend'] = res.trend
        components[f'Seasonal{t}'] = res.seasonal
        series = res.trend + res.resid
    components[f'Remainder'] = res.resid
    res = pd.DataFrame(components)
    if plot:
        res.plot(
            subplots=True, layout=(-1, 1), figsize=(12, 10), color='k',
            title=[*res.columns], legend=False)
        plt.tight_layout()
    return res



def rateMyForecast(train, test, forecast):
    """
    Evalute the forcast per group, given train, test, and forecast tables.

    The function evaluates the metrics per column of the provided table.

    Parameters
    ----------
    train : DataFrame
        DataFrame contaning the train set.
    test : DataFrame
        DataFrame contaning the test set.
    forecast : DataFrame
        DataFrame contaning the forecast set.

    Returns
    -------
    DataFrame
        DataFrame contaning the metrics as columns, groups as rows,
        and scores as values.

    """
    res = pd.DataFrame([
        {'Group': col,
         'RMSE': rmse(test[col], forecast[col]),
         'MAE': mae(test[col], forecast[col]),
         'MASE': mase(test[col], forecast[col], train[col]),
         'RMSSE': rmsse(test[col], forecast[col], train[col])}
        for col in test])
    display(res.set_index('Group'))
    return res.set_index('Group')


#############################################
##             Model Selection             ##
#############################################

def TimeseriesGenerator(X, y, w, h):
    X_train = np.vstack(timeseries_dataset_from_array(
        X, targets=None, sequence_length=w, end_index=len(X)-h))
    y_train = np.vstack(timeseries_dataset_from_array(
        X, targets=None, sequence_length=h, start_index=w))
    X_test = X[None, -w:]
    y_test = y[None, :] if y else None
    return X_train, X_test, y_train, y_test


def cross_val_score(X, est, config, scoring, cv):
    param = config.copy()
    h = param.pop('h')
    w = param.pop('w')
    folds = cv.split(X, h)
    scores = {metric: [] for metric in scoring}
    for train, val in folds:
        X_train, X_test, y_train, y_test = TimeseriesGenerator(
            train, val, w, h)
        est.set_params(**param)
        est.fit(X_train, y_train)
        y_hat = est.predict(X_test)
        for metric in scores:
            scores[metric].append(scoring[metric](y_test, y_hat))
    return scores


def cross_val_predict(X, est, config, cv):
    param = config.copy()
    h = param.pop('h')
    w = param.pop('w', None)
    folds = cv.split(X, h)
    fit_params = {}
    res = {}
    for k, (train, val) in enumerate(folds):
        if w:
            X_train, X_test, y_train, y_test = TimeseriesGenerator(
                train, val, w, h)
            est.set_params(**param)
            est.fit(X_train, y_train)
            y_hat = est.predict(X_test)[0]
        else:
            try:
                model = est(X, **param)
                fit = model.fit(**fit_params)
                y_hat = fit.forecast(h)
            except:
                y_hat = np.full(len(val), np.nan)
        res.update({(k, i): y for i, y in enumerate(y_hat)})
    return res


class TimeSeriesSplit:
    def __init__(self, val_size):
        self.val_size = val_size

    def split(self, design_set, h):
        val_end = len(design_set)
        divider = val_end - h
        dataset = []
        while len(design_set) - divider <= self.val_size:
            dataset.append(
                (design_set[np.arange(0, divider)],
                 design_set[np.arange(divider, val_end)]))
            val_end -= 1
            divider -= 1
        return dataset[::-1]


class GridSearchCV:
    def __init__(self, estimator, param_grid, cv, scoring=[]):
        self.est = estimator
        self.param_grid = param_grid
        self.param_list = [
            dict(zip(param_grid.keys(), params))
            for params in product(*param_grid.values())]
        self.cv = cv
        self.scoring = scoring

    def fit(self, X, scores=False):
        self.cv_results_ = []
        self.df_records_ = []
        for param in tqdm.tqdm(self.param_list):
            if scores:
                res = {
                    'params': param.copy(),
                    **cross_val_score(
                        X, self.est, param, self.scoring, self.cv)}
                rec = {
                    'Lookback': res['params']['w'],
                    'Horizon': res['params']['h'],
                    'Average RMSE': np.mean(res['rmse']),
                    'Stdev RMSE': np.std(res['rmse'])}
                rec['Sum'] = (rec['Average RMSE'] + rec['Stdev RMSE'])
#                 self.best_params = (
#                     self.df.nsmallest(1, 'Sum').iloc[0].to_dict())
            else:
                res = {
                    'params': param.copy(),
                    **cross_val_predict(
                        X, self.est, param, self.cv)}
                rec = res
            self.cv_results_.append(res)
            self.df_records_.append(rec)
        self.df = pd.DataFrame(self.df_records_)


def forecastUsingConfig(est, regions, design_set, test_set):
    forecast = {}
    for region in regions:
        train = design_set[region['Region']]
        test = test_set[region['Region']]
        w = int(region['Lookback'])
        h = int(region['Horizon'])
        X_train, X_test, y_train, y_test = TimeseriesGenerator(
            train, test, w, h)
#         est.set_params(**param)
        fit = est.fit(X_train, y_train)
        forecast[region['Region']] = fit.predict(X_test)[0]
    forecast_set = pd.DataFrame(forecast)
    forecast_set.index = test_set.index
    return forecast_set


###############
## Ensembles ##
###############

class ensemble1:
    def __init__(self, w, s):
        self.w = w
        self.s = s=[7, 30, 365]
    
    def fit(self, ts, lower=np.NINF, upper=np.inf, ):
        series = timeSeriesFiltering(ts, lower, upper)
        self.res = mslt(series, s=self.s)
        
        # Seasonal
        
        # Trend
        self.trend_fit = ETSModel(self.res.Trend, trend='add').fit()
        
        # Residuals
        X_train, _, y_train, _ = TimeseriesGenerator(
            self.res.Remainder, y=None, w=self.w, h=1)
        resid_model = lgb.LGBMRegressor(random_state=1)
        self.resid_fit = resid_model.fit(X_train, y_train)
        
        return self
    
    def forecast(self, h):
        forecasts = {'Data': np.nan}
        forecasts['Trend'] = self.trend_fit.forecast(h)
        for seasonality in self.s:
            forecasts[f'Seasonal{seasonality}'] = snaivef(
                self.res[f'Seasonal{seasonality}'], h, seasonality)
        resid = self.res.Remainder.tolist()
        for _ in range(h):
            f = self.resid_fit.predict([resid[-self.w:]])
            forecasts.setdefault('Remainder', []).extend(f)
            resid.extend(f)
        return pd.DataFrame(forecasts).assign(
            Data=lambda x: np.nansum(x, axis=1))
