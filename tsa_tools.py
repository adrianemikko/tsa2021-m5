# Authors
# * Adriane Mikko A. Amorado
# * Nika Karen O. Espiritu
# * Joanna Bebe G. Quinto

#############
## Imports ##
#############
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.seasonal import STL
from IPython.display import display
from matplotlib import pyplot as plt
from IPython.display import display
from itertools import product
from datetime import datetime
from tsa_benchmarks import *
from tsa_metrics import *
from tsa_wrappers import *
from tsa_preprocessing import *
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

np.set_printoptions(precision=4)
pd.set_option('precision', 4)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_colwidth', None)


###############
## Functions ##
###############


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


def rateMyForecast(
        train: DataFrame,
        test: DataFrame,
        forecast: DataFrame) -> DataFrame:
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
         'RMSE': rmse(np.array(test[col]), np.array(forecast[col])),
         'MAE': mae(np.array(test[col]), np.array(forecast[col])),
         'MASE': mase(np.array(test[col]), np.array(forecast[col]), np.array(train[col])),
         'RMSSE': rmsse(np.array(test[col]), np.array(forecast[col]), np.array(train[col]))}
        for col in test])
    return res.set_index('Group')


def compute_bottomup(df_orig, df_pred, lvl_pred):
    """Pre-processes the original data by level and returns 
    a dictionary of RMSSEs for each time series in each level.

    Parameters
    ----------
    df_orig : DataFrame
        DataFrame contaning the original data (index=date, columns=hts).
    df_pred : DataFrame
        DataFrame contaning the predictions using best model (index=date, columns=hts).
    lvl_pred : int
        Specified hierarchical level of the df_pred.

    Returns
    -------
    res_bylvl : DataFrame
        Nested dictionary of RMSSEs per time series per level
    """

    levels = json.loads(open('levels.json', 'r').read())
    res_bylvl = {}
    lvl_preds = list(sorted(range(2, lvl_pred), reverse=True))
    for x in list(sorted(range(1, lvl_pred), reverse=True)):
        if x in lvl_preds:
            orig = df_orig.sum(level=[levels[str(x)]], axis=1)
            pred = df_pred.sum(level=[levels[str(x)]], axis=1)
        else:
            orig = df_orig.sum(level=levels[str(x)], axis=1)
            pred = df_pred.sum(level=levels[str(x)], axis=1)

        # Test and Train Split
        train = orig.iloc[:1913, ]
        test = orig.iloc[1913:, ]

        # Initialize res dictionary by column
        res_bycol = {}

        if x in lvl_preds:
            for col in orig.columns:
                res_bycol[col] = rmsse(test[col], pred[col], train[col])
        else:
            res_bycol['Total'] = rmsse(test, pred, train)

        res_bylvl[x] = res_bycol

    return res_bylvl


def compute_topdown(df_full, df_pred, lvl_pred, approach='AHP'):
    """Pre-processes the original data by level and returns 
    a dictionary of RMSSEs for each time series in each level.

    Parameters
    ----------
    df_orig : DataFrame
        DataFrame contaning the original data (index=date, columns=hts).
    df_pred : DataFrame
        DataFrame contaning the predictions using best model (index=date, columns=hts).
    lvl_pred : int
        Specified hierarchical level of the df_pred.

    Returns
    -------
    res_bylvl : DataFrame
        Nested dictionary of RMSSEs per time series per level
    """
    levels = json.loads(open('levels.json', 'r').read())
    lvl_preds = list(levels.keys())[9:]
    ldf_pred_tot = df_pred.sum(axis=1)

    if approach == 'AHP':
        res_bylvl = {}
        forc_bylvl = {}

        for x in tqdm.tqdm(lvl_preds):
            propors = {}
            next_lvl_forc = {}
            res_bycol = {}

            lvl = df_full.sum(level=levels[x], axis=1)

            # Test and Train Split
            train = lvl.iloc[:1913, ]
            test = lvl.iloc[1913:, ]

            for col in tqdm.tqdm(lvl.columns.tolist()):
                propors[col] = sum(lvl[col]/lvl.sum(axis=1)) * (1/len(lvl))
                next_lvl_forc[col] = ldf_pred_tot * propors[col]
                res_bycol[col] = (rmsse(test[col],
                                  next_lvl_forc[col],
                                  train[col]))

            forc_bylvl[x] = next_lvl_forc
            res_bylvl[x] = res_bycol

    return res_bylvl


#############################################
##             Model Selection             ##
#############################################

class EndogenousTransformer(BaseEstimator, TransformerMixin):
    """
    Transform a univariate `X` into `X_train` of leght `w` and 
    `y_train` of length `h`. The total no. of data points will be:
    >>> len(X) - w - h + 1

    By default returns X, and y but can be configured to return 
    only `X` or `y`.

    This runs on `TimeseriesGenerator` backend.
    """

    def __init__(
            self, w: int,
            h: int,
            return_X: bool = True,
            return_y: bool = True,
            reshape: bool = False) -> None:
        """Initializes the transformer"""
        self.w = w
        self.h = h
        self.return_X = return_X
        self.return_y = return_y
        self.reshape = reshape

    def fit(self, X, y=None):
        self.X = X if not self.reshape else np.array(X).reshape(-1)
        # print('fitting', self.X.shape, (self.return_X, self.return_y))
        self.y = y
        return self

    def transform(self, X, y=None):
        self.X = X if not self.reshape else np.array(X).reshape(-1)
        # print('transforming', ((X, len(X)) if isinstance(X, list) else self.X.shape), (self.return_X, self.return_y))
        if len(self.X) == self.w:
            return np.array([self.X])
        X_train, _, y_train, _ = TimeseriesGenerator(
            self.X, self.y, self.w, self.h)
        # print('into', y_train.shape, (self.return_X, self.return_y))
        if self.return_X and self.return_y:
            return X_train, y_train
        elif self.return_X:
            return X_train
        elif self.return_y:
            return y_train
        else:
            raise ValueError

    def inverse_transform(self, X):
        return X.flatten()


def cross_val_score(X, est, config, scoring, cv):
    """
    Splits `X` using `cv` and predicts using `est` with `config` params.
    The output will be scored based on `scoring`.
    """
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


class OverlappingTimeSeriesSplit:
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


class GridSearch:
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


def evaluate_methods(
        methods: dict,
        X: Series,
        XOG: Series,
        tscv,
        col,
        w: int,
        h: int) -> dict:
    """
    Evaluates the different forecasting methods defined in a dict:
    >>> methods = {
    ...     str(method_name): {
    ...         `meta`: 'base' | 'stat' | 'ml_recursive' | 'ml_direct',
    ...         `model`: model
    ...     }
    ... }

    Parameters
    ----------
    methods: dict
        A dict containing information about the methods and an instance
        of each.
    X: Series
        Training set which may or may not be pre-processed
    XOG: Series
        Test set where the ground truth will come.
    tscv
        A splitter that returns `train_index` and `test_index`
    col
        Will be used as key for scores.
    w: int
        Lookback window
    h: int
        Forecast horizon

    Returns
    -------
    methods: dict
        Updated dict containing RMSE scores.

    """
    for method, model in methods.items():
        for train_index, test_index in tscv.split(X):
            X_train = X.iloc[train_index]
            X_test = XOG.iloc[train_index].iloc[-w:]
            y_test = XOG.iloc[test_index]
            if model['meta'] in ['stat', 'base']:
                y_pred = model['model'].fit(X_train).forecast(h)
            if model['meta'] in ['ml_recursive']:
                model['model'].fit(None, X_train)
                clear_output()
                y_pred = model['model'].predict(X_test).squeeze()
            if model['meta'] in ['ml_direct']:
                X_train, _, y_train, _ = TimeseriesGenerator(
                    X=X_train,
                    y=None,
                    w=w,
                    h=h)
                model['model'].fit(X_train, y_train)
                y_pred = model['model'].predict([X_test]).squeeze()
            methods[method].setdefault(col, []).append(
                rmse(np.array(y_test), y_pred))
    return methods


###############
## Ensembles ##
###############

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


class ensemble1:
    def __init__(self, w, s):
        self.w = w
        self.s = s = [7, 30, 365]

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
