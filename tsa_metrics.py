#####################################
##             Metrics             ##
#####################################
import numpy as np


def mae(y_true, y_pred):
    """
    Mean absolute error regression loss.

    Parameters
    ----------
    y_true : Series
        Time series containing the true values.
    y_pred : Series
        Time series containing the predicted values.

    Returns
    -------
    score : float
        Score of the predicted vs the actual.
    """
    score = np.mean(np.abs(y_true - y_pred))
    return score


def rmse(y_true, y_pred):
    """
    Root mean squared error regression loss.

    Parameters
    ----------
    y_true : Series
        Time series containing the true values.
    y_pred : Series
        Time series containing the predicted values.

    Returns
    -------
    score : float
        Score of the predicted vs the actual.
    """
    if len(y_true) != len(y_pred):
        raise ValueError('Lengths Mismatch')
    score = np.sqrt(np.mean((y_true - y_pred)**2))
    return score


def mase(y_true, y_pred, s_ts):
    """
    Mean absolute scaled error regression loss.

    Parameters
    ----------
    y_true : Series
        Time series containing the true values.
    y_pred : Series
        Time series containing the predicted values.
    s_ts : Series
        Time series containing the whole train set.

    Returns
    -------
    score : float
        Score of the predicted vs the actual.
    """
    if len(y_true) != len(y_pred):
        raise ValueError('Lengths Mismatch')
    ts = np.array(s_ts)
    score = np.mean(
        np.abs((y_true - y_pred)/np.mean(np.abs(ts[1:] - ts[:-1]))))
    return score


def rmsse(y_true, y_pred, ts):
    """
    Root mean squared scaleed error regression loss.

    Parameters
    ----------
    y_true : Series
        Time series containing the true values.
    y_pred : Series
        Time series containing the predicted values.
    ts : Series
        Time series containing the whole train set.

    Returns
    -------
    score : float
        Score of the predicted vs the actual.
    """
    if len(y_true) != len(y_pred):
        raise ValueError('Lengths Mismatch')
    ts = np.array(ts)
    score = np.sqrt(
        np.mean((y_true - y_pred)**2)/np.mean((ts[1:] - ts[:-1])**2))
    return score


def mape(y_true, y_pred):
    """
    Mean absolute percentage error regression loss.

    Parameters
    ----------
    y_true : Series
        Time series containing the true values.
    y_pred : Series
        Time series containing the predicted values.

    Returns
    -------
    score : float
        Score of the predicted vs the actual.
    """
    if len(y_true) != len(y_pred):
        raise ValueError('Lengths Mismatch')
    score = np.mean(np.abs((y_true - y_pred)/y_true))
    return score


def mase_sea(y_true, y_pred, ts, m):
    """
    Mean absolute squared error regression loss for seasonal forecasts.

    Parameters
    ----------
    y_true : Series
        Time series containing the true values.
    y_pred : Series
        Time series containing the predicted values.
    s_ts : Series
        Time series containing the whole train set.
    m: int
        Seasonality period.

    Returns
    -------
    score : float
        Score of the predicted vs the actual.
    """
    if len(y_true) != len(y_pred):
        raise ValueError('Lengths Mismatch')
    score = np.mean(
        np.abs((y_true - y_pred)/np.mean(np.abs(ts[m:] - ts[:-m]))))
    return score
