#####################################
##             Metrics             ##
#####################################
import numpy as np


def mae(y_true, y_pred):
    score = np.mean(np.abs(y_true - y_pred))
    return score


def rmse(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError('Lengths Mismatch')
    score = np.sqrt(np.mean((y_true - y_pred)**2))
    return score


def mase(y_true, y_pred, s_ts):
    if len(y_true) != len(y_pred):
        raise ValueError('Lengths Mismatch')
    ts = np.array(s_ts)
    score = np.mean(
        np.abs((y_true - y_pred)/np.mean(np.abs(ts[1:] - ts[:-1]))))
    return score


def rmsse(y_true, y_pred, ts):
    if len(y_true) != len(y_pred):
        raise ValueError('Lengths Mismatch')
    ts = np.array(ts)
    score = np.sqrt(
        np.mean((y_true - y_pred)**2)/np.mean((ts[1:] - ts[:-1])**2))
    return score


def mape(y_true, y_pred):
    score = np.mean(np.abs((y_true - y_pred)/y_true))
    return score


def mase_sea(y_true, y_pred, ts, m):
    score = np.mean(
        np.abs((y_true - y_pred)/np.mean(np.abs(ts[m:] - ts[:-m]))))
    return score
