import numpy as np

def meanf(ts, h):
    f = np.mean(ts)
    f = np.repeat(f, repeats=h)
    return f

def naivef(ts, h):
    f = ts[-1]
    f = np.repeat(f, repeats=h)
    return f

def snaivef(ts, h, m):
    f = np.zeros(h)
    for i in range(h):
        f[i] = ts[-(m - i%m)]
    return f

def driftf(ts, h):
    T = len(ts)
    f = np.zeros(h)
    for i in range(h):
        f[i] = ts[-1] + (i+1)*((ts[-1] - ts[0])/(T - 1))
    return f
	
def mae(y_true, y_pred):
    score = np.mean(np.abs(y_true - y_pred))
    return score

def rmse(y_true, y_pred):
    score = np.sqrt(np.mean((y_true - y_pred)**2))
    return score

def mape(y_true, y_pred):
    score = np.mean(np.abs((y_true - y_pred)/y_true))
    return score

def mase(y_true, y_pred, ts):
    score = np.mean(np.abs((y_true - y_pred)/np.mean(np.abs(ts[1:] - ts[:-1]))))
    return score

def mase_sea(y_true, y_pred, ts, m):
    score = np.mean(np.abs((y_true - y_pred)/np.mean(np.abs(ts[m:] - ts[:-m]))))
    return score

def rmsse(y_true, y_pred, ts):
    score = np.sqrt(np.mean((y_true - y_pred)**2)/np.mean((ts[1:] - ts[:-1])**2))
    return score
