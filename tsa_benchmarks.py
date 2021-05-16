########################################
##             Benchmarks             ##
########################################
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