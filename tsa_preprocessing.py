###########################################
##             Preprocessing             ##
###########################################
from pandas.core.series import Series
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
import numpy as np


def timeSeriesFiltering(
        ts: Series,
        lower: float = np.NINF,
        upper: float = np.inf,
        plot: bool = False) -> Series:
    """
    Replace values less than `lower` and more that `upper`
    with interpolated values.

    If `ts` ends or starts with filtered values,
    uses `ffill` or `bfill` respectively

    Can plot the differences.
    """
    if plot:
        fig, axes = plt.subplots(figsize=(12, 5), nrows=2, sharex=True)
        ts.plot(ax=axes[0], title='Original', c='k')
        axes[0].axhline(lower, c='r', lw=0.5, ls=':')
        axes[0].axhline(upper, c='r', lw=0.5, ls=':')

    ts = (ts
          .where((ts >= lower) & (ts <= upper))
          .interpolate(method='time')
          .ffill()
          .bfill()
          )

    if plot:
        ts.plot(ax=axes[1], title='Interpolated', c='k')
        plt.tight_layout()

    return ts


def TimeseriesGenerator(
        X: Series,
        y: Series,
        w: int,
        h: int):
    """
    Returns `X_train`, `X_test`, `y_train`, and `y_test` from a given 
    endog features `X` and `y`; this assumes that the given `y` is of
    lenth `h`, and that there is only a singular pair of `X` and `y`
    in the test set.

    This is originally used in creating train/test set from the CV splits
    which outputs univariate `X` and `y` that can be used in `statsmodels`.
    However this can be repurposed to only output the training set by:
    >>> X_train, _, y_train, _ = TimeseriesGenerator(X, y=None)
    """
    X_train = np.vstack(timeseries_dataset_from_array(
        X, targets=None, sequence_length=w, end_index=len(X)-h))
    y_train = np.vstack(timeseries_dataset_from_array(
        X, targets=None, sequence_length=h, start_index=w))
    X_test = X[None, -w:]
    y_test = y[None, :] if y is not None else None
    return X_train, X_test, y_train, y_test
