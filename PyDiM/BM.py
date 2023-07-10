import numpy as np
import scipy.optimize as opt
import warnings

from PyDiM import _lib as lib
from PyDiM import plot

def bm(series, method="nls", prelimestimates=[], alpha=0.05, oos=None, display=True, stats = True):
    """
    Function that apply the standard Bass Model to fit the data

    ...

    Attributes
    ----------
    series : Series
        a data vector containing the series to be fitted
    method: str
        the estimation method of the model, 'nlm' or 'optim' (default is 'nlm')
    prelimestimates: list
        a vector containing the starting values used by the model
        (default values will be:'market potential' m = sum(series) + 100, 
        'innovation coe_fficient' p = 0.01, 'imitation coe_fficient' q = 0.1)
    alpha = float
        the confidence interval's significance level (default is 0.05)
    oos = int
        the number of predictions after the last observed value in the series, 
        if not specified (default will be set as the 25% of series length)
    display = bool
        if 'True' allows to display the fitted values for cumulative and instantaneous 
        observed data and oss (default is True)
    stats = bool
        if 'True' allows to display the statistic summary of the model (default is True)

    Returns
    -------
    ao = dict
        a dictionary containing all the usefull algorithm output:
        - 'type' : the name of the model,
        - 'functions' : the functions used to fit the cumulative and instantaneous data (e.g. z(t), z'(t)),
        - 'data' : the series given in input,
        - 'method' : optimization method given as input (nls/optim),
        - 'alpha' : alpha value given as input,
        - 'optim' : the parameters optimized,
        - 'residuals' : residuals of the fitting.

    """
    if len(prelimestimates) == 0:
        prelimestimates = np.array([np.sum(series)+100, 0.01, 0.1])
    if series.iloc[1] == 0:
         series = lib.handle_zeros(series)
    if oos == None:
        oos = round(len(series)*0.25)

    t = np.arange(1, len(series)+1, 1)
    cumsum = np.cumsum(series)

    with warnings.catch_warnings(record=True) as w:

        def _z(t, m, p, q):
            return (m * (1 - np.exp(- np.multiply((p + q), t))) / (1 + q / p * np.exp(-np.multiply((p + q), t))))
            
        def _zprime(t, m, p, q):
            return (p+q*_z(t, m, p, q)/m)*(m - _z(t, m, p, q))

        def _residuals(par, t):
            return cumsum - _z(t, par[0], par[1], par[2])

        def _f(par, t):
            return np.sum(_residuals(par, t)**2)

        if method == "nls":
            optim = opt.leastsq(func=_residuals, x0=prelimestimates, args=(t), full_output=1)
            res = optim[2]['fvec']

        elif method == "optim":
            mass = np.sum(series) + 1000
            minim = opt.minimize(fun=_f, x0=prelimestimates, args=(t), bounds=[(1e-10, mass), (1e-10, 1), (1e-10, 1)], method='L-BFGS-B')
            res = _residuals(minim.x, t)
            optim = [minim.x, minim.fun, res]

    if w:
        raise ValueError('Error encountered during the optimization, try with other parameters')

    model = {
        'type' :"Standard Bass Model",
        'functions' : [_z, _zprime],
        'data' : series,
        'method' : method,
        'alpha' : alpha,
        'optim' : optim,
        'residuals' : res,        
        }

    if display:
        plot.dimora_plot(model, 'fit', oos)
    
    return model