import numpy as np
import scipy.optimize as opt

from PyDiM import BM, _lib as lib
from PyDiM import plot

def ggm(series, mt = None, prelimestimates = None, alpha = 0.05, oos=None, display = True):
    """
    Function that apply the Guseo-Guidolin Model to fit the data

    ...

    Attributes
    ----------
    series : Series
        a data vector containing the series to be fitted
     mt : func
        this variable can get in input the market function (default is 'None', in this case
        the standard funciton will be used)
    prelimestimates: list
        a vector containing the starting values used by the model, they must be
        the coefficient estimations of the standard Bass Model
        (default is 'None', in this case, the function will compute the standard
        Bass Model by itself)
    alpha = float
        the confidence interval's significance level (default is 0.05)
    oos = int
        the number of predictions after the last observed value in the series, 
        if not specified (default will be set as the 25% of series length)
    display = bool
        if 'True' allows to display the fitted values for cumulative and instantaneous 
        observed data and oss (default is True)

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
        - 'market_potential' : the market function (m(t)) used by the model,
 
    """

    if prelimestimates == None:
        prelimestimates = np.concatenate((list(BM.bm(series, display=0)['optim'][0]),[0.001, 0.1]))
    if series.iloc[1] == 0:
        series = lib.handle_zeros(series)
    if oos == None:
        oos = round(len(series)*0.25)

    t = np.arange(1, len(series) + 1, 1)
    x_lim = np.arange(1, len(series) + 1 + oos, 1)
    cumsum = np.cumsum(series)

    def _mt_func(t, K, pc, qc):
        mt = K * np.sqrt(np.abs((1 - np.exp(-(pc + qc) * t)) / (1 + (qc / pc) * np.exp(-(pc + qc) * t))))
        return mt

    def _z_base_mt(t, par, mt):
        K, ps, qs, pc, qc = tuple(par[0:])
        z = mt(t, K, pc, qc) * (1 - np.exp(-(ps + qs) * t)) / (1 + (qs / ps) * np.exp(-(ps + qs)*t))
        return z
        
    def _z_defined_mt(t, par, mt):
        K, ps, qs = tuple(par[0:2])
        z = K * mt(t) * (1 - np.exp(-(ps + qs) * t)) / (1 + (qs / ps) * np.exp(-(ps + qs)*t))
        return z
    
    def _zprime(t, par):
        K, ps, qs, pc, qc = tuple(par[0:])
        F_t = (1 - np.exp(-(pc + qc) * t)) / (1 + (qc / pc) * np.exp(-(pc + qc) * t))
        G_t = (1 - np.exp(-(ps + qs) * t)) / (1 + (qs / ps) * np.exp(-(ps + qs) * t))
        
        ft = (pc * (pc+qc)**2 * np.exp(t*(pc+qc))) / ((pc * np.exp(t*(pc+qc)) + qc)**2)
        gt = (ps * (ps+qs)**2 * np.exp(t*(ps+qs))) / ((ps * np.exp(t*(ps+qs)) + qs)**2)
        
        k1_t = (1/2)* F_t**(-1/2) * G_t * ft
        k2_t = np.sqrt(F_t) * gt

        return K*(k1_t + k2_t)

    if type(mt) == 'function':
        _z = _z_defined_mt
    elif mt == 'base' or mt == None:
        _z = _z_base_mt
        mt = _mt_func
    else:
        raise KeyError("'mt' parameter must be either a function or None/left blank")

    def _residuals(par, t, mt):
        return cumsum - _z(t, par, mt)

    optim = opt.leastsq(func=_residuals, x0=prelimestimates, args=(t, mt), full_output=1)
    res = optim[2]['fvec']

    ao = {
        'type' :"Guseo-Guidolin Model",
        'functions' : [_z, _zprime], # <- t, stime[:], mt
        'data' : series,
        'alpha' : alpha,
        'optim' : optim,
        'residuals' : res,
        'market_potential': mt, # <- t, K, pc, qc
        }
    
    if display:
        plot.dimora_plot(ao, 'fit', oos) 

    return ao
