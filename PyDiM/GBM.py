import numpy as np
import scipy.optimize as opt

from PyDiM import _lib as lib
from PyDiM import plot

def gbm(series, shock, nshock, prelimestimates, alpha=0.05, oos=None, display=True):
    """
    Function that apply the Generalized Bass Model to fit the data

    ...

    Attributes
    ----------
    series : Series
        a data vector containing the series to be fitted
    shock : str
        a string specifying the kind of shock to apply on the model fitting
    nshock : int
        the number of shocks
    prelimestimates: list
        a vector containing the starting values used by the model, the vector should contain the following in order:
        [m, p, q, a1, b1, c1, ...]
        (if 'mixed' shock is selected the first tuple [a1, b1, c1] are used for the exp shock, the second tuple
        [a2, b2, c2] are used for the rect shock)
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
        - 'alpha' : alpha value given as input,
        - 'optim' : the parameters optimized,
        - 'residuals' : residuals of the fitting,
        - 'shocks' : tuple containing number and kind of the shocks [nshock, shock],
        - 'x_functions': tuple containing the functions z(t) and z'(t) related with the shocks[intx, xt].
    """

    if shock == None or shock not in ["exp", "rett", "mixed"]:
        print("wrong value in 'shock' variable")
        return 0
    if prelimestimates == None:
        print("add values in prelimestimates")
        return 0
    if nshock <= 0:
        print('Number of shocks must be > 0')
        return 0
    if (nshock == 1 or nshock >= 3) and shock == 'mixed':
        print('Sorry but we have not implemented a model with % i mixed shocks' % nshock)
        return 0

    if type(shock) != list():
        shock = [shock]
    if series[0] == 0:
         series = lib.handle_zeros(series)         
    if oos == None:
        oos = round(len(series)*0.25)

    t = np.arange(1, len(series) + 1, 1)
    cumsum = np.cumsum(series)

    '''EXPONENTIAL SHOCKS GENERALIZED FUNCTIONS'''
    def _exp_intx_gen(t, index, shock_par):
        a = shock_par[3*index]
        b = shock_par[3*index+1]
        c = shock_par[3*index+2]
        intx = c*(1/b)*(np.exp(np.dot(b,(t-a)))-1)*(t>=a)

        return intx

    def _exp_xt_gen(t, index, shock_par):
        a = shock_par[3*index]
        b = shock_par[3*index+1]
        c = shock_par[3*index+2]
        xt = (c*np.exp(np.dot(b, (t-a))))*(t >= a)

        return xt

    '''RECTANGULAR SHOCKS GENERALIZED FUNCTIONS'''
    def _rett_intx_gen(t, index, shock_par):
        a = shock_par[3*index]
        b = shock_par[3*index+1]
        c = shock_par[3*index+2]
        intx = np.dot(c,(t-a))*(t>=a)*(t<=b) + c*(b-a)*(t>b)

        return intx

    def _rett_xt_gen(t, index, shock_par):
        a = shock_par[3*index]
        b = shock_par[3*index+1]
        c = shock_par[3*index+2]
        xt = c*(t>=a)*(t<=b)

        return xt

    '''RECTANGULAR SHOCKS GENERALIZED FUNCTIONS'''
    def _mix_intx_gen(t, index, shock_par):
        if shock[index-1] == 'exp':
            intx = _exp_intx_gen(t, index, shock_par)
        else:
            intx = _rett_intx_gen(t, index, shock_par)
        return intx

    def _mix_xt_gen(t, index, shock_par):
        if shock[index-1] == 'exp':
            xt = _exp_xt_gen(t, index, shock_par)
        else:
            xt = _rett_xt_gen(t, index, shock_par)
        return xt

    if shock[0] == 'exp':
        intx = _exp_intx_gen
        xt = _exp_xt_gen
    elif shock[0] == 'rett':
        intx = _rett_intx_gen
        xt = _rett_xt_gen
    elif shock[0] == 'mixed':
        shock = ['exp', 'rett'] 
        intx = _mix_intx_gen
        xt = _mix_xt_gen

    def _z(shock_par, t, nshock, intx):
        z_part = 0.       
        m = shock_par[0]
        p = shock_par[1]
        q = shock_par[2]

        for i in range(1, nshock+1):
            z_part += intx(t, i, shock_par)                        

        z_prime = z_part + t
        z = m * (1 - np.exp((-(p+q)*z_prime), dtype= np.float64)) / (1+(q/p)*np.exp((-(p+q)*z_prime), dtype=np.float64))
        return z

    def _zprime(shock_par, t, nshock, intx, xt):
        xi = 0        
        m = shock_par[0]
        p = shock_par[1]
        q = shock_par[2]

        for i in range(1, nshock+1):
            xi += xt(t, i, shock_par)

        x_t = 1 + xi
        z_t = _z(shock_par, t, nshock, intx)
        z_prime = (p + q * (z_t/m)) * (m - z_t) * x_t
        return z_prime

    def _residuals(shock_par, t, nshock, intx):
        return cumsum - _z(shock_par, t, nshock, intx)

    optim = opt.leastsq(func=_residuals, x0=prelimestimates, args=(t, nshock, intx), full_output=1)
    
    res = optim[2]['fvec']


    model = {
        'type' :"Generalized Bass Model",
        'functions' : [_z, _zprime],
        'data' : series,
        'method' : "nls",
        'alpha' : alpha,
        'optim' : optim,
        'residuals' : res,        
        'shocks' : [nshock, shock],
        'x_functions': [intx, xt],
        }

    if display:
        plot.dimora_plot(model, 'fit', oos)

    return model
