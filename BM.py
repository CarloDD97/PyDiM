import numpy as np
# from scipy.optimize import least_squares
import scipy.stats as st
import scipy.optimize as opt
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import __lib

def BM(series, method="nls", prelimestimates=[], oos=None, alpha=0.05, display=True, stats = True):
    if len(prelimestimates) == 0:
        prelimestimates = np.array([np.sum(series)+100, 0.01, 0.1])
    if oos == None:
        oos = round(len(series)*0.25)

    # x = np.ones(len(series))
    t = np.arange(1, len(series)+1, 1)
    x_lim = np.arange(1, len(series)+1+oos, 1)
    # s = series
    cumsum = np.cumsum(series)
    
    def ff(t, m, p, q):
            return (m * (1 - np.exp(- np.multiply((p + q), t))) / (1 + q / p * np.exp(-np.multiply((p + q), t))))
        
    def zprime(t, m, p, q):
            return m * (p*(p + q)**2 * np.exp((p + q) * t)) / ((p * np.exp((p + q) * t))+q)**2

    def residuals(par, t):
        return cumsum - ff(t, par[0], par[1], par[2])

    def ff2(par, t):
        return ff(t, par[0], par[1], par[2])

    def f(par, t):
        return np.sum(residuals(par, t)**2)

    # def zprime_return(par, t):
    #     m = par[0]
    #     p = par[1]
    #     q = par[2]
    #     return m * (1 - np.exp(-(p + q) * t)) / (1 + q / p * np.exp(-(p + q) *  t))
                                                                    
    if method == "nls":
        # ls = scipy.optimize.least_squares(fun=ff1, x0=par, args=(t), method='lm', max_nfev=200, verbose=1)
        optim = opt.leastsq(func=residuals, x0=prelimestimates, args=(t), full_output=1)
        stime = optim[0]
        res = optim[2]['fvec']
        est = __lib.get_stats(optim, series, prelimestimates, method, alpha, model='BM')
        # print_summary(aa)

    elif method == "optim":
        par = prelimestimates
        mass = np.sum(series) + 1000
        max = np.sum(series) + 10000
        l_bounds = [1e-10, 1e-10, 1e-10]
        u_bounds = [mass, 1, 1]
        bounds = list(zip(l_bounds, u_bounds))
        optim = opt.minimize(fun=f, x0=par, args=(t), bounds=bounds, method='L-BFGS-B')
        # stima_optim = opt.fmin_l_bfgs_b(func=f, x0=par, args=(t), bounds=bounds, approx_grad=True)#, factr=1e-08, pgtol=0, epsilon=1e-03, m=5, maxiter=100)
        # aa =stima_optim[0]

        stime = optim.x
        # res = residuals(stime, t)
        
        est = __lib.get_stats(optim, series, prelimestimates, alpha, model = 'BM', method=method)

    if stats:
        __lib.print_summary(est)

    if display:
        z = ff(x_lim, stime[0], stime[1], stime[2])
        z_prime = zprime(x_lim, stime[0], stime[1], stime[2]) 
        __lib.plot_models(t, cumsum, x_lim, z, series, z_prime) 

    z = ff2(stime, t)
    z_prime = zprime(t, stime[0], stime[1], stime[2])

    ao = {
        'model' : stime,
        'data' : series,
        'type' :"Standard Bass Model",
        'estimate' : est,
        'functions' : [ff2, zprime],
        'fitted' : z,
        'instantaneous' : z_prime
        }

    del(est)
    return ao