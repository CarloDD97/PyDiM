import numpy as np
# from scipy.optimize import least_squares
import scipy.stats as st
import scipy.optimize as opt
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import __lib

def BM(series, method="nls", prelimestimates=[], oos=None, alpha=0.05, display=True):
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
    
    def ff1(par, t):
        return cumsum - ff(t, par[0], par[1], par[2])

    def ff2(par, t):
        return ff(t, par[0], par[1], par[2])

    def f(par, t):
        return np.sum(ff1(par, t)**2)

    def zprime(t, m, p, q):
        return m * (p + q * (ff(t, m, p, q) / m)) * (1 - (ff(t, m, p, q) / m))

    def zprime_return(par, t):
        m = par[1]
        p = par[2]
        q = par[3]
        return m * (1 - np.exp(-(p + q) * t)) / (1 + q / p * np.exp(-(p + q) *
                                                                    t))
    if method == "nls":
        # ls = scipy.optimize.least_squares(fun=ff1, x0=par, args=(t), method='lm', max_nfev=200, verbose=1)
        ls = opt.leastsq(func=ff1, x0=prelimestimates, args=(t), maxfev=200, full_output=1)
        stime = ls[0]
        res = ls[2]['fvec']
        est = __lib.get_stats(ls, series, prelimestimates, method, alpha)
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
        res = ff1(stime, t)
        est = __lib.get_stats(optim, series, prelimestimates, method, alpha)

    __lib.print_summary(est)

    if display:
        z = [ff(x_lim[i], stime[0], stime[1], stime[2]) for i in range(len(x_lim))]
        z_prime = np.gradient(z)
        __lib.plot_models(t, cumsum, x_lim, z, series, z_prime) 

    s_hat = ff2(stime, t)

    ao = {
        'model' : stime,
        'type' :"Standard Bass Model",
        'estimate' : est,
        # 'coefficients' : aa['Estimate'],
        # 'r_squared' : aa['R-squared'],
        # 'RSS' : aa['RSS'],
        # 'residuals' : aa['Residuals'],
        'fitted' : s_hat,
        'data' : cumsum
        }

    return ao