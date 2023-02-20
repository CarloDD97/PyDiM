import numpy as np
import scipy.stats as st
import scipy.optimize as opt

import __lib
from BM import BM

def GGM(series, prelimestimates = None,
    mt = "base", alpha = 0.05, oos=None,
    display = True):

    if prelimestimates == None:
        prelimestimates = np.concatenate(BM(series, display=0)['estimate'],[0.001, 0.1])
    
    if oos == None:
        oos = round(len(series)*0.25)

    print(prelimestimates)
    t = np.arange(1, len(series) + 1, 1)
    x_lim = np.arange(1, len(series) + 1 + oos, 1)
    cumsum = np.cumsum(series)

    def mt_func(t, K, pc, qc):
        mt = K*np.sqrt((1 - np.exp(-np.dot((pc + qc) , t))) / (1 + (qc / pc) * np.exp(-np.dot((pc + qc) , t))))
        return mt

    def ff_base_mt(t, par):
        K, ps, qs, pc, qc = tuple(par)
        z = mt_func(t, K, pc, qc) * (1 - np.exp(-np.dot((ps + qs) , t))) / (1 + (qs / ps) * np.exp(-np.dot((ps + qs) , t)))
        return z
        
    def ff_defined_mt(t, par):
        K, ps, qs = tuple(par[:3])
        z = K * mt(t) * (1 - np.exp(-np.dot((ps + qs) , t))) / (1 + (qs / ps) * np.exp(-np.dot((ps + qs) , t)))
        return z

    if type(mt) == 'function':
        ff = ff_defined_mt
    elif mt == 'base' or mt == None:
        ff = ff_base_mt
    else:
        print("'mt' parameter must be either a function or None/left blank")
        return 0

    def ff1(par, t):
        return cumsum - ff(t, par)

    def ff2(par, t):
        return ff(t, par)

    ls = opt.leastsq(func=ff1, x0=prelimestimates, args=(t), full_output=1)
    stime = ls[0]
    res = ls[2]['fvec']
    est = __lib.get_stats(ls, series, prelimestimates, method='nls', alpha=alpha, model='GGM')

    # __lib.print_summary(est)

    if display:
        z = [ff(x_lim[i], stime) for i in range(len(x_lim))]
        z_prime = np.gradient(z)
        __lib.plot_models(t, cumsum, x_lim, z, series, z_prime) 

    s_hat = ff2(stime, t)

    ao = {
        'model' : stime,
        'type' :("Guseo-Guidolin Model"),
        'estimate' : est,
        'fitted' : s_hat,
        'instantaneous' : z_prime

        # 'data' : cumsum
        }

    return ao

