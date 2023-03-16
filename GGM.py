from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as st
import scipy.optimize as opt

import __lib
from BM import BM

def GGM(series, prelimestimates = None,
    mt = "base", alpha = 0.05, oos=None,
    display = True, stats = True):

    if prelimestimates == None:
        prelimestimates = np.concatenate(BM(series, display=0)['estimate'],[0.001, 0.1])
    
    if oos == None:
        oos = round(len(series)*0.25)

    print(prelimestimates)
    t = np.arange(1, len(series) + 1, 1)
    x_lim = np.arange(1, len(series) + 1 + oos, 1)
    cumsum = np.cumsum(series)

    def mt_func(t, pc, qc):
        mt = np.sqrt(np.abs((1 - np.exp(-(pc + qc) * t)) / (1 + (qc / pc) * np.exp(-(pc + qc) * t))))
        return mt

    # defined as z(t) on the book (pag. 65)
    def ff_base_mt(t, par, mt):
        K, ps, qs, pc, qc = tuple(par)
        z = K  * mt(t, pc, qc) * (1 - np.exp(-np.dot((ps + qs) , t))) / (1 + (qs / ps) * np.exp(-np.dot((ps + qs) , t)))
        return z
        
    def ff_defined_mt(t, par, mt):
        K, ps, qs = tuple(par[:3])
        z = K * mt(t) * (1 - np.exp(-np.dot((ps + qs) , t))) / (1 + (qs / ps) * np.exp(-np.dot((ps + qs) , t)))
        return z
    
    def zprime(t, par):
        K, ps, qs, pc, qc = tuple(par)
        F_t = (1 - np.exp(-np.dot((pc + qc) , t))) / (1 + (qc / pc) * np.exp(-np.dot((pc + qc) , t)))
        G_t = (1 - np.exp(-np.dot((ps + qs) , t))) / (1 + (qs / ps) * np.exp(-np.dot((ps + qs) , t)))
        k1_t = (1/2)* (1/np.sqrt(F_t)) * G_t * np.gradient(F_t)
        k2_t = np.sqrt(F_t) * np.gradient(G_t)

        return K*(k1_t + k2_t)

    if type(mt) == 'function':
        ff = ff_defined_mt
    elif mt == 'base' or mt == None:
        ff = ff_base_mt
        mt = mt_func
    else:
        print("'mt' parameter must be either a function or None/left blank")
        return 0

    def ff1(par, t, mt):
        return cumsum - ff(t, par, mt)

    def ff2(t, par, mt):
        return ff(t, par, mt)

    ls = opt.leastsq(func=ff1, x0=prelimestimates, args=(t, mt), full_output=1)
    stime = ls[0]
    res = ls[2]['fvec']
    est = __lib.get_stats(ls, series, prelimestimates, method='nls', alpha=alpha, model='GGM')

    if stats:
        __lib.print_summary(est)

    if display:
        z = ff2(x_lim, stime, mt)
        # z_prime = np.gradient(z)
        z_prime = zprime(x_lim, stime)
        __lib.plot_models(t, cumsum, x_lim, z, series, z_prime) 

    z = [ff2(t[i], stime, mt) for i in range(len(t))]
    z_prime = zprime(t, stime)

    ao = {
        'model' : stime, # K, ps, qs, pc, qc
        'data' : series,
        'type' :"Guseo-Guidolin Model",
        'market_potential': mt, # <- t, K, pc, qc
        'functions' : [ff, zprime], # <- t, stime[:], mt
        'estimate' : est,
        'fitted' : z,
        'instantaneous' : z_prime
        }

    return ao

