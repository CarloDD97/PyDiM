import BM
import numpy as np
import scipy.stats as st
import scipy.optimize as opt

import __lib

def GBM(series, shock=None, nshock=0,
        prelimestimates=None, alpha=0.05, oos=None,
        display=True):

    if shock == None or shock not in ["exp", "rett", "mixed"]:
        print("wrong value in 'shock' variable")
        return 0
    if prelimestimates == None:
        print("add values in 'prelimestimates")
        return 0
    if nshock <= 0:
        print('Number of shocks must be > 0')
        return 0
    if (nshock == 1 or nshock >= 3) and shock == 'mixed':
        print('Sorry but we have not implemented a model with % i mixed shocks' % nshock)
        return 0

    if type(shock) != list():
        shock = [shock]
    if oos == None:
        oos = round(len(series)*0.25)

    # bass = BM.BM(series, display=False)
    # bm = bass['estimate']['Estimate'] #se nshock è 0 torno questo -> [m, p, q]

    par = prelimestimates
    t = np.arange(1, len(series) + 1, 1)
    x_lim = np.arange(1, len(series) + 1 + oos, 1)
    # s = series
    cumsum = np.cumsum(series)
    shock_params = np.zeros(nshock)

    '''EXPONENTIAL SHOCKS GENERALIZED FUNCTIONS'''
    def exp_intx_gen(t, index, shock_par):
        a = shock_par[3*index]
        b = shock_par[3*index+1]
        c = shock_par[3*index+2]
        intx = c*(1/b)*(np.exp(np.dot(b,(t-a)))-1)*(t>=a)

        return intx

    def exp_xt_gen(t, index, shock_par):
        a = shock_par[3*index]
        b = shock_par[3*index+1]
        c = shock_par[3*index+2]
        xt = (c*np.exp(np.dot(b, (t-a))))*(t >= a)

        return 1 + xt # 1+ va tolto nella generalizzazione

    '''RECTANGULAR SHOCKS GENERALIZED FUNCTIONS'''
    def rett_intx_gen(t, index, shock_par):
        a = shock_par[3*index]
        b = shock_par[3*index+1]
        c = shock_par[3*index+2]
        intx = np.dot(c,(t-a))*(t>=a)*(t<=b) + c*(b-a)*(t>b)

        return intx

    def rett_xt_gen(t, index, shock_par):
        a = shock_par[3*index]
        b = shock_par[3*index+1]
        c = shock_par[3*index+2]
        xt = c*(t>=a)*(t<=b)

        return 1 + xt

    '''RECTANGULAR SHOCKS GENERALIZED FUNCTIONS'''
    # Qui è un po' rattoppato, faccio in modo che se vogliamo possiamo
    # adattarlo facilmente ad un ordine diverso di shock misti
    def mix_intx_gen(t, index, shock_par):
        if shock[index-1] == 'exp':
            intx = exp_intx_gen(t, index, shock_par)
        else:
            intx = rett_intx_gen(t, index, shock_par)
        return intx

    def mix_xt_gen(t, index, shock_par):
        if shock[index-1] == 'exp':
            xt = exp_intx_gen(t, index, shock_par)
        else:
            xt = rett_intx_gen(t, index, shock_par)
        return xt

    '''per generalizzare decommentare il for e la lista,
    sostituire intx con intx[index] "nshock" con "index"'''
    # for index
    # , shock in enumerate(shocks):
    # intx = None
    # for s in range(nshock):  # computing the shocks

    if shock[0] == 'exp':
        intx = exp_intx_gen
        xt = exp_xt_gen
    elif shock[0] == 'rett':
        intx = rett_intx_gen
        xt = rett_xt_gen
    elif shock[0] == 'mixed':
        shock = ['exp', 'rett'] 
        intx = mix_intx_gen
        xt = mix_xt_gen

    def ff(shock_par, t):
        z_part = 0        
        m = shock_par[0]
        p = shock_par[1]
        q = shock_par[2]

        for i in range(1, nshock+1):
            z_part += intx(t, i, shock_par)
            
        z_prime = t + z_part
        z = m * (1 - np.exp(-(p+q)*z_prime)) / (1+(q/p)*np.exp(-(p+q)*z_prime))
        return z

    def ff1(shock_par, t):
        return cumsum - ff(shock_par, t)

    def ff2(shock_par, t):
        return ff(shock_par, t)

    ls = opt.leastsq(func=ff1, x0=prelimestimates,
                     args=(t), full_output=1)
    stime = ls[0]
    res = ls[2]['fvec']
    est = __lib.get_stats(ls=ls, series=series, prelimestimates=list(prelimestimates[:3]), method='nls', alpha=alpha)

    __lib.print_summary(est)

    if display:
        z = [ff(stime, x_lim[i]) for i in range(len(x_lim))]
        z_prime = np.gradient(z)
        __lib.plot_models(t, cumsum, x_lim, z, series, z_prime) 

    s_hat = ff2(stime, t)

    ao = {
        'model' : stime,
        'type' :"Generalized Bass Model",
        'estimate' : est,
        'fitted' : s_hat,
        'data' : cumsum
        }

    return ao
