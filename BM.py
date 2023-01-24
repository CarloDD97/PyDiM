import numpy as np
# from scipy.optimize import least_squares
import scipy.stats as st
import scipy.optimize as opt
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def BM(series, method="nls", prelimestimates=[], oos=None, alpha=0.05, display=True):
    if len(prelimestimates) == 0:
        prelimestimates = np.array([np.sum(series)+100, 0.01, 0.1])
    if oos == None:
        oos = round(len(series)*0.25)

    x = np.ones(len(series))
    t = np.arange(1, len(series)+1, 1)
    aaa = np.arange(1, len(series)+1+oos, 1)
    s = series
    c = np.cumsum(s)

    ''' Da esportare in classe esterna'''
    def get_stats(ls):
        df = len(series) - len(prelimestimates)
        y_mean = np.mean(prelimestimates)
        TSS = np.sum((series-y_mean)**2)

        if method == 'nls':
            # Get the parameters
            parmEsts = np.round(ls[0], 10)
            # Get the Error variance and standard deviation
            RSS = np.sum(ls[2]['fvec']**2)            
            MSE = RSS / df
            RMSE = np.sqrt(MSE)
            # Get the covariance matrix
            cov = np.abs(MSE * ls[1])
            # Get parameter standard errors
            parmSE = np.diag(np.sqrt(cov))
            # Calculate the t-values
            tvals = parmEsts/parmSE
            # Get p-values
            pvals = (1 - st.t.cdf(np.abs(tvals), df))*2
            # Get biased variance (MLE) and calculate log-likehood
            s2b = RSS / len(series)
            logLik = -len(series)/2 * np.log(2*np.pi) - \
                len(series)/2 * np.log(s2b) - 1/(2*s2b) * RSS
            # Get R-squared
            R_squared = 1 - RSS/TSS
            # Get Lower & Upper bounds
            lower = [(parmEsts[j] + -1 * np.dot(st.norm.ppf(1-alpha/2), parmSE[j]))
                    for j in range(len(parmSE))]
            upper = [(parmEsts[j] + np.dot(st.norm.ppf(1-alpha/2), parmSE[j]))
                    for j in range(len(parmSE))]

        elif method == 'optim':
            parmEsts = ls.x
            parmSE = [0 for i in range(len(parmEsts))]
            lower = [0 for i in range(len(parmEsts))]
            upper = [0 for i in range(len(parmEsts))]
            tvals = [0 for i in range(len(parmEsts))]
            pvals = [0 for i in range(len(parmEsts))]
            RSS = np.round(ls.fun, 4)
            MSE = RSS / df
            RMSE = np.sqrt(MSE)
            R_squared = 1 - RSS/TSS

        stats = {
            'Residuals': ls[2]['fvec'],
            'Param': ['m', 'p', 'q'],
            'Estimate': parmEsts,
            'Std. Error': parmSE,
            'Lower': lower,
            'Upper': upper,
            't-value': tvals,
            'p-value': pvals,
            'RMSE': RMSE,
            'Df': df,
            'R-squared': R_squared,
            'RSS': RSS
        }
        return stats

    def assign_significance(p_val):
        sigs = []
        for val in p_val:
            if val>=0 and val<=0.001:
                sigs.append('***')
            elif val>0.001 and val<=0.01:
                sigs.append('**')
            elif val>0.01 and val<=0.05:
                sigs.append('*')
            elif val>0.05 and val<=0.1:
                sigs.append('.')
            elif val>0.1 and val<=1:
                sigs.append('')
            
        return sigs

    ''' Da esportare in classe esterna'''
    def print_summary(stats):
        print('')
        # print('Non-linear least squares')
        print('Residuals:')
        print('Min.     1st Qu.     Median     Mean    3rd Qu.   Max.')
        res_stat = st.describe(stats['Residuals'])
        print('% 5.3f   % 5.3f  % 5.3f  % 5.3f  % 5.3f  % 5.3f\n' % tuple([res_stat[1][0], \
            np.percentile(stats['Residuals'], 25), np.median(stats['Residuals']), res_stat[2], \
                np.percentile(stats['Residuals'], 75), res_stat[1][1]]))
        print('Coefficents:')
        print("   Estimate   Std. Error  Lower      Upper   p-value")
        significance = assign_significance(stats['p-value'])
        for i in range(len(prelimestimates)):
            print("% s % 5.4f   % 5.4f   % 5.4f   % 5.4f    % 5.4f % s" \
                % tuple([stats['Param'][i], stats['Estimate'][i], stats['Std. Error'][i], \
                stats['Lower'][i], stats['Upper'][i], stats['p-value'][i], significance[i]]))
        print('---')
        print("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n")
        print('Residual Standard Error: % 5.4f on %i degrees of freedom' %
              tuple([stats['RMSE'], stats['Df']]))
        print('Multiple R-squared: % 5.4f Residual sum of squares: % 5.4f' %
              tuple([np.sqrt(stats['R-squared']), stats['RSS']]))
        print('\n')

    def ff(t, m, p, q):
        return (m * (1 - np.exp(- np.multiply((p + q), t))) / (1 + q / p * np.exp(-np.multiply((p + q), t))))
    
    def ff1(par, t):
        return c - ff(t, par[0], par[1], par[2])

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
        aa = get_stats(ls)
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
        aa = get_stats(optim)

    print_summary(aa)

    if display:
        z = [ff(aaa[i], stime[0], stime[1], stime[2]) for i in range(len(aaa))]
        z_prime = np.gradient(z)
        plt.figure(figsize=(20,6))

        plt.subplot(121)
        plt.plot(t, c, 'o-' )
        plt.plot(aaa, z, 'r')
        plt.title('Cumulative')
        plt.legend(['Observed', 'Predicted'])

        plt.subplot(122)
        plt.plot(t, s, 'o-')
        plt.plot(aaa, z_prime, 'r')
        plt.title('Instantaneous')
        plt.legend(['Observed', 'Predicted'])
        plt.show()
