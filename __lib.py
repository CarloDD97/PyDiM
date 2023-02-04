import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

parameters = ['m ', 'p ', 'q ', 'a1', 'b1', 'c1', 'a2', 'b2', 'c2', 'a3', 'b3', 'c3']

def get_stats(ls, series, prelimestimates, method, alpha):
    df = len(series) - len(ls[0])
    # print(df)
    y_mean = np.mean(prelimestimates)
    TSS = np.sum((series-y_mean)**2)

    if method == 'nls':
        # Get the parameters
        parmEsts = ls[0]
        # Get the Error variance and standard deviation
        RSS = np.sum(ls[2]['fvec']**2)            
        MSE = RSS / df
        RMSE = np.sqrt(MSE)
        # Get the covariance matrix
        cov = np.abs(MSE * ls[1])
        # cov = np.abs(MSE * 1)
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
        'Param': parameters[:len(parmEsts)],
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
    print('Min.    1st Qu.   Median  Mean    3rd Qu.  Max.')
    res_stat = st.describe(stats['Residuals'])
    print('% 5.3f   % 5.3f  % 5.3f  % 5.3f  % 5.3f  % 5.3f\n' % tuple([res_stat[1][0], \
        np.percentile(stats['Residuals'], 25), np.median(stats['Residuals']), res_stat[2], \
            np.percentile(stats['Residuals'], 75), res_stat[1][1]]))
    print('Coefficents:')
    print("    Estimate      Std. Error    Lower         Upper          p-value")
    significance = assign_significance(stats['p-value'])
    for i in range(len(stats['Param'])):
        print("% s % .4e   % .4e   % .4e   % .4e    % .4e % s" \
            % tuple([stats['Param'][i], stats['Estimate'][i], stats['Std. Error'][i], \
            stats['Lower'][i], stats['Upper'][i], stats['p-value'][i], significance[i]]))
    print('---')
    print("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n")
    print('Residual Standard Error: % 5.4f on %i degrees of freedom' %
            tuple([stats['RMSE'], stats['Df']]))
    print('Multiple R-squared: % 5.5f Residual sum of squares: % 5.4f' %
            tuple([np.sqrt(stats['R-squared']), stats['RSS']]))
    print('\n')

def plot_models(t, cumsum, x_lim, z, series, z_prime):
        plt.figure(figsize=(20,6))

        plt.subplot(121)
        plt.plot(t, cumsum, 'o-' )
        plt.plot(x_lim, z, 'r')
        plt.xlabel('t')
        plt.ylabel('z(t)')
        plt.title('Cumulative')
        plt.legend(['Observed', 'Predicted'])

        plt.subplot(122)
        plt.plot(t, series, 'o-')
        plt.plot(x_lim, z_prime, 'r')
        plt.xlabel('t')
        plt.ylabel("z'(t)")
        plt.title('Instantaneous')
        plt.legend(['Observed', 'Predicted'])
        plt.show()