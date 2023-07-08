import numpy as np
import scipy.stats as st

from PyDiM import _lib as lib

def __assign_significance(p_val):
    sigs = []
    for val in p_val:
        if type(val) is str:
            sigs.append('')
        elif val>=0 and val<=0.001:
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

def __standard_summary(stats):
    print('')
    # print('Non-linear least squares')
    print('Residuals:')
    print('Min.         1st Qu.     Median    Mean       3rd Qu.    Max.')
    res_stat = st.describe(stats['Residuals'])
    print('% 5.6f   % 5.6f  % 5.6f  % 5.6f  % 5.6f  % 5.6f\n' % tuple([res_stat[1][0], \
        np.percentile(stats['Residuals'], 25), np.median(stats['Residuals']), res_stat[2], \
            np.percentile(stats['Residuals'], 75), res_stat[1][1]]))
    
    print('Coefficents:')
    print("    Estimate      Std. Error    Lower         Upper          p-value")
    significance = __assign_significance(stats['p-value'])

    if type(stats['Std. Error'][0]) is str:
        string = "% s % .4e    % s             % s             % s              % s % s"
    else:
        string = "% s % .4e   % .4e   % .4e   % .4e    % .4e % s"
        
    for i in range(len(stats['Param'])):
        print(string \
            % tuple([stats['Param'][i], stats['Estimate'][i], stats['Std. Error'][i], \
            stats['Lower'][i], stats['Upper'][i], stats['p-value'][i], significance[i]]))
    print('---')
    
    print("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n")
    print('Residual Standard Error: % 5.6f on %i degrees of freedom' %
            tuple([stats['RMSE'], stats['Df']]))
    print('Multiple R-squared: % 5.6f Residual sum of squares: % 5.6f' %
            tuple([np.sqrt(stats['R-squared']), stats['RSS']]))
    print('\n')
#########################################################

def __ucrcd_summary(stats):
    print('')
    # print('Non-linear least squares')
    for i in range(len(stats['Residuals'])):
        print('Residuals Series % i:' % (i+1))
        print('Min.         1st Qu.     Median      Mean    3rd Qu.     Max.')
        res_stat = st.describe(stats['Residuals'][i])
        print('% 5.6f   % 5.6f  % 5.6f  % 5.6f  % 5.6f  % 5.6f\n' % tuple([res_stat[1][0], \
            np.percentile(stats['Residuals'][i], 25), np.median(stats['Residuals'][i]), res_stat[2], \
                np.percentile(stats['Residuals'][i], 75), res_stat[1][1]]))
    

    print('Coefficents:')
    print("       Estimate      Std. Error    Lower         Upper          p-value")
    significance = __assign_significance(stats['p-value'])
    for i in range(len(stats['Param'])):
        print("% s % .4e   % .4e   % .4e   % .4e    % .4e % s" \
            % tuple([stats['Param'][i], stats['Estimate'][i], stats['Std. Error'][i], \
            stats['Lower'][i], stats['Upper'][i], stats['p-value'][i], significance[i]]))
    print('---')
    print("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n")

    for j in range(len(stats['RMSE'])):
        print('Residual Standard Error series % i: % 5.6f on %i degrees of freedom' %
                tuple([(j+1), stats['RMSE'][j], stats['Df'][j]]))
    print('Multiple R-squared: % 5.6f Residual sum of squares: % 5.6f' %
            tuple([stats['R-squared'], stats['RSS']]))
    print('\n')

def print_summary(model):

    if model['type'] == "UCRCD Model":
        stats = model['estimate']
        __ucrcd_summary(stats)
    else:
        params = lib.set_params(model['type'])
        stats = lib.get_stats(model['optim'], model['data'], model['alpha'], parameters=params)
        __standard_summary(stats)

    return True