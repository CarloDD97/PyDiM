import numpy as np
import scipy.stats as st
import scipy.optimize as opt
import pandas as pd

import __lib
from BM import BM

def UCRCD(series1, series2, par = "double", prelimest_series1 = None, 
        prelimest_series2= None, alpha=0.05, delta=0.01, gamma=0.01, display=True):
    
    series1 = series1.reset_index(drop=True)
    series2 = series2.reset_index(drop=True)
    # The below operation is not present in the R's implementation, altought
    # this case should be considered to avoid errors.
    if len(series2) > len(series1):
        series1, series2 = tuple([series2, series1])

    if prelimest_series1 == None:
        prelimest_series1 = BM(series1, display=0)['estimate']
    if prelimest_series2 == None:
        prelimest_series2 = BM(series2, display=0)['estimate']

    m1, p1c, q1c = tuple(prelimest_series1['Estimate'][:])
    m2, p2, q2 = tuple(prelimest_series2['Estimate'][:])
    del(prelimest_series1, prelimest_series2)

    len_s1 = len(series1)
    len_s2 = len(series2)
    c2i = len_s1 - len_s2
    print(len_s1, len_s2, c2i)
    c2 = c2i
    
    tot = np.concatenate([series1.reset_index(drop=False), series2.reset_index(drop=False)])
    cumsum1 = np.cumsum(series1)
    cumsum2 = np.cumsum(series2)
    end = len(series1)

    if c2i > 0 :
        s1 = series1[:c2i].reset_index(drop=True)
        series1 = series1[c2 : end].reset_index(drop=True)
        t = np.arange(1, c2i+1, 1)
        s2 = np.zeros(c2i)
        Z1 = np.cumsum(s1)
        Z2 = np.cumsum(s2)

        BMs1 = BM(s1, display=0)
        model = BMs1['model']
        stats1 = BMs1['estimate']
        m1, p1a, q1a = tuple(stats1['Estimate'])
        
        m1_= [stats1['Estimate'][0], stats1['Std. Error'][0], stats1['Lower'][0], stats1['Upper'][0], stats1['p-value'][0]]
        p1a_ = [stats1['Estimate'][1], stats1['Std. Error'][1], stats1['Lower'][1], stats1['Upper'][1], stats1['p-value'][1]]
        q1a_ = [stats1['Estimate'][2], stats1['Std. Error'][2], stats1['Lower'][2], stats1['Upper'][2], stats1['p-value'][2]]

        pred_BM1 = BMs1['fitted'] - np.insert(BMs1['fitted'][:-1], 0, 0) # instantanous

        o_bass = pd.DataFrame.from_dict({'t': t, 's1': s1, 's2': s2, 'Z1': Z1, 'Z2': Z2})
        p_bass = pd.DataFrame.from_dict({'t': t, 'pred_1': pred_BM1, 'pred_2': s2})

    ###############################################
    ################ Shared Params ################

    t = np.arange(c2, end, 1)
    Z1 = np.cumsum(series1)
    Z2 = np.cumsum(series2)

    data = pd.DataFrame.from_dict({'t': t, 's1': series1, 's2': series2, 'Z1': Z1, 'Z2': Z2})
    expdf = pd.melt(data.iloc[:,:3], id_vars='t', var_name="product", value_name="response")

    
    def model(t, parms):
        Z = Z1+Z2
        if par == 'unique':
            mc, p1c, p2, q1c, q2, delta = tuple(parms)
            z1 = mc * (p1c + (q1c + delta) * Z1 / mc + q1c * Z2 / mc) * (1 - Z / mc)  
            z2 = mc * (p2 + (q2 - delta) * Z1 / mc + q2 * Z2 / mc) * (1 - Z /mc)
        elif par == 'double':
            mc, p1c, p2, q1c, q2, delta, gamma = tuple(parms)
            z1 = mc * (p1c + (q1c + delta) * Z1 / mc + q1c * Z2 / mc) * (1 - Z / mc)  
            z2 = mc * (p2 + (q2 - gamma) * Z1 / mc + q2 * Z2 / mc) * (1 - Z /mc)

        return {'z1':z1, 'z2': z2, 't':t}

    def res_model(parms, t):
        est = model(t=t, parms=parms)
        data = pd.DataFrame.from_dict({'t':est['t'], 'z1_s': est['z1'], 'z2_s': est['z2']})
        preddf = pd.melt(data, id_vars='t', var_name='product', value_name='response')
        residuals = preddf['response'] - expdf['response']
        return residuals

    ###############################################  
      
    if par == 'unique':
        parms = [(m1+m2)*2, p1c, p2, q1c, q2, delta]
        # print(parms)
        fitval1 = opt.leastsq(func=res_model, x0=parms, args=(t), maxfev=10000, full_output=1)
        
        df = len_s2*2 - len(parms)
        stats = __lib.get_stats(fitval1, tot, prelimestimates=parms, alpha=alpha, model='UCRCD', method='nls', df=df)
        # __lib.print_summary(stats)

        parest = stats['Estimate']
        mc, p1c, p2, q1c, q2, delta = tuple(stats['Estimate'])
        
        stack_stats = np.row_stack([[stats['Estimate'][i], stats['Std. Error'][i], stats['Lower'][i],\
                 stats['Upper'][i], stats['p-value'][i]] for i in range(len(stats['Estimate']))])

        if c2i > 0:
            Estimate = [m1, p1a, q1a, mc, p1c, q1c + delta, q1c, p2, q2, q2-delta]
            Estimate1 = np.row_stack([m1_, p1a_, q1a_, stack_stats])
            
            estimates = model(t, parest)
            z_prime = pd.DataFrame.from_dict({'t':estimates['t'], 'pred_1': estimates['z1'], 'pred_2': estimates['z2']})
            del(estimates)

            data['t'] = np.arange(c2,end)
            data_o = pd.concat([o_bass, data], ignore_index=1)
            data = data_o
            z_prime['t'] = np.arange(c2,end)
            data_p = pd.concat([p_bass, z_prime], ignore_index=1)
            z_prime = data_p

        if c2i == 0:
            Estimate = [mc, p1c, q1c + delta, q1c, p2, q2, q2-delta]
            Estimate1 = stack_stats
            
            estimates = model(t, parest)
            z_prime = pd.DataFrame.from_dict({'t':estimates['t'], 'pred_1': estimates['z1'], 'pred_2': estimates['z2']})
            del(estimates)

            data['t'] = np.arange(c2,end)
            z_prime['t'] = np.arange(c2,end)
            
        obs = pd.melt(data.iloc[:,:3], id_vars=['t'], var_name='product', value_name='consumption')
        pred = pd.melt(z_prime, id_vars=['t'], var_name='product', value_name='consumption')
        res = obs['consumption'] - pred['consumption']

    if par == 'double':
        parms = [(m1+m2)*2, p1c, p2, q1c, q2, delta, gamma]

        fitval1 = opt.leastsq(func=res_model, x0=parms, args=(t), maxfev=10000, full_output=1)
        
        df = len_s2*2 - len(parms)
        stats = __lib.get_stats(fitval1, tot, prelimestimates=parms, alpha=alpha, model='UCRCD', method='nls', df=df)
        # __lib.print_summary(stats)

        parest = stats['Estimate']
        mc, p1c, p2, q1c, q2, delta, gamma = tuple(stats['Estimate'])
        
        stack_stats = np.row_stack([[stats['Estimate'][i], stats['Std. Error'][i], stats['Lower'][i],\
                 stats['Upper'][i], stats['p-value'][i]] for i in range(len(stats['Estimate']))])

        if c2i > 0:
            Estimate = [m1, p1a, q1a, mc, p1c, q1c + delta, q1c, p2, q2, q2-gamma]
            Estimate1 = np.row_stack([m1_, p1a_, q1a_, stack_stats])
            
            estimates = model(t, parest)
            z_prime = pd.DataFrame.from_dict({'t':estimates['t'], 'pred_1': estimates['z1'], 'pred_2': estimates['z2']})
            del(estimates)

            data['t'] = np.arange(c2,end)
            data_o = pd.concat([o_bass, data], ignore_index=1)
            data = data_o
            del(data_o)
            z_prime['t'] = np.arange(c2,end)
            data_p = pd.concat([p_bass, z_prime], ignore_index=1)
            z_prime = data_p
            del(data_p)

        if c2i == 0:
            Estimate = [mc, p1c, q1c + delta, q1c, p2, q2, q2-gamma]
            Estimate1 = stack_stats
            
            estimates = model(t, parest)
            z_prime = pd.DataFrame.from_dict({'t':estimates['t'], 'pred_1': estimates['z1'], 'pred_2': estimates['z2']})
            del(estimates)

            data['t'] = np.arange(c2,end)
            z_prime['t'] = np.arange(c2,end)

        obs = pd.melt(data.iloc[:,:3], id_vars=['t'], var_name='product', value_name='consumption')
        pred = pd.melt(z_prime, id_vars=['t'], var_name='product', value_name='consumption')
        res = obs['consumption'] - pred['consumption']

    #######################################

    if c2i >= 0:
        data = [obs['consumption'][0:end], obs['consumption'][end+c2 : 2*end]]
        fitted = [pred['consumption'][0:end], pred['consumption'][end+c2 : 2*end]]
        resid = [res[0:end], res[end+c2: 2*end]]

    tss = np.sum((obs['consumption']- np.mean(obs['consumption']))**2)
    rss = np.sum(res**2)
    r_squared = 1 - rss/tss

    ss1 = obs['consumption'][0:end]
    ss2 = obs['consumption'][end+c2 : 2*end]
    cc1 = np.cumsum(ss1)
    cc2 = np.cumsum(ss2)

    pp1 = pred['consumption'][0:end]
    pp2 = pred['consumption'][end+c2 : 2*end]
    gg1 = np.cumsum(pp1)
    gg2 = np.cumsum(pp2)

    FITTED = [gg1, gg2]
    DATA = [cc1, cc2]
    RESIDUALS = [gg1-cc1, gg2-cc2]

    df = [len_s1-len(Estimate1[:,0]), len_s2-len(Estimate1[:,0])]
    MSE = [np.sum(resid[0]**2) / df[0], np.sum(resid[1]**2) / df[1]]
    RMSE = np.sqrt(MSE)

    t = np.arange(0, end)
    t2 = np.arange(c2, end)

    est = {
        'Residuals': resid,
        'Param': __lib.set_params('UCRCD'),
        'Estimate': Estimate1[:,0],
        'Std. Error': Estimate1[:,1],
        'Lower': Estimate1[:,2],
        'Upper': Estimate1[:,3],
        't-value': None,
        'p-value': Estimate1[:,4],
        'RMSE': RMSE,
        'Df': df,
        'R-squared': r_squared,
        'RSS': rss
    }
    
    __lib.print_ucrcd_summary(est)

    if display:
        __lib.plot_ucrcd(t, t2, cc1, cc2, ss1, ss2, gg1, gg2, pp1, pp2)

    ao = {
        'model' : Estimate1[:,0],
        'type' :"UCRCD Model",
        'estimate' : est,
        'fitted' : FITTED
        }
        
    return ao

