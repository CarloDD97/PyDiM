import numpy as np
import scipy.optimize as opt
import pandas as pd

from PyDiM import BM, _lib as lib
from PyDiM import plot

def ucrcd(series1, series2, par = "double", prelimest_series1 = None, 
        prelimest_series2= None, alpha=0.05, delta=0.01, gamma=0.01, 
        display=True):
    """
    Function that apply the Generalized Bass _model to fit the data

    ...

    Attributes
    ----------
    series1, series2 : Series
        a data vector containing the series to be fitted
    par: string
        constraint used on the series:
            - 'double' will use two different contraints (e.g. delta and gamma)
            - 'unique' will use just one constraint two times (e.g. delta)
        (default is 'double')
    prelimest_series1, prelimest_series2: None or list
        vectors containing the starting values used by the model, they must be
        the coefficient estimations of the standard Bass model
        (default is 'None', in this case, the function will compute the standard
        Bass model on both the series by itself)
    alpha : float
        the confidence interval's significance level (default is 0.05)
    delta : float
        preliminary estimate of delta
    gamma : float
        preliminary estimate of gamma
    display = bool
        if 'True' allows to display the fitted values for cumulative and instantaneous 
        observed data and oss (default is True)
    stats = bool
        if 'True' allows to display the statistic summary of the _model (default is True)

    Returns
    -------
    ao = dict
        a dictionary containing all the usefull algorithm output:
        - 'type' : the name of the model (e.g. "UCRCD model"),
        - 'mode' : regime of the competition unique/double,
        - 'estimate' : dictionary whick includes alle the statistics displayed in the summary,
        - 'data' : tuple with the series given in input,
        - 'instantaneous': tuple containing the fitted instantaneous values,
        - 'alpha' : alpha value given as input.
    """
    series1 = series1.dropna()
    series2 = series2.dropna()

    # The below operation is not present in the R's implementation, altought
    # this case should be considered to avoid errors.
    if series1.iloc[1] == 0:
        series1 = lib.handle_zeros(series1)     
    if series2.iloc[1] == 0:
        series2 = lib.handle_zeros(series2)   

    if len(series2) > len(series1):
        series1, series2 = tuple([series2, series1])

    if prelimest_series1 == None:
        prelimest_series1 = BM.bm(series1, display=0)['optim'][0]
    if prelimest_series2 == None:
        prelimest_series2 = BM.bm(series2, display=0)['optim'][0]

    index1 = series1.index
    index2 = series2.index

    m1, p1c, q1c = tuple(prelimest_series1)
    m2, p2, q2 = tuple(prelimest_series2)

    c2i = len(series1) - len(series2)
    c2 = c2i
    m1s = m1

    tot = np.concatenate([series1, series2])
    data1 = np.cumsum(series1)
    data2 = np.cumsum(series2)
    end = len(series1)

    if c2i > 0 : 
        s1 = series1.iloc[:c2i]
        series1 = series1.iloc[c2i : end]
        t = np.arange(1, c2i+1, 1)
        s2 = np.zeros(c2i)
        Z1 = np.cumsum(s1)
        Z2 = np.cumsum(s2)

        print(series1.iloc[1])
        BMs1 = BM.bm(s1, display=0)
        
        m1, p1a, q1a = BMs1['optim'][0]
        
        pred_BM1 = BMs1['functions'][1](t, m1, p1a, q1a)

        o_bass = pd.DataFrame.from_dict({'t': t, 's1': s1, 's2': s2, 'Z1': Z1, 'Z2': Z2})
        p_bass = pd.DataFrame.from_dict({'t': t, 'pred_1': pred_BM1, 'pred_2': s2})

    ###############################################
    ################ Shared Params ################

    t = np.arange(c2, end, 1)
    Z1 = data1.iloc[c2:end]
    Z2 = data2

    data = pd.DataFrame.from_dict({'t': t, 's1': series1, 's2': series2, 'Z1': Z1, 'Z2': Z2})
    expdf = pd.melt(data.iloc[:,:3], id_vars='t', var_name="product", value_name="response")
    
    def _model(t, params, par):
        Z = Z1+Z2

        if par == 'unique':
            mc, p1c, p2, q1c, q2, delta = tuple(params) 
            z2 = (p2 + (q2 - delta) * Z1 / mc + q2 * Z2 / mc) * (mc - Z)
        elif par == 'double':
            mc, p1c, p2, q1c, q2, delta, gamma = tuple(params)
            z2 = (p2 + (q2 - gamma) * Z1 / mc + q2 * Z2 / mc) * (mc - Z)
        
        z1 = (p1c + (q1c + delta) * Z1 / mc + q1c * Z2 / mc) * (mc - Z) 

        return {'z1':z1, 'z2': z2, 't':t}

    def  _res_model(params, t, par):
        est = _model(t=t, params=params, par=par)
        data = pd.DataFrame.from_dict({'t':est['t'], 'z1_s': est['z1'], 'z2_s': est['z2']})
        preddf = pd.melt(data, id_vars='t', var_name='product', value_name='response')
        residuals = preddf['response'] - expdf['response']
        return residuals

    ###############################################  
    if par == 'unique':
        params = [(m1s+m2)*2, p1c, p2, q1c, q2, delta]

    if par == 'double':
        params = [(m1s+m2)*2, p1c, p2, q1c, q2, delta, gamma]

    # fit the parameters
    fitval1 = opt.leastsq(func= _res_model, x0=params, args=(t, par), maxfev=10000, full_output=1)
    
    df = len(series1) + len(series2) - len(params)    

    parest = fitval1[0]
    
    estimates = _model(t, parest, par)
    z_prime = pd.DataFrame.from_dict({'t':estimates['t'], 'pred_1': estimates['z1'], 'pred_2': estimates['z2']})

    data['t'] = np.arange(c2,end)
    z_prime['t'] = np.arange(c2,end)

    # computing the coefficients stats    
    stats2 = lib.get_stats(fitval1, tot, alpha, lib.set_params('UCRCD'), df=df)

    competition_stats = np.row_stack([[stats2['Estimate'][i], stats2['Std. Error'][i], stats2['Lower'][i],\
        stats2['Upper'][i], stats2['p-value'][i]] for i in range(len(stats2['Estimate']))])

    if c2i > 0:
        stats1 = lib.get_stats(BMs1['optim'], BMs1['data'], BMs1['alpha'], lib.set_params(BMs1['type']))    

        no_competition_stats = np.row_stack([[stats1['Estimate'][i], stats1['Std. Error'][i], stats1['Lower'][i],\
        stats1['Upper'][i], stats1['p-value'][i]] for i in range(len(stats1['Estimate']))])
        # Estimate = [m1, p1a, q1a, mc, p1c, q1c + delta, q1c, p2, q2, q2-gamma]
        Estimate1 = np.row_stack([no_competition_stats, competition_stats])
        param = lib.set_params('UCRCD')[:len(Estimate1)]

        data_o = pd.concat([o_bass, data], ignore_index=1)
        data = data_o

        data_p = pd.concat([p_bass, z_prime], ignore_index=1)
        z_prime = data_p

    if c2i == 0:
        # Estimate = [mc, p1c, q1c + delta, q1c, p2, q2, q2-gamma]
        Estimate1 = competition_stats
        param = lib.set_params('UCRCD')[3:3+len(Estimate1)]

    ### Final adjustments and statistics ###
    obs = pd.melt(data.iloc[:,:3], id_vars=['t'], var_name='product', value_name='consumption')
    pred = pd.melt(z_prime, id_vars=['t'], var_name='product', value_name='consumption')

    ss1 = obs['consumption'][0:end]
    ss2 = obs['consumption'][end+c2 : 2*end]

    pp1 = pred['consumption'][0:end]
    pp2 = pred['consumption'][end+c2 : 2*end]

    res = obs['consumption'] - pred['consumption']

    res1 = res[0:end]
    res2 = res[end+c2: 2*end]

    tss = np.sum((obs['consumption']- np.mean(obs['consumption']))**2)
    rss = np.sum(res**2)
    r_squared = 1 - rss/tss

    df = [len(ss1)-len(Estimate1[:,0]), len(ss2)-len(Estimate1[:,0])]
    MSE = [np.sum(res1**2) / df[0], np.sum(res2**2) / df[1]]
    RMSE = np.sqrt(MSE)

    est = {
        'Residuals': [res1, res2],
        'Param': param,
        'Estimate': Estimate1[:,0],
        'Std. Error': Estimate1[:,1],
        'Lower': Estimate1[:,2],
        'Upper': Estimate1[:,3],
        't-value': None,
        'p-value': Estimate1[:,4],
        'RMSE': RMSE,
        'Df': df,
        'R-squared': r_squared,
        'RSS': rss,
    }
   
    model = {
        'type' : "UCRCD Model",
        'functions' : _model,
        'mode' : par,
        'competition_start' : c2,
        'estimate' : est,
        'data': [ss1, ss2],
        'instantaneous': [pp1, pp2],
        'indexes' : [index1, index2],
        'alpha' : alpha,
        }

    if display:
        plot.dimora_plot(model, 'fit') 
        
    return model

