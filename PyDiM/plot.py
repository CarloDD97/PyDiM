import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf

import datetime
import time
from dateutil.relativedelta import relativedelta
from collections.abc import Iterable

from PyDiM import _predict as predict

def dimora_plot(model, plot_type = 'all', oos=0, legend=None, index_as_label=False):
    """
    Function used to plot the data
    ...

    Attributes
    ----------
    model : dict
        estmations returned by the diffusion models implemented
    plot_type : str
        string indicating the desired plots:
            - 'fit' : fitted values
            - 'res' : residuals
            - 'all' : both fitted and residual plots
    oos : int
        the number of predictions after the last observed value in the series, 
        if not specified (default will be set as the 25% of series length)

    """

    def _plot(len_series, data, len_w_oss, new_pred, model_pred, title, ax, labels, legend):
        ax.plot(len_series, data, 'k.-', linewidth = .7, markersize=2.)
        ax.plot(len_w_oss, new_pred, 'g--')
        ax.plot(len_series, model_pred, 'r')
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_title(title)
        ax.legend(legend)
    
    def _plot2(len_series, data, model_pred, title, ax, labels, legend):
        ax.plot(len_series[0], data[0], 'k.-', linewidth = .7, markersize=2.)
        ax.plot(len_series[1], data[1], '*--', color='gray', linewidth = .7, markersize=2.)
        ax.plot(len_series[0], model_pred[0], 'r', len_series[1], model_pred[1], 'g')
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_title(title)
        ax.legend(legend)

    def _plot_res(t, res, ax, title_res, title_acf):
        ax[0].stem(t, res, markerfmt='k.', basefmt='k-')
        ax[0].set_title(title_res)
        ax[0].set_xlabel('t')
        ax[0].set_ylabel('Residuals')

        plot_acf(res, title = title_acf, ax=ax[1], color= 'black', marker='.')
        ax[1].set_xlabel('Lags')
        ax[1].set_ylabel('ACF')

    plt.rcParams['figure.figsize'] = [15, 10]
    
    if model['type'] == "UCRCD Model":
        product = 2
    else: product = 1

    if product == 1:
        if legend == None:
            legend = ['Observed', f'oos = {oos}', 'Predicted']
        elif type(legend) != str and len(legend) > 1:
            print("Warning: more than one legend passed for univariate series plot, first one will be used instead")
            l_str = f"{legend[0]}, oos = {oos}, Predicted"
            legend = l_str.split(", ")
        else:
            if type(legend) == str:
                l_str = f"{legend}, oos = {oos}, Predicted"
                legend = l_str.split(", ")
            elif isinstance(legend, Iterable):
                l_str = f"{legend[0]}, oos = {oos}, Predicted"
                legend = l_str.split(", ")
            else:
                raise TypeError("Only strings or Itable objects can be used as legend")
        
        if oos == None:
            oos = round(len(model['data'])*0.25)

        t = np.arange(1, len(model['data'])+1, 1)

        cumsum = np.cumsum(model['data'])
        res = model['residuals']

        xlim = np.arange(1, len(model['data']) + oos +1, 1)

        start_time = time.time()

        z_fit, z_prime_fit = predict.dimora_predict(model, t)
        z, z_prime = predict.dimora_predict(model, xlim)

        predictions_time = time.time()

        title_res = 'Residuals over time'
        title_acf = 'ACF Residual'

        print("-----  Predict Total %s seconds ---" % (predictions_time - start_time)) 

        start_time = time.time()

        if not index_as_label:
            ind = t
            xlim = np.arange(1, len(model['data']) + oos +1, 1)
        else:
            ind = model['data'].index   
            if (type(ind) is pd.DatetimeIndex) or (type(ind) is pd.Timestamp): #if indexes are timeseries:
                freq = pd.infer_freq(ind) #infer the frequence
                if freq == 'S': #frequence in seconds
                    xlim = ind.union([ind[-1] + datetime.timedelta(seconds=i) for i in range (1,oos+1)])
                    legend[1] += ' seconds'
                elif freq == 'T': #frequence in minutes
                    xlim = ind.union([ind[-1] + datetime.timedelta(minutes=i) for i in range (1,oos+1)])
                    legend[1] += ' minutes'
                elif freq == 'H': #hourly frequence
                    xlim = ind.union([ind[-1] + datetime.timedelta(hours=i) for i in range (1,oos+1)])
                    legend[1] += ' hours'
                elif freq == 'D': #daily frequence
                    xlim = ind.union([ind[-1] + datetime.timedelta(days=i) for i in range (1,oos+1)])
                    legend[1] += ' days'
                elif freq == 'W': #weekly frequence
                    xlim = ind.union([ind[-1] + datetime.timedelta(weeks=i) for i in range (1,oos+1)])
                    legend[1] += ' weeks'
                elif freq == 'MS': #monthly frequence
                    xlim = ind.union([ind[-1] + relativedelta(months=i) for i in range (1,oos+1)])
                    legend[1] += ' months'
                elif freq[:2] == 'AS': #yearly frequence
                    xlim = ind.union([ind[-1] + relativedelta(years=i) for i in range (1,oos+1)])
                    legend[1] += ' years'
                else:
                    print('index frequency not recognized, switching to sequential step-size')
                    ind = t
                    xlim = np.arange(1, len(model['data']) + oos +1, 1)
            else:
                # useful for sequential indexes with step-size > 1
                xlim = ind.union(np.arange(ind[-1], ind[-1]+oos+1, (ind[-1]-ind[-2]))) 

        if plot_type == 'fit':
            fig, ax = plt.subplots(1,2)          
            _plot(ind, cumsum, xlim, z, z_fit, 'Cumulative', ax[0], ['', 'z(t)'], legend)
            _plot(ind, model['data'], xlim, z_prime, z_prime_fit, 'Instantaneous', ax[1], ['t', "z'(t)"], legend)

        elif plot_type == 'res':
            fig, ax = plt.subplots(1,2)
            _plot_res(t, res, ax, title_res, title_acf)

        else:
            fig, ax = plt.subplots(2,2)
            _plot(ind, cumsum, xlim, z, z_fit, 'Cumulative', ax[0,0], ['Years', 'z(t)'], legend)
            _plot(ind, model['data'], xlim, z_prime, z_prime_fit, 'Instantaneous', ax[0,1], ['', "z'(t)"], legend)
            _plot_res(t, res, ax[1], title_res, title_acf)

    elif product == 2:
        if legend == None:
            legend = ['Observed 1', 'Observed 2', 'Predicted 1', 'Predicted 2']
        elif len(legend) > 2:
            print("Warning: more than 2 legend items passed for UCRCD plot, first two will be used instead")
            l_str = f"{legend[0]}, {legend[1]}, Predicted {legend[0]}, Predicted {legend[1]}"
            legend = l_str.split(", ")
        else:
            l_str = f"{legend[0]}, {legend[1]}, Predicted {legend[0]}, Predicted {legend[1]}"
            legend = l_str.split(", ")

        l1 = len(model['data'][0])
        l2 = len(model['data'][1])
        t = tuple([np.arange(0 , l1), np.arange(l1-l2, l1)])

        instant_data = model['data']
        cumsum_data = tuple([np.cumsum(instant_data[0]), np.cumsum(instant_data[1])])
        
        z_prime = model['instantaneous']
        z = tuple([np.cumsum(instant_data[0]), np.cumsum(instant_data[1])])

        res = model['estimate']['Residuals']

        if not index_as_label:
            ind = t
        else:
            ind = tuple([model['indexes'][0], model['indexes'][1]])

        if plot_type == 'fit':
            fig, ax = plt.subplots(1,2)    
            _plot2(ind, cumsum_data, z, 'Cumulative', ax[0], ['', 'z(t)'], legend=legend)
            _plot2(ind, instant_data, z_prime, 'Instantaneous', ax[1], ['', "z'(t)"], legend=legend)

        elif plot_type == 'res':
            fig, ax = plt.subplots(2,2)
            _plot_res(t[0], res[0], ax[0], 'Residuals over time series 1', 'ACF Residual series 1')
            _plot_res(t[1], res[1], ax[1], 'Residuals over time series 2', 'ACF Residual series 2')

        else:
            fig, ax = plt.subplots(3,2)
            _plot2(ind, cumsum_data, z, 'Cumulative', ax[0,0], ['', 'z(t)'], legend=legend)
            _plot2(ind, instant_data, z_prime, 'Instantaneous', ax[0,1], ['', "z'(t)"], legend=legend)

            _plot_res(t[0], res[0], ax[1], 'Residuals over time series 1', 'ACF Residual series 1')
            _plot_res(t[1], res[1], ax[2], 'Residuals over time series 2', 'ACF Residual series 2')

    plt.subplots_adjust(left=0.06,
                    bottom=0.08,
                    right=0.98,
                    top=0.85,
                    wspace=0.2,
                    hspace=0.4)

    if model['type'] == "Generalized Bass Model":
        specs = f" with {model['shocks'][0]} {model['shocks'][1][0]} shock(s)"
    else:
        specs = ""

    plt.suptitle(model['type']+specs)
    plt.show()
 
    return True      