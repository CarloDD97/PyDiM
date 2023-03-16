import matplotlib.pyplot as plt
import __lib, predict
import numpy as np
import BM
from statsmodels.graphics.tsaplots import plot_acf
# from statsmodels.api import sm


def dimora_plot(model, type = 'all', oss=0):

    def plot(len_series, data, len_w_oss, new_pred, model_pred, title, ax, labels):
        ax.plot(len_series, data, 'k.-' )
        ax.plot(len_w_oss, new_pred, 'g')
        ax.plot(len_series, model_pred, 'r')
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_title(title)
        ax.legend(['Observed', 'Predicted+oos', 'Predicted'])
    
    def plot2(len_series, data, model_pred, title, ax, labels):
        ax.plot(len_series[0], data[0], 'k.-', len_series[1], data[1], 'g.-')
        ax.plot(len_series[0], model_pred[0], 'r', len_series[1], model_pred[1], 'b')
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_title(title)
        ax.legend(['Observed 1', 'Observed 2', 'Predicted 1', 'Predicted 2'])

    def plot_res(t, res, ax, title_res, title_acf):
        ax[0].stem(t, res, markerfmt='k.', basefmt='k-')
        ax[0].set_title(title_res)
        ax[0].set_xlabel('t')
        ax[0].set_ylabel('Residuals')

        plot_acf(res, title = title_acf, ax=ax[1], color= 'black', marker='.')
        ax[1].set_xlabel('Lags')
        ax[1].set_ylabel('ACF')
    
    if model['type'] != "UCRCD Model":
        product = 1
    else: product = 2

    if product == 1:
        t = np.arange(1, len(model['fitted'])+1, 1)
        cumsum = model['fitted']
        res = model['estimate']['Residuals']

        xmax = len(model['fitted']) + oss
        maxx = np.max([len(model['fitted']), xmax])
        xlim = np.arange(1, maxx+1, 1)

        z, z_prime = predict.dimora_predict(model, xlim)

        title_res = 'Residuals over time'
        title_acf = 'ACF Residual'

        if type == 'fit':
            fig, ax = plt.subplots(1,2)            
            plot(t, cumsum, xlim, z, model['fitted'], 'Cumulative', ax[0], ['t', 'z(t)'])
            plot(t, model['data'], xlim, z_prime, model['instantaneous'], 'Instantaneous', ax[1], ['t', "z'(t)"])

        elif type == 'res':
            fig, ax = plt.subplots(1,2)
            plot_res(t, res, ax, title_res, title_acf)

        else:
            fig, ax = plt.subplots(2,2)
            plot(t, cumsum, xlim, z, model['fitted'], 'Cumulative', ax[0,0], ['t', 'z(t)'])
            plot(t, model['data'], xlim, z_prime, model['instantaneous'], 'Instantaneous', ax[0,1], ['t', "z'(t)"])
            plot_res(t, res, ax[1], title_res, title_acf)

    elif product == 2:
        t = tuple([np.arange(1, len(model['fitted'][0])+1, 1), np.arange(1, len(model['fitted'][1])+1, 1)])
        cumsum = tuple([model['fitted'][0], model['fitted'][1]])
        z, z_prime = predict.dimora_predict(model)
        res = model['estimate']['Residuals']

        title_res = 'Residuals over time'
        title_acf = 'ACF Residual'

        if type == 'fit':
            fig, ax = plt.subplots(1,2)            
            plot2(t, cumsum, z, 'Cumulative', ax[0], ['t', 'z(t)'])
            plot2(t, model['data'], z_prime, 'Instantaneous', ax[1], ['t', "z'(t)"])

        elif type == 'res':
            fig, ax = plt.subplots(2,2)
            plot_res(t[0], res[0], ax[0], 'Residuals over time series 1', 'ACF Residual series 1')
            plot_res(t[1], res[1], ax[1], 'Residuals over time series 2', 'ACF Residual series 2')

        else:
            fig, ax = plt.subplots(3,2)
            plot2(t, cumsum, z, 'Cumulative', ax[0,0], ['t', 'z(t)'])
            plot2(t, model['data'], z_prime, 'Instantaneous', ax[0,1], ['t', "z'(t)"])

            plot_res(t[0], res[0], ax[1], 'Residuals over time series 1', 'ACF Residual series 1')
            plot_res(t[1], res[1], ax[2], 'Residuals over time series 2', 'ACF Residual series 2')

    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.6)
    plt.show()        