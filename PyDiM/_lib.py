import numpy as np
import scipy.stats as st

def set_params(model):
    if model == "Standard Bass Model" or model == "Generalized Bass Model":
        parameters = ['m ', 'p ', 'q ', 'a1', 'b1', 'c1', 'a2', 'b2', 'c2', 'a3', 'b3', 'c3']
    elif model == "Guseo-Guidolin Model":
        parameters = ['K ', 'pc', 'qc', 'ps', 'qs']
    elif model == 'UCRCD':
        parameters = ['ma   ', 'p1a  ', 'q1a  ', 'mc   ', 'p1c  ', 'p2   ', 'q1c  ', 'q2   ', 'delta', 'gamma']
    else:
        parameters = None
    return parameters

# function added to handle series starting with 0s
def handle_zeros(series):
    i=0
    while series[i+1] == 0:
        i += 1
    return series[i:].reset_index(drop=True)

def get_stats(ls, series, alpha, parameters, method = 'nls', df = None):
    parameters = parameters
    
    if df != None: df = df
    else: df = len(series) - len(ls[0])
    # print(df)
    y_mean = np.mean(ls[0])
    TSS = np.sum((series-y_mean)**2)

    if method == "nls":
        # Get the parameters
        parmEsts = ls[0]
        # Get the Error variance and standard deviation
        res = ls[2]['fvec']
        RSS = np.sum(res**2)            
        MSE = RSS / df
        RMSE = np.sqrt(MSE)
        # Get the covariance matrix
        cov = np.abs(MSE * ls[1])
        # cov = np.abs(MSE * 1)
        # Get parameter standard errors
        parmSE = np.diag(np.sqrt(cov))
        # Calculate the t-values
        tvals = parmEsts/parmSE
        # Get p-values, 2-sided test
        pvals = (1 - st.t.cdf(np.abs(tvals), df))*2
        # pvals = 1 - st.t.cdf(np.abs(tvals), df)
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

    elif method == "optim":
        parmEsts = ls[0]
        parmSE = ['-' for i in range(len(parmEsts))]
        lower = ['-' for i in range(len(parmEsts))]
        upper = ['-' for i in range(len(parmEsts))]
        tvals = ['-' for i in range(len(parmEsts))]
        pvals = ['-' for i in range(len(parmEsts))]
        RSS = np.round(ls[1], 4)
        MSE = RSS / df
        RMSE = np.sqrt(MSE)
        R_squared = 1 - RSS/TSS

    stats = {
        'Residuals': res,
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