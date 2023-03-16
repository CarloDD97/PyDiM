

def dimora_predict(model, t = None):
    if model['type'] == 'Standard Bass Model':
        m, p, q = model['estimate']['Estimate']
        fit = model['functions'][0]([m,p,q], t)
        instant = model['functions'][1](t, m, p, q)

        return fit, instant

    elif model['type'] == 'Generalized Bass Model':
        params = model['estimate']['Estimate']
        fit = model['functions'][0](params, t, model['shocks'][0], model['x_functions'][0])
        instant = model['functions'][1](params, t, model['shocks'][0], model['x_functions'][0], model['x_functions'][1])
        
        return fit, instant        

    elif model['type'] == 'Guseo-Guidolin Model':
        params = model['estimate']['Estimate']
        fit = model['functions'][0](t, params, model['market_potential'])
        instant = model['functions'][1](t, params)
        
        return fit, instant 
    
    elif model['type'] == 'UCRCD Model':
        params = model['estimate']['Estimate']
        # fit = model['model'](t, params, model['par'])
        # instant = np.gradient(fit)
        fit = model['fitted']
        instant = model['instantaneous']

        return fit, instant

    else: return 0