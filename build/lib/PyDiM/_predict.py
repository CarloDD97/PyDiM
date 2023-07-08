def dimora_predict(model, t = None):
    if model['type'] == "Standard Bass Model":
        m, p, q = model['optim'][0]
        fit = model['functions'][0](t, m, p, q)
        instant = model['functions'][1](t, m, p, q)

        return fit, instant

    elif model['type'] == "Generalized Bass Model":
        params = model['optim'][0]
        fit = model['functions'][0](params, t, model['shocks'][0], model['x_functions'][0])
        instant = model['functions'][1](params, t, model['shocks'][0], model['x_functions'][0], model['x_functions'][1])
        
        return fit, instant        

    elif model['type'] == "Guseo-Guidolin Model":
        params = model['optim'][0]
        fit = model['functions'][0](t, params, model['market_potential'])
        instant = model['functions'][1](t, params)
        
        return fit, instant 
    
    elif model['type'] == "UCRCD Model":
        raise KeyError("UCRCD does not allows for predictions in this implementation")

    else: 
        raise KeyError("Model type not recognized, be sure to input a PyDiM's model")