import os
import pickle
import numpy as np


def load(filename):
    """Reload compiled models for reuse."""
    print("Trying to load pickle in:")
    print(os.getcwd())
    return pickle.load(open(filename,'rb'))

def create_model_and_fit(DATA, name, sampling_iter, warmingUp, chains):
    print("get model and fit:"+os.getcwd())
    try:
        model = load(name)
    except:
        model = load("RE_approach.pic")
    print("sampling_iter", sampling_iter)
    print("sampling in: " + os.getcwd())
    print("warmup"+str(warmingUp))
    print("chains"+str(chains))

    #try:
    #    inv_metric = np.load("inv_metric_sampler.npy")
    #    control = {"inv_metric": inv_metric,
    #           "adapt_engaged": True
    #           }
    #    fit = model.sampling(DATA,
    #                     n_jobs = -1,
    #                     chains=chains,
    #                     thin=1,
    #                     warmup=warmingUp,#4000,
    #                     iter=int(sampling_iter),
    #                     verbose=True,
    #                     control = control,
    #                     refresh = 100,
    #                     test_grad = None)
    #except:
    fit = model.sampling(DATA,
                         n_jobs = -1,
                         chains=chains,
                         thin=2,
                         warmup=warmingUp,#4000,
                         iter=int(sampling_iter),
                         verbose=True,
                         refresh = 100,
                         test_grad = None)

    print("finished sampling")
    try:
        fit.summary()
    except:
        print("could not create fit summary")

    return fit, model

def fit_patch_fluo(DATA, name, sampling_iter, chains, warmup, seed, invMetric, stepsize, trained):
    print("sampling_iter", sampling_iter)
    print("sampling in: " + os.getcwd())
    print("warmup"+str(warmup))
    print("chains"+str(chains))
    print("get model and fit:"+os.getcwd())
    print(name)

    if trained == True:
        control = {"stepsize": stepsize,
               "inv_metric": invMetric,
               "adapt_engaged": True}


        try:
             model = load(name)
        except:
            model = load("RE_fluores.pic")
        print("Hallooooo")
        fit = model.sampling(DATA,
                             n_jobs=-1,
                             chains=chains,
                             thin=2,
                             warmup=warmup,  # 4000,
                             iter=sampling_iter,
                             verbose=True,
                             refresh=8,
                             test_grad=None,
                             seed = seed,
                             control = control)
    else:


        try:
            model = load(name)
        except:
            model = load("RE_fluores.pic")
        print("Hallooooo")
        fit = model.sampling(DATA,
                                 n_jobs=-1,
                                 chains=chains,
                                 thin=2,
                                 warmup=warmup,  # 4000,
                                 iter=sampling_iter,
                                 verbose=True,
                                 refresh=8,
                                 test_grad=None,
                                 seed=seed)


        #init_list = {"rates": [100,500,10,100,2543],
        #             "ratios":[0.5,0.5,0.5,0.5,0.0005],
        #             "N_ion_trace":[1000,1000,1000,1000,1000],
        #             "OpenVar": 0.01,
        #             "lambda_fluoresc" : 0.75,
        #             }
        #fit = model.optimizing(DATA,
        #                       iter = 1000,
        #                       init = init_list,
        #                      verbose= True,
        #                       as_vector =False )
    print(fit)

    print("finished sampling")


    try:
        fit.summary()
    except:
        print("could not create fit summary")

    return fit, model





def PredicPriorDistri(DATA, name, sampling_iter, chains, warmup, seed, invMetric, stepsize, trained, algo):
    print("sampling_iter", sampling_iter)
    print("sampling in: " + os.getcwd())
    print("warmup"+str(warmup))
    print("chains"+str(chains))
    print("get model and fit:"+os.getcwd())
    print(name)

    if trained == True:
        control = {"stepsize": stepsize,
               "inv_metric": invMetric,
               "adapt_engaged": True}


        try:
             model = load(name)
        except:
            model = load("RE_fluores.pic")
        print("Hallooooo")
        fit = model.sampling(DATA,
                             n_jobs=-1,
                             chains=chains,
                             thin=1,
                             warmup=warmup,  # 4000,
                             iter=sampling_iter,
                             verbose=True,
                             refresh=8,
                             test_grad=None,
                             seed = seed,
                             control = control,
                             algorithm=algo)
    else:


        try:
            model = load(name)
        except:
            model = load("RE_fluores.pic")
        print("Hallooooo")
        fit = model.sampling(DATA,
                                 n_jobs=-1,
                                 chains=chains,
                                 thin=1,
                                 warmup=warmup,  # 4000,
                                 iter=sampling_iter,
                                 verbose=True,
                                 refresh=8,
                                 test_grad=None,
                                 seed=seed,
                                 algorithm=algo)


        #init_list = {"rates": [100,500,10,100,2543],
        #             "ratios":[0.5,0.5,0.5,0.5,0.0005],
        #             "N_ion_trace":[1000,1000,1000,1000,1000],
        #             "OpenVar": 0.01,
        #             "lambda_fluoresc" : 0.75,
        #             }
        #fit = model.optimizing(DATA,
        #                       iter = 1000,
        #                       init = init_list,
        #                      verbose= True,
        #                       as_vector =False )
    print(fit)

    print("finished sampling")


    try:
        fit.summary()
    except:
        print("could not create fit summary")

    return fit, model