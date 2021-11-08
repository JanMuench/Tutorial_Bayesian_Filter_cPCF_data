import os
import numpy as np
import pandas as pd
import pickle
import xarray as xr

def save_data_CCO(fit, folder, data_start, data_dec):
    print(fit)

    if not os.path.exists(folder):
        os.makedirs(folder)
    os.chdir(folder)
    print(os.getcwd())



    np.save("data_start", data_start)
    np.save("data_dec", data_dec)


    try:
        sampling_data = pd.DataFrame(["rates","ratio"],fit.extract(permuted=True))
        sampling_data.to_csv("samples")
        print("success in saveing samples from the posterior")
    except:
        print("cannot make a pandas data frame")

    try:
        sampling_data.to_csv("sampling_daten")
    except:
        print("could not save fit_data")


    equi_values = fit.extract("equi_values", permuted=True)["equi_values"]
    theta = fit.extract("rates", permuted=True)
    ratio = fit.extract("ratio", permuted=True)

    lp__ = fit.extract("lp__", permuted=True)["lp__"]
    lp__ = pd.DataFrame(data=lp__)


    lp__.to_csv("lp__")

    occupat = fit.extract("occupat", permuted=True)["occupat"]
    try:
        latent_time = fit.extract("LATENT_TIME", permuted = True)["LATENT_TIME"]
        np.save("latent_time", np.array(latent_time))
    except:
        print("LATENT TIME doesn t exist")

    try:
        latent_time_decay = fit.extract("LATENT_TIME_DECAY", permuted=True)["LATENT_TIME_DECAY"]
        np.save("latent_time_decay", np.array(latent_time_decay))
    except:
        print("LATENT TIME doesn t exist")

    try:
        occupat_dec = fit.extract("occupat_decay", permuted=True)["occupat_decay"]
        np.save("occupat_dec2", np.array(occupat_dec))
    except:
        print("occupat_decay doesn t exist")



    #mu = fit.extract("mu", permuted = True)["mu"]
    #np.save("mu", np.array(mu))

    np.save("equi_values2", np.array(equi_values))
    np.save("occupat2",np.array(occupat))


    log_lik_t = fit.extract("log_lik_t", permuted=True)["log_lik_t"]
    log_lik_h = fit.extract("log_lik_h", permuted=True)["log_lik_h"]

    np.save("log_lik_t2", np.array(log_lik_t))
    np.save("log_lik_h2", np.array(log_lik_h))



    print(occupat)

    column_names = list()
    N_free_param = 3
    for id in range(1,N_free_param):
        column_names.append("theta["+str(id)+"]")
    try:
        theta = pd.DataFrame(data = theta["rates"], columns =column_names)
        theta.to_csv("test")
    except:
        print("could not save theta")


    for id in range(1,N_free_param):
        column_names.append("rates["+str(id)+"]")

    try:
        ratio = pd.DataFrame(data=ratio["ratio"])
    except:
        print("ratio data frame ratio")
    try:
        ratio.to_csv("ratio")
    except:
        print("could not save")



def save_data_fluorescenc(fit, data_start, data_dec, N_free_param, execution_time, chains, sampling_iter, seed):
    #if not os.path.exists(folder):
    #    os.makedirs(folder)
    #os.chdir(folder)


    try:
        LogLikeLihood = fit.extract("LogLikeLihood", permuted=True)["LogLikeLihood"]
        np.save("marginalLikelihood", np.array(LogLikeLihood))
    except:
        pass



    try:
        stepsize = fit.get_stepsize()[0]
        print("step size"+str(stepsize))
        # by default .get_inv_metric returns a list
        inv_metric = fit.get_inv_metric(as_dict=True)[0]
        init_last_pos = fit.get_last_position()[0]
        last = pd.DataFrame.from_dict(init_last_pos, orient='index')
        pd.to_pickle(last, "last_param_position")

        np.save("inv_metric_sampler", inv_metric)
        np.save("seed", seed)
        np.save("setp_size", stepsize)
    except:
        print("could not save control params")
        pass



    print("saving in: "+os.getcwd())

    np.save("data_start", data_start)
    np.save("data_dec", data_dec)

    exec_time = pd.DataFrame({"exec_time": execution_time,"chains": chains, "samplin_iter":sampling_iter}, index = [0])
    exec_time.to_csv("execution_time_in_seconds")

    #try:
    #    sampling_data = pd.DataFrame(fit.extract(["rates","ratio", ], permuted=True))
    #except:
    #    print("cannot make a pandas data frame")

    #try:
    #    sampling_data.to_csv("sampling_daten")
    #except:
    #    print("could not save fit_data")



    try:
        param_likelihood_hold= fit.extract("param_likelihood_start_hold", permuted=True)["param_likelihood_start_hold"]
        print(param_likelihood_hold)
        param_likelihood_hold = np.swapaxes(param_likelihood_hold, 0,1)
        print("param_like.shape: " + str(param_likelihood_hold.shape))
        param_hold = xr.DataArray(data=param_likelihood_hold[:, :, :, :, :],
                             dims=("N_conc_time_series", "samples_posterior", "signal_type",
                                   "mean_and_correlations", "data_point"),
                             coords={
                                 # "N_conc_time_series":["0.0625", "0.125", "0.25", "0.5", "1", "2","4","8","16","64"],
                                 "signal_type": ["fluores", "current"],
                                 "mean_and_correlations": ["mean", "corr_1", "corr_2"]})
        param_hold.to_netcdf("param_likelihood_start_hold")

        param_likelihood_decay_hold = np.array(
            fit.extract("param_likelihood_decay_hold", permuted=True)["param_likelihood_decay_hold"])
        param_likelihood_decay_hold = np.swapaxes(param_likelihood_decay_hold, 0, 1)
        param_hold = xr.DataArray(data=param_likelihood_decay_hold[:, :, :, :, :],
                             dims=("N_conc_time_series", "samples_posterior", "signal_type",
                                   "mean_and_correlations", "data_point"),
                             coords={
                                 # "N_conc_time_series": ["0.0625", "0.125", "0.25", "0.5", "1", "2", "4", "8", "16", "64"],
                                 "signal_type": ["fluores", "current"],
                                 "mean_and_correlations": ["mean", "corr_1", "corr_2"]})
        param_hold.to_netcdf("param_likelihood_decay_hold")

        SummandsLogLikTraces_hold = fit.extract("SummandsLogLikTraces_hold", permuted=True)["SummandsLogLikTraces_hold"]
        np.save("SummandsLogLikTraces_hold",SummandsLogLikTraces_hold)

    except:
        print("no hold out set")



    try:
        param_likelihood= fit.extract("param_likelihood_start", permuted=True)["param_likelihood_start"]
        print(param_likelihood)
        param_likelihood = np.swapaxes(param_likelihood, 0,1)
        print("param_like.shape: " + str(param_likelihood.shape))
    except:
        pass


    major_axis = list()

    for i in range( 1 , 21):
        major_axis.append(str(i))

    try:
        param = xr.DataArray(data=param_likelihood[:,:,:,:,:],
                    dims= ("N_conc_time_series","samples_posterior","signal_type",
                           "mean_and_correlations","data_point"),
                         coords={#"N_conc_time_series":["0.0625", "0.125", "0.25", "0.5", "1", "2","4","8","16","64"],
                                 "signal_type": ["fluores", "current"],
                                 "mean_and_correlations":["mean", "corr_1", "corr_2"]})
        param.to_netcdf("param_likelihood_start")



        param_likelihood_decay = np.array(fit.extract("param_likelihood_decay", permuted=True)["param_likelihood_decay"])
        param_likelihood_decay = np.swapaxes(param_likelihood_decay, 0, 1)
        param = xr.DataArray(data=param_likelihood_decay[:, :, :, :,:],
                         dims=("N_conc_time_series", "samples_posterior", "signal_type",
                               "mean_and_correlations", "data_point"),
                         coords={
                             #"N_conc_time_series": ["0.0625", "0.125", "0.25", "0.5", "1", "2", "4", "8", "16", "64"],
                             "signal_type": ["fluores", "current"],
                             "mean_and_correlations": ["mean", "corr_1", "corr_2"]})
        param.to_netcdf("param_likelihood_decay")
    except:
        print("likelihood wasnt saved")
        pass

    SummandsLogLikTraces = fit.extract("SummandsLogLikTraces", permuted=True)["SummandsLogLikTraces"]
    np.save("SummandsLogLikTraces",SummandsLogLikTraces)



    ## Splitted Kalman filter stuff
    try:
        param = xr.DataArray(data=param_likelihood[:,:, :, :, :, :],
                             dims=("N_conc_time_series","Splits", "samples_posterior", "signal_type",
                                   "mean_and_correlations", "data_point"),
                             coords={
                                 # "N_conc_time_series":["0.0625", "0.125", "0.25", "0.5", "1", "2","4","8","16","64"],
                                 "signal_type": ["fluores", "current"],
                                 "mean_and_correlations": ["mean", "corr_1", "corr_2"]})
        param.to_netcdf("param_likelihood_start")

        param_likelihood_decay = np.array(
            fit.extract("param_likelihood_decay", permuted=True)["param_likelihood_decay"])
        param_likelihood_decay = np.swapaxes(param_likelihood_decay, 0, 1)
        param = xr.DataArray(data=param_likelihood_decay[:,:, :, :, :, :],
                             dims=("N_conc_time_series","Splits", "samples_posterior", "signal_type",
                                   "mean_and_correlations", "data_point"),
                             coords={
                                 # "N_conc_time_series": ["0.0625", "0.125", "0.25", "0.5", "1", "2", "4", "8", "16", "64"],
                                 "signal_type": ["fluores", "current"],
                                 "mean_and_correlations": ["mean", "corr_1", "corr_2"]})
        param.to_netcdf("param_likelihood_decay")
    except:
        print("likelihood wasnt saved")
        pass



    try:
        backround_sigma = np.array(fit.extract("var_exp", permuted=True)["var_exp"])
        np.save("measurement_sigma",np.array(backround_sigma))
    except:
        print("could save backround noise")
    try:
        N_traces = fit.extract("N_ion_trace", permuted=True)["N_ion_trace"]
        np.save("N_traces",np.array(N_traces))
    except:
        print("N_traces param to fit")

    try:
        lp__ = fit.extract("lp__", permuted=True)["lp__"]
        lp__ = pd.DataFrame(data=lp__)
        lp__.to_csv("lp__")
    except:
        print("lp_ saving doesn t work")

    try:
        OpenVar = fit.extract("OpenVar", permuted = True)["OpenVar"]
        np.save("var_open", np.array(OpenVar ))
    except:
        print("var_open  doesn t exist")




    try:
        latent_time = fit.extract("LATENT_TIME", permuted = True)["LATENT_TIME"]
        np.save("latent_time", np.array(latent_time))
    except:
        print("LATENT TIME doesn t exist")

    try:
        latent_time_decay = fit.extract("LATENT_TIME_DECAY", permuted=True)["LATENT_TIME_DECAY"]
        np.save("latent_time_decay", np.array(latent_time_decay))
    except:
        print("LATENT TIME doesn t exist")

    try:
        occupat_dec = fit.extract("occupat_decay", permuted=True)["occupat_decay"]
        np.save("occupat_dec2", np.array(occupat_dec))
    except:
        print("occupat_decay doesn t exist")



    #mu = fit.extract("mu", permuted = True)["mu"]
    #np.save("mu", np.array(mu))
    try:
        equi_values = fit.extract("equi_values", permuted=True)["equi_values"]
        np.save("equi_values2", np.array(equi_values))
    except:
        print("could not open equi_values")

    try:
        occupat = fit.extract("occupat", permuted=True)["occupat"]
        print(occupat)
        np.save("occupat2",np.array(occupat))
    except:
        print("could not save occupat")

    try:
        log_lik_t = fit.extract("log_lik_t", permuted=True)["log_lik_t"]
        np.save("log_lik_t2", np.array(log_lik_t))
    except:
        print("could not save log_lik_t")

    try:
        log_lik_h = fit.extract("log_lik_h", permuted=True)["log_lik_h"]
        np.save("log_lik_h2", np.array(log_lik_h))
    except:
        print("cold not save log_lik_h")




    column_names = list()
    for id in range(1,np.int(N_free_param/2+1)):
        column_names.append("theta["+str(id)+"]")

    try:
        i_single =fit.extract("i_single_channel", permuted = True)["i_single_channel"]
        np.save("i_single", np.array(i_single))
    except:
        print("i_single problems")



    theta = fit.extract("rates", permuted=True)
    theta = pd.DataFrame(data = theta["rates"], columns =column_names)
    theta.to_csv("test")


    for id in range(1,np.int(N_free_param/2+1)):
        column_names.append("rates["+str(id)+"]")

    try:
        ratio = fit.extract("ratio", permuted=True)
        ratio = pd.DataFrame(data=ratio["ratio"])
    except:
        print("ratio to data frame ratio did not work")

    try:
        ratio.to_csv("ratio")
    except:
        print("could not save ratio")

    try:
        lamb = fit.extract("lambda_fluoresc", permuted=True)
        lamb = pd.DataFrame(data=lamb["lambda_fluoresc"])
    except:
        print("ratio to data frame ratio did not work")

    try:
        lamb.to_csv("lambda_fluoresc")
    except:
        print("could not save ratio")

    try:
        var_fluoresc = fit.extract("var_fluoresc", permuted=True)
        var_fluoresc = pd.DataFrame(data=var_fluoresc["var_fluoresc"])
        var_fluoresc.to_csv("var_fluoresc")
    except Exception as e:
        print(e)
        print("could not save var_fluorescs")



def save_data_new(fit, data_start, data_dec,dataStartHold, dataDecHold, N_free_param, execution_time,seed):
        try:
            stepsize = fit.get_stepsize()
            print("step size" + str(stepsize))[0]
            # by default .get_inv_metric returns a list
            inv_metric = fit.get_inv_metric(as_dict=True)[0]
            init = fit.get_last_position()[0]

            # increment seed by 1

            control = {"stepsize": stepsize,
                   "inv_metric": inv_metric,
                   "adapt_engaged": False
                   }
            np.save("inv_metric_sampler", inv_metric)
            np.save("last_param_position", init)
            np.save("seed", seed)
            np.save("setp_size", stepsize)
        except:
            print("could not save control params")
            pass



        # if not os.path.exists(folder):
        #    os.makedirs(folder)
        # os.chdir(folder)
        print("saving in: " + os.getcwd())

        np.save("data_start", data_start)
        np.save("data_dec", data_dec)
        np.save("data_start_hold", dataStartHold)
        np.save("data_dec_hold", dataDecHold)

        exec_time = np.array(execution_time)
        np.save("execution_time_in_seconds", exec_time)

        try:
            sampling_data = pd.DataFrame(fit.extract(["rates", "ratio", ], permuted=True))
        except:
            print("cannot make a pandas data frame")

        try:
            sampling_data.to_csv("sampling_daten")
        except:
            print("could not save fit_data")

        for name in ("param_likelihood_start","ParamLikeliStartHoldout"):
            try:
                param_likelihood = np.array(fit.extract(name, permuted=True)[name])
                param_likelihood = np.swapaxes(param_likelihood, 0, 1)
                print("param_like.shape: "+ param_likelihood.shape)
            except:
                print("param likihood existiert nicht")

            try:
                major_axis = list()
                for i in range(1, 21):
                    major_axis.append(str(i))

                param = xr.DataArray(data=param_likelihood[:, :, :, :],
                                 dims=("N_conc_time_series", "samples_posterior", "data_point", "parameter_likelihood"),
                                 coords={
                                     "N_conc_time_series": ["0.0625", "0.125", "0.25", "0.5", "1", "2", "4", "8", "16",
                                                            "64"],
                                     "parameter_likelihood": ["mean", "sigma"]})
                param.to_netcdf(name)
            except:
                print("could not save likelihood")
        for fname in ("param_likelihood_decay", "ParamLikeliDecayHoldout"):
            try:
                param_likelihood_decay = np.array(
                fit.extract(fname, permuted=True)[fname])
                param_likelihood_decay = np.swapaxes(param_likelihood_decay, 0, 1)
                param = xr.DataArray(data=param_likelihood_decay[:, :, :, :],
                                 dims=("N_conc_time_series", "samples_posterior", "data_point", "parameter_likelihood"),
                                 coords={
                                     "N_conc_time_series": ["0.0625", "0.125", "0.25", "0.5", "1", "2", "4", "8", "16",
                                                            "64"],
                                     "parameter_likelihood": ["mean", "sigma"]})
                param.to_netcdf(fname)
            except:
                print("could not save likelihood")

        try:
            backround_sigma = np.array(fit.extract("var_exp", permuted=True)["var_exp"])
            np.save("measurement_sigma", np.array(backround_sigma))
        except:
            print("could save backround noise")
        try:
            N_traces = fit.extract("N_ion_trace", permuted=True)["N_ion_trace"]
            np.save("N_traces", np.array(N_traces))
        except:
            print("N_traces param to fit")

        try:
            hyper_mu_N = fit.extract("hyper_mu_N", permuted=True)["hyper_mu_N"]
            sigma_N = fit.extract("sigma_N", permuted=True)["sigma_N"]
            np.save("hyper_mu_N", hyper_mu_N)
            np.save("sigma_N", sigma_N)
        except:
            pass

        try:
            mu_i = fit.extract("mu_i", permuted=True)["mu_i"]
            sigma_i = fit.extract("sigma_i", permuted=True)["sigma_i"]
            np.save("mu_i", mu_i)
            np.save("sigma_i", sigma_i)
        except:
            pass


        try:
            N_traces = fit.extract("mu_N", permuted=True)["mu_N"]
            np.save("mu_N", np.array(N_traces))
        except:
            print("mu_N param to fit")

        try:
            N_traces = fit.extract("var_N", permuted=True)["var_N"]
            np.save("var_N", np.array(N_traces))
        except:
            print("var_N param to fit")

        try:
            mu_k = fit.extract("mu_k", permuted=True)["mu_k"]
            np.save("mu_k", np.array(mu_k))
            sigma_k = fit.extract("sigma_k", permuted=True)["sigma_k"]
            np.save("sigma_k", np.array(sigma_k))
        except:
            pass



        try:
            open_variance = fit.extract("open_variance", permuted=True)["open_variance"]
            np.save("open_variance", np.array(open_variance))
        except:
            print("could not save open_variance param to fit")

        try:
            lp__ = fit.extract("lp__", permuted=True)["lp__"]
            lp__ = pd.DataFrame(data=lp__)
            lp__.to_csv("lp__")
        except:
            print("lp_ saving doesn t work")

        try:
            latent_time = fit.extract("LATENT_TIME", permuted=True)["LATENT_TIME"]
            np.save("latent_time", np.array(latent_time))
        except:
            print("LATENT TIME doesn t exist")

        try:
            latent_time_decay = fit.extract("LATENT_TIME_DECAY", permuted=True)["LATENT_TIME_DECAY"]
            np.save("latent_time_decay", np.array(latent_time_decay))
        except:
            print("LATENT TIME doesn t exist")

        try:
            occupat_dec = fit.extract("occupat_decay", permuted=True)["occupat_decay"]
            np.save("occupat_dec2", np.array(occupat_dec))
        except:
            print("occupat_decay doesn t exist")

        # mu = fit.extract("mu", permuted = True)["mu"]
        # np.save("mu", np.array(mu))
        try:
            equi_values = fit.extract("equi_values", permuted=True)["equi_values"]
            np.save("equi_values2", np.array(equi_values))
        except:
            print("could not open equi_values")

        try:
            occupat = fit.extract("occupat", permuted=True)["occupat"]
            print(occupat)
            np.save("occupat2", np.array(occupat))
        except:
            print("could not save occupat")

        try:
            log_lik_t = fit.extract("log_lik_t", permuted=True)["log_lik_t"]
            np.save("log_lik_t2", np.array(log_lik_t))
        except:
            print("could not save log_lik_t")

        try:
            log_lik_h = fit.extract("logLikHoldout", permuted=True)["logLikHoldout"]
            np.save("logLikHoldout", np.array(log_lik_h))
        except:
            print("cold not save log_lik_h")

        column_names = list()
        for id in range(1, np.int(N_free_param / 2 + 1)):
            column_names.append("theta[" + str(id) + "]")

        try:
            lambda_brigthness = fit.extract("lambda_brigthness", permuted=True)["lambda_brigthness"]
            np.save("lambda_brigthness", np.array(lambda_brigthness))
        except:
            print("lambda_brigthness")

        try:
            time_error = fit.extract("time_error", permuted=True)["time_error"]
            np.save("time_error", np.array(time_error))
        except:
            print("lambda_brigthness")

        theta = fit.extract("rates", permuted=True)
        theta = pd.DataFrame(data=theta["rates"], columns=column_names)
        theta.to_csv("test")

        for id in range(1, np.int(N_free_param / 2 + 1)):
            column_names.append("rates[" + str(id) + "]")

        try:
            ratio = fit.extract("ratio", permuted=True)
            ratio = pd.DataFrame(data=ratio["ratio"])
        except:
            print("ratio to data frame ratio did not work")

        try:
            ratio.to_csv("ratio")
        except:
            print("could not save ratio")




def save_data_cross(fit, data_start, data_dec, N_free_param, execution_time, chains, sampling_iter):
    #if not os.path.exists(folder):
    #    os.makedirs(folder)
    #os.chdir(folder)
    print("saving in: "+os.getcwd())

    np.save("data_start", data_start)
    np.save("data_dec", data_dec)

    exec_time = np.array([execution_time, chains, sampling_iter])
    np.save("execution_time_in_seconds", exec_time)

    try:
        sampling_data = pd.DataFrame(fit.extract(["rates","ratio", ], permuted=True))
    except:
        print("cannot make a pandas data frame")

    try:
        sampling_data.to_csv("sampling_daten")
    except:
        print("could not save fit_data")

    try:
        param_likelihood= np.array(fit.extract("param_likelihood_start", permuted=True)["param_likelihood_start"])
        param_likelihood = np.swapaxes(param_likelihood, 0,1)
        print(param_likelihood.shape)
    except:
        print("param likihood existiert nicht")


    try:
        major_axis = list()
        for i in range( 1 , 21):
            major_axis.append(str(i))

        param = xr.DataArray(data=param_likelihood[:,:,:,:],
                    dims= ("N_conc_time_series","samples_posterior","data_point","parameter_likelihood"),
                         coords={"N_conc_time_series":["0.0625", "0.125", "0.25", "0.5", "1", "2","4","8","16","64"],
                                 "parameter_likelihood": ["mean", "sigma"]})
        param.to_netcdf("param_likelihood_start")


        param_likelihood_decay = np.array(fit.extract("param_likelihood_decay", permuted=True)["param_likelihood_decay"])
        param_likelihood_decay = np.swapaxes(param_likelihood_decay, 0, 1)
        param = xr.DataArray(data=param_likelihood_decay[:, :, :, :],
                         dims=("N_conc_time_series", "samples_posterior", "data_point", "parameter_likelihood"),
                         coords={
                             "N_conc_time_series": ["0.0625", "0.125", "0.25", "0.5", "1", "2", "4", "8", "16", "64"],
                             "parameter_likelihood": ["mean", "sigma"]})
        param.to_netcdf("param_likelihood_decay")
    except:
        print("could not save likelihood")


    try:
        backround_sigma = np.array(fit.extract("var_exp", permuted=True)["var_exp"])
        np.save("measurement_sigma",np.array(backround_sigma))
    except:
        print("could save backround noise")
    try:
        N_traces = fit.extract("N_ion_trace", permuted=True)["N_ion_trace"]
        np.save("N_traces",np.array(N_traces))
    except:
        print("N_traces param to fit")

    try:
        lp__ = fit.extract("lp__", permuted=True)["lp__"]
        lp__ = pd.DataFrame(data=lp__)
        lp__.to_csv("lp__")
    except:
        print("lp_ saving doesn t work")




    try:
        latent_time = fit.extract("LATENT_TIME", permuted = True)["LATENT_TIME"]
        np.save("latent_time", np.array(latent_time))
    except:
        print("LATENT TIME doesn t exist")

    try:
        latent_time_decay = fit.extract("LATENT_TIME_DECAY", permuted=True)["LATENT_TIME_DECAY"]
        np.save("latent_time_decay", np.array(latent_time_decay))
    except:
        print("LATENT TIME doesn t exist")

    try:
        occupat_dec = fit.extract("occupat_decay", permuted=True)["occupat_decay"]
        np.save("occupat_dec2", np.array(occupat_dec))
    except:
        print("occupat_decay doesn t exist")



    #mu = fit.extract("mu", permuted = True)["mu"]
    #np.save("mu", np.array(mu))
    try:
        equi_values = fit.extract("equi_values", permuted=True)["equi_values"]
        np.save("equi_values2", np.array(equi_values))
    except:
        print("could not open equi_values")

    try:
        occupat = fit.extract("occupat", permuted=True)["occupat"]
        print(occupat)
        np.save("occupat2",np.array(occupat))
    except:
        print("could not save occupat")

    try:
        log_lik_t = fit.extract("log_lik_t", permuted=True)["log_lik_t"]
        np.save("log_lik_t", np.array(log_lik_t)[::4,:,::])
    except Exception as e:
        print(str(e))
        print("could not save log_lik_t")


    log_lik_h = fit.extract("logLikeHold", permuted=True)["logLikeHold"]
    np.save("log_lik_h" , np.array(log_lik_h)[::4,:,::])
    #except Exception as e:
    #    print(str(e))
    #    print("cold not save log_lik_h")




    column_names = list()
    for id in range(1,np.int(N_free_param/2+1)):
        column_names.append("theta["+str(id)+"]")

    try:
        i_single =fit.extract("i_single_channel", permuted = True)["i_single_channel"]
        np.save("i_single", np.array(i_single))
    except:
        print("i_single problems")



    theta = fit.extract("rates", permuted=True)
    theta = pd.DataFrame(data = theta["rates"], columns =column_names)
    theta.to_csv("test")


    for id in range(1,np.int(N_free_param/2+1)):
        column_names.append("rates["+str(id)+"]")

    try:
        ratio = fit.extract("ratio", permuted=True)
        ratio = pd.DataFrame(data=ratio["ratio"])
    except:
        print("ratio to data frame ratio did not work")

    try:
        ratio.to_csv("ratio")
    except:
        print("could not save ratio")

    try:
        lamb = fit.extract("lambda_fluoresc", permuted=True)
        lamb = pd.DataFrame(data=lamb["lambda_fluoresc"])
    except:
        print("ratio to data frame ratio did not work")

    try:
        lamb.to_csv("lambda_fluoresc")
    except:
        print("could not save ratio")






def main():
    save_data(bla)
if __name__ == "__main__":
    main()