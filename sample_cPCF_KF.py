import os
import numpy as np
import pandas as pd
import datetime
import sys
assert("linux" in sys.platform), "This code only runs Linux"

import time as ti
import sys
from fit_and_load_stan_model import fit_patch_fluo as fit_fluo_patch
from save_sampling import save_data_fluorescenc as save_cPCF
from save_sampling import save_data_new as save_electro_physio
from pystan.constants import MAX_UINT

def get_data(std_patch, plot, N_ion_numb):

    #os.chdir(pathlocal)
    print(os.getcwd())
    Ion_channel = N_ion_numb
    print("Ion_channel: "+str(Ion_channel))
    if plot == True:
        plt.figure(figsize = (24,18), tight_layout = True)

    id = 1
    all_data = list()
    #for name in ["Ergebnisse_0.0625_M_auf_0.0625M","Ergebnisse_0.125_M_auf_0.0625M",
    #             "Ergebnisse_0.25_M_auf_0.0625M","Ergebnisse_0.5_M_auf_0.0625M", "Ergebnisse_1_M_auf_0.0625M",
    #             "Ergebnisse_2_M_auf_0.0625M",
    #             "Ergebnisse_4_M_auf_0.0625M","Ergebnisse_8_M_auf_0.0625M","Ergebnisse_16_M_auf_0.0625M",
    #             "Ergebnisse_32_M_auf_0.0625M",
    #             "Ergebnisse_64_M_auf_0.0625M"]:
    current = 0
    # i take the second series but thats what we usually did anyways at selesction all data
    for name in [ "0.0625M", "0.125M",
                     "0.25M", "0.5M", "1M", "2M",
                     "4M", "8M", "16M", "32M",
                     "64M"]:
        print(os.getcwd())
        current += 1
        splits = name.split("M")
        lig_back = 0.375 * 1000 / (5 * 0.25) * float(splits[0])
        lig_ref = 0.375 * 4000 / (5 * 0.25) * 64




        if plot == True:
            plt.subplot(3,4,id)
            plt.grid()
            plt.xticks([0.1,0.3,0.6,0.8,1.0,1.2])

            plt.xlim(xmax =1.4)
            plt.ylabel(r"$y_{signal}$")
            plt.xlabel(r"$time$")
            id +=1
        os.chdir("data/"+name)

        #try:
            #fluor = pd.read_excel("1_Sim_N_"+str(Ion_channel)+"_theor_Fluor_Werte.xlsx").values*Ion_channel
            #np.savetxt("1_Sim_N_"+str(Ion_channel)+"_theor_Fluor_Werte.csv",fluor)
        #except Exception as e:
            #print(str(e))
            #fluor = pd.read_excel("1_Sim_N_" + str(Ion_channel) + "_theor_Fluor_Werte.xls").values * Ion_channel
        if plot == True:
            plt.plot(fluor[:, 0]/10000000, fluor[:, 1], ls ="--", label ="mean_F", color = "red")
        try:
            fluor = pd.read_excel("1_Sim_N_"+str(Ion_channel)+"_Fluor_MW.xlsx").values*Ion_channel
            np.savetxt("1_Sim_N_"+str(Ion_channel)+"_Fluor_MW.csv",fluor)
        except:
            fluor = pd.read_excel("1_Sim_N_" + str(Ion_channel) + "_Fluor_MW_"+str(name[:-1])+"M.xlsx").values * Ion_channel

        ###measurement_normal = np.random.poisson(lig_back, size=fluor[:, 1].shape)   - lig_back/lig_ref* np.random.poisson(lig_ref, size=fluor[:, 1].shape)
        measurement_normal = np.random.poisson(lig_back, size=fluor[:, 1].shape) \
                             - lig_back / lig_ref * np.random.poisson(lig_ref, size=fluor[:, 1].shape)
        fluor[:, 1] = 0.75 * fluor[:, 1]
        measurement_poisson = 1 * np.random.poisson(fluor[:, 1])
        fluor[999:5999, 1] = measurement_poisson[999:5999] + measurement_normal[999:5999]

        ## old noise model some backround noise
        #measurement_normal = np.random.normal(loc = 0, scale = 1, size=fluor[:, 1].shape)
        #fluor[:, 1] = 0.75 * fluor[:, 1]
        #measurement_poisson = 1*np.random.poisson(fluor[:, 1])
        #measurement_poisson = measurement_poisson + measurement_normal
        #fluor[:, 1] = measurement_poisson
        print("std measurement "+str(np.std(measurement_normal)))

        #try:
        #    current = pd.read_excel("1_Sim_N_"+str(Ion_channel)+"_theor_Strom_Werte.xlsx").values*Ion_channel
        #    np.savetxt("1_Sim_N_" + str(Ion_channel) + "_theor_Strom_Werte.csv", current)
        #except:
        #    current = pd.read_excel("1_Sim_N_" + str(Ion_channel) + "_theor_Strom_Werte.xls").values * Ion_channel

        try:
            current = pd.read_excel("1_Sim_N_"+str(Ion_channel)+"_Strom_MW.xlsx").values*Ion_channel
            np.savetxt("1_Sim_N_" + str(Ion_channel) + "_Strom_MW.csv",current)
        except:
            current = pd.read_excel("1_Sim_N_" + str(Ion_channel) + "_Strom_MW_"+str(name[:-1])+"M.xlsx").values * Ion_channel

        single_var = np.power(0.1, 2)


        open_access_noise = np.random.normal(loc=0, scale=np.sqrt(single_var * current[:,1]), size=current[:,1].shape)
        #current[:, 1] = current[:,1] + measurement_noise + open_access_noise

        measurement_noise = np.random.normal(loc=0, scale=std_patch, size=current[:,1].shape)
        #current[:, 1] = current[:, 1] + measurement_noise
        current[:, 1] = current[:, 1] + measurement_noise + open_access_noise

        data = np.array([fluor[:,1], current[:,1]])
        all_data.append(data)



        os.chdir("..")
        os.chdir("..")
    if plot == True:
        plt.legend()
        plt.savefig("Current_and_Patch_traces.pdf")
    all_data = np.array(all_data)
    all_data = all_data[[0,1,2,3,4,5,6,7,8,10],:,:]
    #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #print(all_data.shape)
    #np.savetxt("current.txt", all_data[:, 1,:])
    #np.savetxt("fluor.txt", all_data[:, 0,:])
    #np.savetxt("time.txt", current[:, 0] / (Ion_channel * 1000))
    # print("Finshied")


    current = np.loadtxt("current.txt")
    fluor = np.loadtxt("fluor.txt")
    all_data = np.swapaxes(np.array([fluor,current]),0,1)#, axis=0, out=None)
    time = np.loadtxt("time.txt")
    os.chdir("..")
    print(all_data.shape)
    return all_data, time



def get_data_new():

    os.chdir("data")
    print(os.getcwd())
    current = np.loadtxt("current.txt")
    fluor = np.loadtxt("fluor.txt")
    all_data = np.swapaxes(np.array([fluor,current]),0,1)#, axis=0, out=None)
    time = np.loadtxt("time.txt")
    os.chdir("..")
    print(all_data.shape)
    return all_data, time

def data_slices_beg_new(data,time, skip):



    y_1 = data[0,:,1100:5100:int(400/skip)]
    y_2 = data[1,:,1040:5040:int(400/skip)]
    y_3 = data[2,:,1035:5035:int(400/skip)]
    y_4 = data[3,:,1030:5030:int(400/skip)]
    y_5 = data[4,:,1010:5010:int(400/skip)]
    y_6 = data[5,:,1008:5008:int(400/skip)]
    y_7 = data[6,:,1005:5005:int(400/skip)]
    y_8 = data[7,:,1004:5004:int(400/skip)]
    y_9 = data[8,:,1003:4003:int(300/skip)]
    y_10 = data[9,:,1002:4002:int(300/skip)]


    after_jump = np.array(#[y_1,
                           [y_1, y_2, y_3,y_4, y_5, y_6, y_7, y_8, y_9, y_10])
    print(after_jump.shape)

    #y_2 = data[1,0,1020:1047:9]
    #y_3 = data[2,0,1007:1037:10]
    #y_4 = data[3,0,1005:1023:6]
    #y_5 = data[4,0,1003:1009:2]
    #y_6 = data[5, 0, 1001:1007:2]
    #y_7 = data[6, 0, 1001:1004:1]

    #fluoresc_early_kin = np.array(  # [y_1,
    #                                [ y_3, y_4, y_5 ,y_6])#, y_8, y_9, y_10])

    #print(fluoresc_early_kin)


    y_1 = data[0,:,998]
    y_2 = data[1,:,998]
    y_3 = data[2,:,998]
    y_4 = data[3,:,998]
    y_5 = data[4,:,998]
    y_6 = data[5,:,998]
    y_7 = data[6,:,998]
    y_8 = data[7,:,998]
    y_9 = data[8,:,998]
    y_10 = data[9,:,998]

    equi_before_jump = np.array([y_1,y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9, y_10])


    time = time - time[0]
    dif_time = np.array([time[int(400/skip)],
                         time[int(400/skip)],
                          time[int(400/skip)], time[int(400/skip)],
                         time[int(400/skip)], time[int(400 / skip)],time[int(400 / skip)],
                         time[int(400 / skip)],
                         time[int(300 / skip)], time[int(300 / skip)]])
    #dif_time_early = np.array([ time[int(10)],
    #                    time[int(6)], time[int(2)], time[int(2)]])

    print(time[0])
    time = time - time[998]
    time_offset = np.array([time[1100],
                            time[1040],
                             time[1035], time[1030], time[1010],
                           time[1008], time[1005], time[1004], time[1003], time[1002]])

    #time_offset_early = np.array(  # [time[1250],
    #    [
    #     time[1007], time[1005], time[1003],
    #     time[1001]])  # , time[1004], time[1003], time[1002]])

    return after_jump, dif_time, time_offset, equi_before_jump, #fluoresc_early_kin, #dif_time_early, #time_offset_early



def data_slices_decay_new(data, time, skip):
    y_1 = data[0,:,6001:7001:int(100/skip)]
    y_2 = data[1,:,6001:7001:int(100/skip)]
    y_3 = data[2,:,6001:8001:int(200/skip)]
    y_4 = data[3,:,6001:8001:int(200/skip)]
    y_5 = data[4,:,6001:8001:int(200/skip)]
    y_6 = data[5,:,6001:9001:int(300/skip)]
    y_7 = data[6,:,6001:9001:int(300/skip)]
    y_8 = data[7,:,6001:11001:int(500/skip)]
    y_9 = data[8,:,6001:12001:int(600/skip)]
    y_10 = data[9,:,6001:12001:int(600/skip)]
    before_jump = np.array([y_1,y_2,y_3, y_4, y_5, y_6,y_7, y_8, y_9, y_10])

    y_1 = data[0,:,5998]
    y_2 = data[1,:,5998]
    y_3 = data[2,:,5998]
    y_4 = data[3,:,5998]
    y_5 = data[4,:,5998]
    y_6 = data[5,:,5998]
    y_7 = data[6,:,5998]
    y_8 = data[7,:,5998]
    y_9 = data[8,:,5998]
    y_10 = data[9,:,5998]
    after_jump = np.array([y_1,y_2,y_3, y_4, y_5, y_6, y_7, y_8, y_9, y_10])


    time = time - time[0]

    #dif_time_dec = np.array([time[int(400/skip)],time[int(400/skip)], time[int(400/skip)],#400
    #                         time[int(100/skip)], time[int(400/skip)], time[int(400 / skip)], time[int(400 / skip)], #400
    #                              time[int(500 / skip)], time[int(500 / skip)], time[int(700 / skip)]])

    dif_time_dec = np.array([time[int(100/skip)],
                             time[int(100/skip)],
                              time[int(200/skip)], time[int(200/skip)],
                                  time[int(200/skip)],time[int(300 / skip)],time[int(300 / skip)],
                                  time[int(500 / skip)], time[int(600 / skip)], time[int(600 / skip)]])
    time = time - time[998]
    time_off_set_pseudo_equi = time[5998]

    time = time - time[5998]
    time_offset_dec = np.array([time[6001],
                                time[6001],
                                 time[6001], time[6001], time[6001],
                           time[6001], time[6001], time[6001], time[6001], time[6001]])
    return before_jump, dif_time_dec, time_offset_dec, after_jump, time_off_set_pseudo_equi




def main():
    os.environ["STAN_NUM_THREADS"] = "72"
    plot = False
    if plot == True:
        from matplotlib import pyplot as plt
        plt.rcParams["xtick.labelsize"] = 18
        plt.rcParams["ytick.labelsize"] = 18
        plt.rcParams["axes.labelsize"] = 18
        plt.rcParams["axes.labelweight"] = 18
    print("working in: "+os.getcwd())
    localtime = ti.asctime(ti.localtime(ti.time()))
    curr_work_dir = os.getcwd()
    print(localtime)

    if len(sys.argv) != 2:
        print('Invalid Numbers of Arguments. Script will be terminated.')
        return
    else:


        N_ion_numb = int(sys.argv[1])



    skip = 5.0
    N_free_param = 6
    Ion_channels = N_ion_numb


    std_patch = 1.0
    starting_path = os.getcwd()
    all_data, time = get_data_new()
    data_start, dif_time, time_of_set_arr, equi_before_jump = data_slices_beg_new(all_data,time, skip)
    data_dec, dif_time_dec, time_of_set_dec, equi_after_jump, time_offset_pseudo_equi = data_slices_decay_new(all_data, time, skip)
    data_start = np.swapaxes(data_start,1,2)
    data_dec = np.swapaxes(data_dec,1,2)
    os.chdir(starting_path)

    all_data_hold, time_hold = get_data_new()
    data_start_hold, dif_time_hold, time_of_set_arr_hold, equi_before_jump_hold = data_slices_beg_new(all_data_hold,time, skip)
    data_dec_hold, dif_time_dec_hold, time_of_set_dec_hold, equi_after_jump_hold, time_offset_pseudo_equi_hold = data_slices_decay_new(all_data_hold, time, skip)
    data_start_hold = np.swapaxes(data_start_hold,1,2)
    data_dec_hold = np.swapaxes(data_dec_hold,1,2)
    #print(data_start)


    Sum_N_traces = 1
    #print("std" +str(std_patch))
    Sampling_data_param = {"var_exp_hat": np.power(std_patch,2),

                            "y_start": data_start[:,:,:],
                            "y_equi_before_jump": equi_before_jump[:,:],
                           "y_dec": data_dec[:,:,:],
                           "y_equi_after_jump": equi_after_jump[:,:],
                           "dif_time": dif_time,
                           "dif_time_dec": dif_time_dec,
                           "off_set_time_arr": time_of_set_arr,
                           "time_off_set_dec": time_of_set_dec,
                           "time_offset_off_ligand_wash":time_offset_pseudo_equi,
                           "SUM_N_traces": Sum_N_traces,
                           "y_start_hold": data_start_hold,
                           "y_equi_before_jump_hold": equi_before_jump_hold[:, :],
                           "y_dec_hold": data_dec_hold,
                           "y_equi_after_jump_hold": equi_after_jump_hold[:, :],
                            "N_data": [len(data_start[0,:,0]), len(data_dec[0,:,0])],
                           "N_conc": data_start.shape[0],
                           "N_ion_ch": Ion_channels,
                           "M_states": 4,
                           "N_free_para": N_free_param,
                           "N_open_states": 1,
                           "ligand_conc": [
                               [1, 0,      1, 0,      1, 1],
                               [1, 0.0625, 1, 0.0625, 1, 1],
                               [1, 0.125,  1, 0.125,  1, 1],
                               [1, 0.25,   1, 0.25,   1, 1],
                               [1, 0.5,    1, 0.5,    1, 1],
                               [1, 1,      1, 1,      1, 1],
                               [1, 2,      1, 2,      1, 1],
                               [1, 4,      1, 4,      1, 1],
                               [1, 8,      1, 8,      1, 1],
                               [1, 16,     1, 16,     1, 1],
                               #[1, 32,     1, 32,     1, 1],
                               [1, 64,     1, 64,     1, 1]],
                           "ligand_conc_decay": [
                               [1, 0.0, 1, 0.0,     1, 1],
                               [1, 0.0,   1, 0.0,   1, 1],
                               [1, 0.0,   1, 0.0,   1, 1],
                               [1, 0.0,   1, 0.0,   1, 1],
                               [1, 0.0,   1, 0.0,   1, 1],
                               [1, 0.0,   1, 0.0,   1, 1],
                               [1, 0.0,   1, 0.0,   1, 1],
                               [1, 0.0,   1, 0.0,   1, 1],
                               [1, 0.0,   1, 0.0,   1, 1],
                               [1, 0.0,   1, 0.0,   1, 1]]


                           }

    print(Sampling_data_param)

    print(os.getcwd())
    #if skip ==1:
        #warm_up = 4000
        #sampling_iter = 36000#14500  # 50000
    #else:
        #warm_up = 4000
        #sampling_iter = 36000
    warm_up = 4000#1000#2000
    sampling_iter = 26000#26000

    chains = 4
    statistical_model = "Kalman_fluorescence.pic"
    #statistical_model = "moffat.pic"
    prog_time_start = ti.time()
    os.chdir(curr_work_dir)
    seed = np.random.randint(0, MAX_UINT, size=1)
    invMetric = 0
    stepsize = 0
    trained = False
    fit, model = fit_fluo_patch(Sampling_data_param,
                                  statistical_model,
                                  sampling_iter,
                                  chains,warm_up,
                                  seed, invMetric,
                                  stepsize, trained)
    time_prog_delta = ti.time() - prog_time_start
    execution_time = datetime.timedelta(seconds=time_prog_delta).total_seconds()
    print("execution time: " + str(execution_time))

    folder_save = ""
    print(fit)
    print(os.getcwd())
    if statistical_model == "Kalman_fluorescence.pic":
        save_cPCF(fit, data_start, data_dec, N_free_param, execution_time, chains, sampling_iter, seed)
    else:
        save_electro_physio(fit, data_start, data_dec, N_free_param, execution_time)
    return


if __name__ == "__main__":
    main()