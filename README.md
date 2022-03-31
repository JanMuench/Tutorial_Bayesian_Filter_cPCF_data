# Tutorial for Patch-clamp data analysis
## General Info


This tutorial is an example code for confocal patch-clamp fluorometry measurements which is part of the publication “Bayesian inference of kinetic schemes for ion channels by Kalman filtering”. The work was done in Stan https://mc-stan.org/ with the PyStan https://pystan.readthedocs.io/en/latest/ interface version 2.19.1.2 which is currently outdated. We plan to update the tutorial folder to PyStan 3 in the near future. The code is parallelized for the multiple CPUs of a node on a computation cluster. Each individual sampling chain is trivially calculated in parallel provided by the Stan language.

The package contains a file containing the STAN code [“Kalman_fluorescence.txt”](Kalman_fluorescence.txt) as well as the python script [“compile_CCCCO_normal_split.py”](compile_CCCCO_normal_split.py) to compile the code. Finally, a Python script [“sample_PC_data.py”](sample_cPCF_KF.py) which acts as the interface between the data and the sampler. To adapt the code to your data, basic Stan programming skills are required. The Python knowledge and the Python scripts are not obligatory, because Stan can interact many high level data analysis programming languages (R, Python, shell, MATLAB, Julia, Stata) . Some tutorials about Stan Bayesian statistics and model selection can be found here https://mc-stan.org/users/documentation/tutorials or here
https://ourcodingclub.github.io/tutorials/stan-intro/. Also youtube has many good starting tutorials.

The topology of the kinetic scheme is uniquely defined by a rate matrix. Our example code demonstrates the analysis with a two ligand-gated 4 state model of confocal patch-clamp fluorometry data. The rate matrix is defined in the function ["create_rate_matrix line 537"](Kalman_fluorescence.txt#L537)  and its sub functions ["multiply_ligandconc_CCCO line 286"](Kalman_fluorescence.txt#L286)  and `assign_param_to_rate_matrix_CCCO` line 308 in the file `Kalman_fluorescence.txt`. 
The mean observation is defined in ["define_observation_model_CCCO line 265"](Kalman_fluorescence.txt#L265) with the function `define_observation_model_CCCO`  which is called in line 663.

## Step by step:


<details>
<summary><b> How to start the posterior sampling of the example cPC data as test run on a node of cluster. </b></summary>
1. One needs to install `Stan` and `PyStan`.

2. One executes `compile_CCCCO_normal_split.py` by prompting
`python3  compile_CCCCO_normal_split.py` into the command line one the computer cluster where the programm is going to be excuted.
That compiles the Stan code `KF.txt` into an executable program `KF_CCCO.pic`.
	 
3. Prompting `python3 sample_cPCF_KF.py 2000` executes a Python program which acts 
as an interface between the data from `data/current8000` and 	    
sampling algorithm `KF_CCCO.pic`. In the folder, data are 4 numpy arrays. The numpy 
array `current8000.npy` has the data of 10 different ligand concentrations with two
ligand jumps from zero to the concentration and back to zero. The numpy array  `Time.npy`
is the time axis of all traces in the  current array. The ligand concentrations are saved 
in `ligand_conc.txt` and `ligand_conc_decay.txt`. 
### Paralized over the CPUs of a node
Additionally, each time trace 
is cutted that activation or deactivation is treated as an individual time trace on an 
individual CPU. We assumed that we only needed 5 patches. So two ligand concentrations 
were measured from one patch. For optimal caluclation efficiency, 10 time traces 
require 20 CPUs (activation and decay). 40 CPU to apply cross validaton times 4 for 
4 independent sample chains.
The function `map_rect` distributes the timetraces data,the parameter samples  and the function
`wrapped_calculate_likelyhood_less_memo` and distributes the claculation task dedined in `wrapped_calculate_likelyhood_less_memo`
to all CPUs.


</details>




<details>
<summary><b> The output of the sampler in contrast to the arameters which are actaully sampled. </b></summary>
4. The output of samples as we used them in the publication.
4.1 The csv file `rate_matrix_params` saves the samples of the posterior of the rate 
matrix. Simply analysing them means that we marginalized all other parameters out. Note
that the dwell times are on a scaled log space 	thus one has to multiply them by a 
scaling factor for the actual log space. 
4.2 The single-channel current samples are saved in an numpy array `i_single.npy`.
4.3 The samples of the variance parameter are saved in the numpy array 	file `measurement_sigma.npy`.
4.4 The samples of the open-channel variance parameter are saved in the numpy array file `open_variance.npy`.
4.5 The samples of the “Ion channels per time trace parameter” are saved in the numpy array file `N_traces.npy`.


</details>

<details>
<summary><b> How to start the posterior sampling of the example PC data as test run on a node of cluster. </b></summary>
Each row of the ligand matrix defines an 
array whose entries are element-wise multiplied to the rates in the function 
`multiply_ligandconc_CCCO`. Ligand-independent rates are multiplied by one and the ligand
depended rates are multiplied with a ligand concentration. Within the script  
“sample_PC_data.py” in the functions “data_slices_beg_new” and 	“data_slices_decay_new” 
the time points of the concentration jumps are defined
	
	
5. To adapt the kinetic scheme one needs to change a few things within KF.txt  which are 
the observation model matrix H and the functions related to the kinetic scheme. Then 
“KF.txt” needs to be recompiled:
5.1. The row vector “conduc_state” needs to  be changed to the desired signal model. It 
represents the matrix H of the 	article which generates the mean signal for a given 
ensemble state. If more than  two conducting classes (non-conducting and conducting) are
modeled, additional single-channel current parameters need to be defined in the parameters block.

	
5.2 The function “multiply_ligandconc_CCCO” needs to be adapted. That function takes the parameters from
the parameters block and computes the rates of the rate matrix. They are then passed to the 
“assign_param_to_rate_matrix_CCCO” function. Note that this example code has four dwell times as parameters and two 
ratios from them the six rates are constructed. We recommend to use a log uniform prior for the 
dwell time and a beta distribution or rather a Dirichlet distribution for the 	probabilities which transition is taken.
5.3 The function “assign_param_to_rate_matrix_CCCO” assigns rates to 	the off diagonal elements. Note that 
a closed first order Markov system requires that each diagonal element is the negative sum of its column. 
That property is enforced in function “assign_diagonal_elements”. Note that this is redundant as we start 
in the 	parameters block with the dwell times as parameters. But we could have chosen a different 
parametrization to begin with. We argue in the paper to use this parametrisation in order to use a 
Jeffreys prior but there a couple of other options.
5.4 The mean observation needs to be changed in line 806
5.5 If the amount of open-channel states with differing open-channel noise variances for each state needs to be calculated,
the function “calc_sigma_and_mean” must be adapted	

</details>

## Some other comments 
Although we recommend to have the dwell times (diagonal elements of the rate matrix) as parameters, we recalculate them which is reminiscent of former parameterizations.

Note, that PyStan 3 is not downward compatible. Thus, using PyStan 3 requires some minor changes to the compiling sampling and saving of the samples. To visualize and analyze a posterior/draw from the posterior, we recommend the package Corner.py. To implement a posterior post processing and diagnosis in a Bayesian workflow, we highly recommend using the Arviz package.

Note, that in the first KF analysis round we do not report the derived quantities such as mean signal and covariance for a given time. We discussed in the Appendix of the paper that this would expand the total runtime of the program by roughly two orders of magnitude. To show the posterior of the mean signal and the posterior of the variance, we suggest to use a subset of the posterior samples and feed it to the KF to do the filtering. It requires minimal changes to the KF code.

The data used for the posterior is “data_start.npy” and “data_dec.npy”. The suffix “hold” means that this would be the data used as a hold-out data set if one would do cross validation.
