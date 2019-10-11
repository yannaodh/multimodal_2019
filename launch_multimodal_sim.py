import sys
sys.path.append('/home/ysweeney/Dropbox/notebooks/')
sys.path.append('/Users/yann/Dropbox/notebooks/')

import py_scripts_yann

import recurrent_network_functions as rec_net
import numpy as np
import itertools

from matplotlib import pyplot as plt
from scipy import stats

import seaborn as sns

from sacred import Experiment
ex = Experiment()

from sacred.observers import MongoObserver
#try:
#    ex.observers.append(MongoObserver.create(db_name='sacred_yann'))
#    print 'using mongodb'
#except:
#    print 'no mongodb'
try:
    from sacred.observers import FileStorageObserver
    ex.observers.append(FileStorageObserver.create('/mnt/DATA/ysweeney/data/multimodal/multimodal_net_runs'))
except:
    try:
        ex.observers.append(FileStorageObserver.create('/media/ysweeney/HDD/multimodal/multimodal_net_runs'))
    except:
        print 'no observers'
import cPickle
from tempfile import mkdtemp
import shutil
import os

try:
    import mkl
    mkl.set_num_threads(1)
except:
    'Couldnt set mkl threads'
    pass

import run_multimodal

import multiprocessing

@ex.config
def config():
    sim_pars = {
        'N_E': 200,
        'W_max': 0.08,
        'W_inh_min': -1.5,
        'prune_threshold': 0.0,
        'H_max': 10.0,
        'H_secondary': 7.0,
        'N_secondary_groups': 2,
        'H_min': 1.0,
        'sample_res': 50000,
        'alpha': 2.5e-9,
        'eta': 10.0e-8,
        'BCM_lambda':1.0,#- 5e-7,
        'theta_BCM': np.ones(100)*5.0,
        'theta_BCM_dt': 10e-5,
        'init_Winh': -0.275,
        'init_Wexc': 0.05,
        'frac_A_only': 0.2,
        'frac_V_only': 0.4,
        'frac_AV': 0.4,         # A+V+AV must sum to 1
        'frac_nonresponsive': 0.0,
        'distribute_exps' : True,
        'N_cores' : 50,
        'N_sims': 50
        }
    sim_pars['N_I'] = int(sim_pars['N_E']*0.2)

    #par_sweep_key = 'alpha'
    #par_sweep_vals = [5e-10,1e-9,2e-9,5e-9,1e-8]
    sim_pars['par_sweep_key'] = 'BCM_lambda'
    sim_pars['par_sweep_vals'] = [1.0,1.0-1e-7,1.0-5e-7]
    sim_pars['par_sweep_key_2'] = 'H_secondary'
    sim_pars['par_sweep_vals_2'] = [5.0,6.0,7.0,8.0]
    sim_pars['sim_title'] = 'N_200_N_secondary_2_H_max_10_secondary_7_lambda_1_10nets_audiovisual_random_W_max_p08'

    pairing_pars = {
        'T_pairstim': int(2e4),
        'alpha': 5.0e-8,
        'eta': 0.0e-8,
        'disinhibitory_tone_response': 0.0,
        'enhance_response_threshold': 1.5,
        'set_theta_BCM': True,
        'theta_BCM': 9.0,
        'prune_synapses': False,
        'prune_threshold': 0.3,
        'H_max': 10.0,
        'H_secondary': 10.0,
        'N_instantiations': 50,
        'ext_OU_sigma': 2.5
    }

    sim_pars['pairing_pars'] = pairing_pars

    sim_pars['pairing_pars']['pairing_path'] = 'alpha_'+str(pairing_pars['alpha'])+'_eta_'+str(pairing_pars['eta'])+'BCM_9p0_2e4_no_pruning_spont_H_10_10_OU_2p5_gap_trialsx5'

def run_iter(iter_pars):
    run_multimodal.run_multimodal_net(int(50e5),int(1e5),int(10e5),'none','plastic_threshold_sam_approx_ratios_fracresponsive_N100_2_secondary_groups_Hv_10_Haud_5_short_theta_BCM_10e-5_alpha_AV_full_responsive_2p5e-9_'+str(par_sweep_key)+'_'+str(par_sweep_vals[i])+'.pkl',iter_pars)

@ex.capture
def run_iter_distributed(sim_pars,i,_run):
    iter_pars = sim_pars.copy()
    exp_results = run_multimodal.run_multimodal_net(int(80e5),int(0e5),int(5e4),'none',None,iter_pars)

    exp_dir = mkdtemp(dir="./")
    # create a filename for storing some data
    data_file = os.path.join(exp_dir,str(sim_pars['sim_title'])+'_'+str(i)+'.pkl')
    # assume some random results
    with open(data_file, 'wb') as f:
            print("writing results")
            cPickle.dump(exp_results, f)
    # add the result as an artifact, note that the name here is important
    # as sacred otherwise will try to save to the oddly named tmp subdirectory created
    ex.add_artifact(data_file, name=os.path.basename(data_file))
    # at the very end of the run delete the temporary directory
    # sacred will have taken care of copying all the results files over to the run directoy
    shutil.rmtree(exp_dir)

    return exp_results


#for i in xrange(len(par_sweep_vals)):
#    rec_net_pars[par_sweep_key] = par_sweep_vals[i]
#    print rec_net_pars[par_sweep_key]
#    #run_multimodal.run_multimodal_net(int(50e5),int(1e5),int(20e5),'none','plastic_threshold_sam_approx_ratios_allresponsive_N100_medium'+str(par_sweep_key)+'_'+str(par_sweep_vals[i])+'.pkl',rec_net_pars)
#    multiprocessing.Process(target=run_iter,args=[rec_net_pars]).start()

def analyse_results_distributed(sim_pars,results_path):
    iter_pars = sim_pars.copy()
    exp_results = py_scripts_yann.load_pickle(results_path)

    analysis_results = run_multimodal.plot_W_groups_change(exp_results,iter_pars)

    exp_dir = mkdtemp(dir="./")
    # create a filename for storing some data
    data_file = os.path.join(exp_dir,results_path+str('_analysis_')+str(iter_pars['pairing_pars']['pairing_path']))
    # assume some random results
    with open(data_file, 'wb') as f:
        print("writing results")
        cPickle.dump(analysis_results, f)
    # add the result as an artifact, note that the name here is important
    # as sacred otherwise will try to save to the oddly named tmp subdirectory created
    ex.add_artifact(data_file, name=os.path.basename(data_file))
    # at the very end of the run delete the temporary directory
    # sacred will have taken care of copying all the results files over to the run directoy
    shutil.rmtree(exp_dir)

    return exp_results

@ex.command
def launch_multiple_analysis(sim_pars):
    from multiprocessing import Pool, Process, Manager
    import glob

    file_paths = glob.glob(sim_pars['analysis_pars']['launch_analysis_from_dir']+'/*.pkl')[:sim_pars['pairing_pars']['N_instantiations']]

    with Manager() as manager:
        #exp_results_list = manager.list()  # <-- can be shared between processes.
        processes = []
        for file_path in file_paths:
            print file_path
            p = Process(target=analyse_results_distributed, args=(sim_pars,file_path))  # Passing the list
            np.random.seed()
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

@ex.main
def auto_main(sim_pars):
    print 'running multicore'
    from multiprocessing import Pool, Process, Manager
    if sim_pars['distribute_exps']:
        with Manager() as manager:
            #exp_results_list = manager.list()  # <-- can be shared between processes.
            processes = []
            sim_pars['N_sims'] = int(sim_pars['N_sims']/sim_pars['N_cores'])
            for i in range(sim_pars['N_cores']):
                p = Process(target=run_iter_distributed, args=(sim_pars,i))  # Passing the list
                np.random.seed()
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            #py_scripts_yann.save_pickle_safe('/mnt/DATA/ysweeney/data/topdown_learning/'+str(sim_title)+'.pkl',exp_results_list)
        #p = Pool(N_cores)
        #par_sweep_key = ['N_sims']
        #p.map(run_iter,int(sim_pars['N_sims']/N_cores))
    else:
        with Manager() as manager:
            processes = []
            for par_val in sim_pars['par_sweep_vals']:
                if not sim_pars['par_sweep_key_2'] == None:
                    iter_pars = sim_pars.copy()
                    iter_pars[sim_pars['par_sweep_key']] = par_val
                    for par_val_2 in sim_pars['par_sweep_vals_2']:
                        iter_pars[sim_pars['par_sweep_key_2']] = par_val_2
                        p = Process(target=run_iter_distributed, args=(iter_pars,sim_pars['par_sweep_key']+str(iter_pars[sim_pars['par_sweep_key']])+sim_pars['par_sweep_key_2']+str(iter_pars[sim_pars['par_sweep_key_2']])))  # Passing the list
                        np.random.seed()
                        p.start()
                        processes.append(p)
                else:
                    iter_pars = sim_pars.copy()
                    iter_pars[sim_pars['par_sweep_key']] = par_val
                    p = Process(target=run_iter_distributed, args=(iter_pars,sim_pars['par_sweep_key']+str(iter_pars[sim_pars['par_sweep_key']])))  # Passing the list
                    np.random.seed()
                    p.start()
                    processes.append(p)
            for p in processes:
                p.join()

        #p = Pool(len(sim_pars['par_sweep_vals']))
        #p.map(run_iter,par_sweep_vals)
        #p.map(func_star, itertools.izip(sim_pars['par_sweep_vals'], itertools.repeat(sim_pars)))


if __name__ == "__main__":
    import sys
    if len(sys.argv)> 1 and sys.argv[1] == 'analysis':
        if len(sys.argv)< 2:
            ex.run('launch_multiple_analysis')
        else:
            'launch analysis with dir given', sys.argv[2]
            sim_pars_pass = {'sim_pars':{'analysis_pars':{'launch_analysis_from_dir':sys.argv[2]}}}
            ex.run('launch_multiple_analysis',sim_pars_pass)
#    elif len(sys.argv)> 1 and sys.argv[1] == 'analysis_gather':
#        if len(sys.argv)>3 and type(sys.argv[2]) is str:
#            gather_multiple_decoding_measures(sys.argv[2],sys.argv[3])
#        else:
#            import os
#            gather_multiple_decoding_measures(os.getcwd(),sys.argv[2])
    else:
        ex.run()
