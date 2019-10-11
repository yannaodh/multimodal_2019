
# coding: utf-8

# In[6]:

import sys
sys.path.append('/home/ysweeney/Dropbox/notebooks/')
import py_scripts_yann

import recurrent_network_functions as rec_net

import numpy as np
import itertools

from matplotlib import pyplot as plt
from scipy import stats

import seaborn as sns
from NeuroTools import stgen
from pdb import set_trace as bp

def get_group_connect_W(W,visual_groups,secondary_groups):
    v_groups = []
    a_groups = []
    av_groups = []
    groups = []
    N_vis = len(np.unique(visual_groups)[1:])
    N_aud = len(np.unique(secondary_groups)[1:])
    for i in xrange(N_vis):
        v_groups.append(np.where(visual_groups == i)[0])
        groups.append(np.where(visual_groups == i)[0])
    for i in xrange(N_aud):
        a_groups.append(np.where(secondary_groups == i)[0])
        groups.append(np.where(secondary_groups == i)[0])
    for i,j in itertools.product(range(N_vis),range(N_aud)):
        #print i,j
        av_groups.append(list(set(groups[i]).intersection(set(groups[N_vis+j]))))
        groups.append(list(set(groups[i]).intersection(set(groups[N_vis+j]))))
        v_groups[i] = list(set(v_groups[i]).difference(set(av_groups[-1])))
        a_groups[j] = list(set(a_groups[j]).difference(set(av_groups[-1])))
        groups[i] = list(set(groups[i]).difference(set(av_groups[-1])))
        groups[N_vis+j] = list(set(groups[N_vis+j]).difference(set(av_groups[-1])))

    W_av_group = np.zeros((len(av_groups),len(av_groups)))
    for i, j in itertools.product(range(len(av_groups)), range(len(av_groups))):
        W_temp = 0.0
        N_temp = 0.0
        for k, l in itertools.product(av_groups[i], av_groups[j]):
            W_temp += W[k, l]
            N_temp += 1
        W_av_group[i, j] = W_temp/(N_temp+1e-5)

    W_group = np.zeros((len(groups),len(groups)))
    for i, j in itertools.product(range(len(groups)), range(len(groups))):
        W_temp = 0.0
        N_temp = 0.0
        for k, l in itertools.product(groups[i], groups[j]):
            W_temp += W[k, l]
            N_temp += 1
        W_group[i, j] = W_temp/(N_temp+1e-5)

    return W_group,groups

def get_PC_responses(results,W,T=1e5,spont_activity=False,rec_net_pars={}):
    from scipy import stats
    rates, inputs = get_responses(results, W, int(T), rec_net_pars , spont_activity)
    print rates.shape
    pop_rates = np.mean(rates,axis=1)
    print pop_rates
    #plt.plot(pop_rates)
    #plt.show()
    PC = np.zeros(rates.shape[1])
    tone_0_response = np.zeros(rates.shape[1])
    non_tone_0_response = np.zeros(rates.shape[1])
    vis_0_response = np.zeros(rates.shape[1])
    vis_1_response = np.zeros(rates.shape[1])
    tone_0_present = np.zeros(rates.shape[0])
    tone_1_present = np.zeros(rates.shape[0])
    vis_0_present = np.zeros(rates.shape[0])
    vis_1_present = np.zeros(rates.shape[0])
    assert rates.shape[0] == inputs.shape[0], 'using sample_res!=1 for frozen net! should use 1'
    tone_0_present[inputs[:,1]==0] = 1
    tone_1_present[inputs[:,1]==1] = 1
    vis_0_present[inputs[:,0]==0] = 1
    vis_1_present[inputs[:,0]==1] = 1
    for i in xrange(rates.shape[1]):
        PC[i] = stats.pearsonr(pop_rates,rates[:,i])[0]
        tone_0_response[i] = np.mean(rates[tone_0_present==1,i])
        non_tone_0_response[i] = np.mean(rates[tone_1_present==1,i])
        vis_0_response[i] = np.mean(rates[vis_0_present==1,i])
        vis_1_response[i] = np.mean(rates[vis_1_present==1,i])
    corr_matrix = np.corrcoef(rates,rowvar=0)
    return PC, tone_0_response,non_tone_0_response , corr_matrix, vis_0_response, vis_1_response

def get_exp_stats(result):
    W_before,_ = get_group_connect_W(results['W_plastic'],results['visual_groups'],results['secondary_groups'])
    W_after,groups = get_group_connect_W(results['W_depriv'],results['visual_groups'],results['secondary_groups'])
    W_change = W_after-W_before

    PC_plastic, tone_0_response,non_tone_0_response,_ = get_PC_responses(results,results['W_plastic'],5e5)
    PC_depriv, tone_0_response_after,non_tone_0_response_after,_ = get_PC_responses(results,results['W_depriv'],5e5)
    PC, _ , _, _= get_PC_responses(results,results['W_plastic'],5e5,True)

    return W_before,W_after,groups,W_change,PC,tone_0_response,tone_0_response_after

def plot_avg_W_groups_change(results_string,N_res):
    results = py_scripts_yann.load_pickle(results_string + str(0)+'.pkl')
    W,_ = get_group_connect_W(results['W_plastic'],results['visual_groups'],results['secondary_groups'])
    W_before = np.zeros((N_res,W.shape[0],W.shape[1]))
    W_after = np.zeros((N_res,W.shape[0],W.shape[1]))
    W_change = np.zeros((N_res,W.shape[0],W.shape[1]))
    PC_group = np.zeros((N_res,W.shape[0]))
    tone_0_response_group = np.zeros((N_res,W.shape[0]))
    tone_0_response_group_after = np.zeros((N_res,W.shape[0]))

    for i in xrange(N_res):
        results = py_scripts_yann.load_pickle(results_string + str(i)+'.pkl')
        W_before[i],_ = get_group_connect_W(results['W_plastic'],results['visual_groups'],results['secondary_groups'])
        W_after[i],groups = get_group_connect_W(results['W_depriv'],results['visual_groups'],results['secondary_groups'])
        W_change[i] = W_after[i]-W_before[i]

        PC_plastic, tone_0_response,non_tone_0_response,_ = get_PC_responses(results,results['W_plastic'],5e5)
        PC_depriv, tone_0_response_after,non_tone_0_response_after,_ = get_PC_responses(results,results['W_depriv'],5e5)
        PC, _ , _, _ = get_PC_responses(results,results['W_plastic'],5e5,True)
        tone_0_response_change = tone_0_response_after-tone_0_response
        tone_0_response_group[i] = [np.mean(tone_0_response[g]) for g in groups]
        tone_0_response_group_after[i] = [np.mean(tone_0_response_after[g]) for g in groups]
        PC_group[i] = [10.0*np.mean(PC[g]) for g in groups]

    #W_before,_ = get_group_connect_W(results['W_plastic'],results['visual_groups'],results['secondary_groups'])
    #W_after,groups = get_group_connect_W(results['W_depriv'],results['visual_groups'],results['secondary_groups'])
    #W_change = W_after-W_before

    #PC_plastic, tone_0_response,non_tone_0_response = get_PC_responses(results,results['W_plastic'],5e5)
    #PC_depriv, tone_0_response_after,non_tone_0_response_after = get_PC_responses(results,results['W_depriv'],5e5)
    #PC, _ , _= get_PC_responses(results,results['W_plastic'],5e5,True)

    import networkx as nx

    #net = nx.from_numpy_matrix(W_before)
    net = nx.from_numpy_matrix(np.mean(W_before,axis=0))
    pos = nx.spring_layout(net)
    weights = [net[u][v]['weight'] * 20.0 for u, v in net.edges]
    #net_change = nx.from_numpy_matrix(W_change)
    net_change = nx.from_numpy_matrix(np.mean(W_change,axis=0))
    weights_change = [net_change[u][v]['weight'] * 20.0 for u, v in net.edges]

    #tone_0_response_change = tone_0_response_after-tone_0_response
    tone_0_response_change = np.mean(tone_0_response_after,axis=0)-np.mean(tone_0_response,axis=0)
    tone_0_response_group = [np.mean(tone_0_response[g]) for g in groups]
    tone_0_response_group_after = [np.mean(tone_0_response_after[g]) for g in groups]
    PC_group = np.mean(PC_group,axis=0)
    #PC_group = [10.0*np.mean(PC[g]) for g in groups]

    import matplotlib
    jet = cm = plt.get_cmap('RdBu_r')
    cNorm  = matplotlib.colors.Normalize(vmin=-2.0, vmax=2.0)
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=jet)
    colorList = []
    for i in weights_change:
        colorVal = scalarMap.to_rgba(i)
        colorList.append(colorVal)

    import seaborn as sns
    color_palette = ['b','g','r','y']
    color_palette = color_palette[:int(max(results['secondary_groups'])+1)]
    colors = ['black']*int(max(results['visual_groups'])+1) + color_palette*(1+int(max(results['visual_groups'])+1))

    nx.draw(net, pos=pos, width=weights, edge_color=colorList,node_color=colors,node_size=tone_0_response_group)
    plt.show()
    nx.draw(net, pos=pos, width=weights, edge_color=colorList,node_color=colors,node_size=tone_0_response_group_after)
    plt.show()
    nx.draw(net, pos=pos, width=weights, edge_color=colorList,node_color=colors,node_size=PC_group)
    plt.show()

    plt.scatter(PC,PC_plastic)
    plt.show()
    plt.scatter(PC,PC_depriv)
    plt.show()
    plt.scatter(PC_plastic,PC_depriv)
    plt.show()

    return [PC_group,tone_0_response_group,tone_0_response_group_after,W_before,W_change,W_after]

def plot_W_groups_change(results,sim_pars):

    W_before,_ = get_group_connect_W(results['W_plastic'],results['visual_groups'],results['secondary_groups'])
    W_after,groups = get_group_connect_W(results['W_depriv'],results['visual_groups'],results['secondary_groups'])
    W_change = W_after-W_before

    rec_net_pars = results['rec_net_pars']
    rec_net_pars.update(sim_pars['pairing_pars'])
    rec_net_pars['pairing_pars'] = sim_pars['pairing_pars']

    rates_during_simult, inputs_during_simult = get_responses(results, results['W_plastic'], int(2e5), rec_net_pars, False, False,sim_pars['pairing_pars']['T_pairstim'],True)
    tone_0_present = np.zeros(rates_during_simult.shape[0])
    tone_0_present[inputs_during_simult[:, 1] == 0] = 1
    tone_0_response_simult = np.zeros(rates_during_simult.shape[1])
    for i in xrange(rates_during_simult.shape[1]):
        tone_0_response_simult[i] = np.mean(rates_during_simult[tone_0_present == 1, i])

    rates_during_pair, inputs_during_pair, _, _, W_pair, W_repeat = get_responses(results, results['W_plastic'], int(2e5), rec_net_pars, False, True,sim_pars['pairing_pars']['T_pairstim'])
    tone_0_present = np.zeros(rates_during_pair.shape[0])
    tone_0_present[inputs_during_pair[:, 1] == 0] = 1
    corrs_during_pair = np.corrcoef(rates_during_pair[tone_0_present == 1, :],rowvar=0)

    W_pair_groups,_ = get_group_connect_W(W_pair,results['visual_groups'],results['secondary_groups'])
    W_repeat_groups,_ = get_group_connect_W(W_repeat,results['visual_groups'],results['secondary_groups'])
    rates_pair, inputs_pair = get_responses(results,W_pair,int(2e5),rec_net_pars,False)
    rates_repeat, inputs_repeat = get_responses(results,W_repeat,int(2e5),rec_net_pars,False)
    #rates_pair, inputs_pair, rates_repeat, inputs_repeat, W_pair, W_repeat = get_responses(results, results['W_plastic'], int(2e5), rec_net_pars, False, True)
    tone_0_response_pair = np.zeros(rates_pair.shape[1])
    non_tone_0_response_pair = np.zeros(rates_pair.shape[1])
    tone_0_present_pair = np.zeros(rates_pair.shape[0])
    tone_1_present_pair = np.zeros(rates_pair.shape[0])
    vis_0_present_pair = np.zeros(rates_pair.shape[0])
    vis_1_present_repeat = np.zeros(rates_pair.shape[0])
    assert rates_pair.shape[0] == inputs_pair.shape[0], 'using sample_res!=1 for frozen net! should use 1'
    tone_0_present_pair[inputs_pair[:, 1] == 0] = 1
    tone_1_present_pair[inputs_pair[:, 1] == 1] = 1
    vis_0_present_pair[inputs_pair[:, 0] == 0] = 1
    vis_0_response_pair = np.zeros(rates_pair.shape[1])
    vis_1_present_repeat[inputs_pair[:, 0] == 1] = 1
    vis_1_response_repeat = np.zeros(rates_pair.shape[1])

    tone_0_response_repeat = np.zeros(rates_repeat.shape[1])
    non_tone_0_response_repeat = np.zeros(rates_repeat.shape[1])
    tone_0_present_repeat = np.zeros(rates_repeat.shape[0])
    tone_0_present_repeat[inputs_repeat[:, 1] == 0] = 1
    tone_1_present_repeat = np.zeros(rates_repeat.shape[0])
    tone_1_present_repeat[inputs_repeat[:, 1] == 1] = 1
    for i in xrange(rates_pair.shape[1]):
        tone_0_response_pair[i] = np.mean(rates_pair[tone_0_present_pair == 1, i])
        non_tone_0_response_pair[i] = np.mean(rates_pair[tone_1_present_pair == 1, i])
        tone_0_response_repeat[i] = np.mean(rates_repeat[tone_0_present_repeat == 1, i])
        non_tone_0_response_repeat[i] = np.mean(rates_repeat[tone_1_present_repeat == 1, i])
        vis_0_response_pair[i] = np.mean(rates_pair[vis_0_present_pair == 1, i])
        vis_1_response_repeat[i] = np.mean(rates_pair[vis_1_present_repeat == 1, i])

    PC_plastic, tone_0_response,non_tone_0_response, _ , vis_0_response_before, vis_1_response_before = get_PC_responses(results,results['W_plastic'],5e5,False,rec_net_pars)
    #PC_depriv, tone_0_response_after,non_tone_0_response_after = get_PC_responses(results,results['W_depriv'],5e5)
    PC, _ , _, corr_matrix, _, _ = get_PC_responses(results,results['W_plastic'],5e5,True,rec_net_pars)
    PC_after, _ , _, corr_matrix_after, _, _ = get_PC_responses(results,W_pair,5e5,True,rec_net_pars)

    tone_0_response_group = [np.mean(tone_0_response[g]) for g in groups]
    tone_0_response_group_pair = [np.mean(tone_0_response_pair[g]) for g in groups]
    tone_0_response_group_repeat = [np.mean(tone_0_response_repeat[g]) for g in groups]
    non_tone_0_response_group = [np.mean(non_tone_0_response[g]) for g in groups]
    non_tone_0_response_group_pair = [np.mean(non_tone_0_response_pair[g]) for g in groups]
    non_tone_0_response_group_repeat = [np.mean(non_tone_0_response_repeat[g]) for g in groups]
    vis_0_response_group = [np.mean(vis_0_response_before[g]) for g in groups]
    vis_0_response_group_pair = [np.mean(vis_0_response_pair[g]) for g in groups]
    vis_1_response_group = [np.mean(vis_1_response_before[g]) for g in groups]
    vis_1_response_group_pair = [np.mean(vis_1_response_repeat[g]) for g in groups]
    PC_group = [np.mean(PC[g]) for g in groups]

    #W_before,_ = get_group_connect_W(results['W_plastic'],results['visual_groups'],results['secondary_groups'])
    #W_after,groups = get_group_connect_W(results['W_depriv'],results['visual_groups'],results['secondary_groups'])
    #W_change = W_after-W_before

    #PC_plastic, tone_0_response,non_tone_0_response = get_PC_responses(results,results['W_plastic'],5e5)
    #PC_depriv, tone_0_response_after,non_tone_0_response_after = get_PC_responses(results,results['W_depriv'],5e5)
    #PC, _ , _= get_PC_responses(results,results['W_plastic'],5e5,True)

    #import networkx as nx

    #net = nx.from_numpy_matrix(W_before)
    #net = nx.from_numpy_matrix(np.mean(W_before,axis=0))
    #pos = nx.spring_layout(net)
    #weights = [net[u][v]['weight'] * 20.0 for u, v in net.edges]
    #net_change = nx.from_numpy_matrix(W_change)
    #net_change = nx.from_numpy_matrix(np.mean(W_change,axis=0))
    #weights_change = [net_change[u][v]['weight'] * 20.0 for u, v in net.edges]

    #import matplotlib
    #jet = cm = plt.get_cmap('RdBu_r')
    #cNorm  = matplotlib.colors.Normalize(vmin=-2.0, vmax=2.0)
    #scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=jet)
    #colorList = []
    #for i in weights_change:
    #    colorVal = scalarMap.to_rgba(i)
    #    colorList.append(colorVal)

    #import seaborn as sns
    #color_palette = ['b','g','r','y']
    #color_palette = color_palette[:results['rec_net_pars']['N_secondary_groups']]
    #colors = ['black']*int(max(results['visual_groups'])+1) + color_palette*(1+int(max(results['visual_groups'])+1))

#    nx.draw(net, pos=pos, width=weights, edge_color=colorList,node_color=colors,node_size=tone_0_response_group)
#    plt.show()
#    nx.draw(net, pos=pos, width=weights, edge_color=colorList,node_color=colors,node_size=tone_0_response_group_after)
#    plt.show()
#    nx.draw(net, pos=pos, width=weights, edge_color=colorList,node_color=colors,node_size=PC_group)
#    plt.show()
#
#    plt.scatter(PC,PC_plastic)
#    plt.show()
#    plt.scatter(PC,PC_depriv)
#    plt.show()
#    plt.scatter(PC_plastic,PC_depriv)
#    plt.show()

#    results = {'PC':PC,'PC_group':PC_group,'tone_0_response_before':tone_0_response,'tone_0_response_after':tone_0_response_after,'non_tone_0_response':non_tone_0_response,
#               'non_tone_0_response_after':non_tone_0_response_after,'tone_0_groups_before':tone_0_response_group,'tone_0_groups_after':tone_0_response_group_after,
#               'W_group_before':W_before,'W_group_after':W_after}

    analysis_results = {'PC':PC,'groups':groups,'PC_group':PC_group,'tone_0_response_before':tone_0_response,'tone_0_response_pair':tone_0_response_pair,'non_tone_0_response_before':non_tone_0_response,
                'non_tone_0_response_pair':non_tone_0_response_pair,'tone_0_response_repeat':tone_0_response_repeat,'non_tone_0_response_repeat':non_tone_0_response_repeat,
               'tone_0_groups_before':tone_0_response_group,'tone_0_groups_pair':tone_0_response_group_pair,'tone_0_groups_repeat':tone_0_response_group_repeat, 'non_tone_0_groups_before':non_tone_0_response_group,
                        'non_tone_0_groups_pair': non_tone_0_response_group_pair, 'non_tone_0_groups_repeat': non_tone_0_response_group_repeat,
                'W_group_before':W_before,'W_group_pair':W_pair_groups,'W_group_repeat':W_repeat_groups,'W_pair':W_pair,'W_repeat':W_repeat,'W_plastic':results['W_plastic'],
                        'corr_matrix':corr_matrix, 'corr_matrix_after':corr_matrix_after, 'corrs_during_pair':corrs_during_pair,'tone_0_response_simultaneous':tone_0_response_simult,
                        'vis_0_response_pair':vis_0_response_pair,'vis_1_response_repeat':vis_1_response_repeat, 'vis_0_response_before':vis_0_response_before, 'vis_1_response_before': vis_1_response_before
                        ,'vis_0_response_group':vis_0_response_group,'vis_0_response_group_pair':vis_0_response_group_pair,'vis_1_response_group': vis_1_response_group,'vis_1_response_group_pair':vis_1_response_group_pair }


    return analysis_results #[PC_group,tone_0_response_group,tone_0_response_group_after,W_before,W_change,W_after]


def plot_results_multimodal(results):

    fig,axes = plt.subplots()
    axes.plot(results['pop_rate_plastic']+results['pop_rate_pruned'] + results['pop_rate_depriv'],lw=5)

    fig,axes = plt.subplots()
    axes.plot(np.append(np.append(results['sample_rates_plastic'],results['sample_rates_pruned'],axis=0),results['sample_rates_depriv'],axis=0))
    axes.plot(np.append(np.mean(results['sample_rates_plastic'],axis=1),np.mean(results['sample_rates_depriv'],axis=1),axis=0),'black',lw=3)

    axes.set_title('firing rates',fontsize=20)
    axes.set_xlabel('timesteps',fontsize=20)
    axes.set_ylabel('firing rate',fontsize=20)

    fig,axes = plt.subplots()
    axes.plot(np.append(np.append(results['sample_inh_rates_plastic'],results['sample_inh_rates_pruned'],axis=0),results['sample_inh_rates_depriv'],axis=0))
    axes.plot(np.append(np.mean(results['sample_inh_rates_plastic'],axis=1),np.mean(results['sample_inh_rates_depriv'],axis=1),axis=0),'black',lw=3)
    axes.set_title('inh firing rates',fontsize=20)
    axes.set_xlabel('timesteps',fontsize=20)
    axes.set_ylabel('firing rate',fontsize=20)

    fig,axes = plt.subplots()
    axes.plot(np.append(np.append(results['sample_weights'],results['sample_weights_pruned'],axis=0),results['sample_weights_depriv'],axis=0))
    axes.plot(np.append(results['sample_weights'],results['sample_weights_depriv'],axis=0))
    axes.set_title('sample synaptic weights',fontsize=20)
    axes.set_xlabel('timesteps',fontsize=20)
    axes.set_ylabel('synaptic weight',fontsize=20)

    fig,axes = plt.subplots()
    axes.plot(np.append(np.append(results['sample_inh_weights'],results['sample_inh_weights_pruned'],axis=0),results['sample_inh_weights_depriv'],axis=0))
    axes.plot(np.append(results['sample_inh_weights'],results['sample_inh_weights_depriv'],axis=0))
    axes.set_title('sample synaptic weights',fontsize=20)
    axes.set_xlabel('timesteps',fontsize=20)
    axes.set_ylabel('synaptic weight',fontsize=20)

    fig,axes = plt.subplots()
    axes.hist([results['x_plastic'],results['x_depriv']])
    axes.set_title('final firing rate distribution',fontsize=20)
    axes.set_xlabel('firing rate',fontsize=20)
    axes.set_ylabel('count',fontsize=20)

    fig,axes = plt.subplots()
    axes.hist(results['W_plastic'].flatten())
    axes.set_title('pre-depriv synaptic weight distribution',fontsize=20)
    axes.set_xlabel('synaptic weight',fontsize=20)
    axes.set_ylabel('count',fontsize=20)

    fig,axes = plt.subplots()
    axes.hist(results['W_depriv'].flatten())
    axes.set_title('post-paired synaptic weight distribution',fontsize=20)
    axes.set_xlabel('synaptic weight',fontsize=20)
    axes.set_ylabel('count',fontsize=20)

    fig,axes = plt.subplots()
    plt.pcolor(results['W_plastic'],cmap='RdBu_r',vmin=-1.0*rec_net.W_max,vmax=rec_net.W_max)
    axes.set_title('pre-paired W',fontsize=20)
    plt.colorbar()


    W_exc = results['W_plastic'][results['secondary_groups']>=0].copy()
    W_exc = W_exc[:,results['secondary_groups']>=0]
    sg = list(results['secondary_groups'])
    while -1 in sg: sg.remove(-1)
    groups_sorted = np.argsort(sg)
    W_exc_sorted = W_exc[:,groups_sorted]
    W_exc_sorted = W_exc_sorted[groups_sorted,:]

    fig,axes = plt.subplots()
    plt.pcolor(W_exc_sorted,cmap='RdBu_r',vmin=-1.0*rec_net.W_max,vmax=rec_net.W_max)
    axes.set_title('auditory W',fontsize=20)
    plt.colorbar()

    W_exc = results['W_plastic'][results['visual_groups']>=0].copy()
    W_exc = W_exc[:,results['visual_groups']>=0]
    sg = list(results['visual_groups'])
    while -1 in sg: sg.remove(-1)
    groups_sorted = np.argsort(sg)
    W_exc_sorted = W_exc[:,groups_sorted]
    W_exc_sorted = W_exc_sorted[groups_sorted,:]

    fig,axes = plt.subplots()
    plt.pcolor(W_exc_sorted,cmap='RdBu_r',vmin=-1.0*rec_net.W_max,vmax=rec_net.W_max)
    axes.set_title('visual W',fontsize=20)
    plt.colorbar()

    fig,axes = plt.subplots()
    plt.pcolor(results['W_pruned'],cmap='RdBu_r',vmin=-1.0*rec_net.W_max,vmax=rec_net.W_max)
    axes.set_title('pruned W',fontsize=20)
    plt.colorbar()

    within_visual_weights = []

    for i,j in itertools.permutations(range(rec_net.N_E),2):
        if results['visual_groups'][i] == results['visual_groups'][j]:
            within_visual_weights.append(results['W_plastic'][i,j])
            within_visual_weights.append(results['W_plastic'][j,i])

    within_auditory_weights = []

    for i,j in itertools.permutations(range(rec_net.N_E),2):
        if results['secondary_groups'][i] == results['secondary_groups'][j]:
            within_visual_weights.append(results['W_plastic'][i,j])
            within_visual_weights.append(results['W_plastic'][j,i])


    print 'within visual ', np.mean(within_visual_weights), np.std(within_visual_weights)
    print 'within auditory ', np.mean(within_auditory_weights), np.std(within_auditory_weights)

    fig,axes = plt.subplots()
    plt.pcolor(results['W_depriv'],cmap='RdBu_r',vmin=-1.0*rec_net.W_max,vmax=rec_net.W_max)
    axes.set_title('post-paired W',fontsize=20)
    plt.colorbar()


    W_exc = results['W_depriv'][results['secondary_groups']>=0].copy()
    W_exc = W_exc[:,results['secondary_groups']>=0]
    sg = list(results['secondary_groups'])
    while -1 in sg: sg.remove(-1)
    groups_sorted = np.argsort(sg)
    W_exc_sorted = W_exc[:,groups_sorted]
    W_exc_sorted = W_exc_sorted[groups_sorted,:]

    fig,axes = plt.subplots()
    plt.pcolor(W_exc_sorted,cmap='RdBu_r',vmin=-1.0*rec_net.W_max,vmax=rec_net.W_max)
    axes.set_title('auditory post-paired W',fontsize=20)
    plt.colorbar()

    W_exc = results['W_depriv'][results['visual_groups']>=0].copy()
    W_exc = W_exc[:,results['visual_groups']>=0]
    sg = list(results['visual_groups'])
    while -1 in sg: sg.remove(-1)
    groups_sorted = np.argsort(sg)
    W_exc_sorted = W_exc[:,groups_sorted]
    W_exc_sorted = W_exc_sorted[groups_sorted,:]

    fig,axes = plt.subplots()
    plt.pcolor(W_exc_sorted,cmap='RdBu_r',vmin=-1.0*rec_net.W_max,vmax=rec_net.W_max)
    axes.set_title('visual post-paired W',fontsize=20)
    plt.colorbar()

def run_multimodal_net(T_plastic=int(1e5),T_prune=int(1e5),T_pairstim =int(1e5),seed_type='diverse',savefile='default_run.pkl',rec_net_pars={}):
    rec_net.dt = .05

    rec_net.N_E = 80
    rec_net.N_I = 20

    rec_net.r_max= 20.0

    rec_net.N_orientations = 4

    recurrent_factor = 1.0

    rec_net.cbar = recurrent_factor*1.05/rec_net.N_E
    rec_net.w_theta = 0*0.25*recurrent_factor*1.3/rec_net.N_E
    rec_net.a = rec_net.cbar
    rec_net.b = 5.0*rec_net.cbar

    rec_net.ext_OU_noise = True
    rec_net.ext_OU_tau = 50.0
    #rec_net.ext_OU_sigma= (.2**2)*(1.0/rec_net.dt + rec_net.ext_OU_tau)/rec_net.ext_OU_tau
    rec_net.ext_OU_sigma = 0.0

    x = np.zeros((rec_net.N_E+rec_net.N_I,1))

    H_0 = 3.0
    rec_net.H = np.ones((rec_net.N_E+rec_net.N_I))*H_0

    rec_net.H_max = 7.0 #*H_0

    rec_net.H_min = H_0

    for key in rec_net_pars.keys():
        setattr(rec_net,key,rec_net_pars[key])

    #W,Theta,C_E,orientation_prefs = rec_net.generate_connectivity_matrix(rec_net.N_E,rec_net.N_I)
    W = np.zeros((rec_net.N_E+rec_net.N_I,rec_net.N_E+rec_net.N_I))

    rec_net.W_max = 0.08*recurrent_factor #0.25 #1.0*recurrent_factor

    rec_net.W_inh_min = -0.04*recurrent_factor

    rec_net.alpha = 1e-8*0.1*recurrent_factor*2.5
    rec_net.BCM_target = 5.0

    rec_net.BCM_lambda= 1.0

    rec_net.eta = rec_net.alpha*1.0*1.0
    rec_net.pruned_synapses = False


    rec_net.theta_BCM = np.ones(rec_net.N_E)*rec_net.BCM_target*1.10
    rec_net.theta_BCM_dt = 1.0e-5

    #rec_net.N_secondary_groups = 2
    rec_net.fraction_secondary = 1.0

    #_secondary_groups = np.array([[x]*(rec_net.N_E/rec_net.N_secondary_groups) for x in xrange(rec_net.N_secondary_groups)]).flatten()
    #np.random.shuffle(_secondary_groups)
    #rec_net.secondary_groups = _secondary_groups[:rec_net.N_E*rec_net.fraction_secondary]

    rec_net.H_max_variable = np.array([1.0,1.0,1.0,1.0])*0.5*rec_net.H_max
    rec_net.H_secondary_variable = np.array([1.0,1.0,1.0,1.0])*0.5*rec_net.H_secondary

    rec_net.H_visual_baseline = rec_net.H_max*0.0
    rec_net.H_auditory_baseline = rec_net.H_secondary*0.0

    rec_net.visual_responsive = np.zeros(rec_net.N_E)
    rec_net.visual_responsive[:int(rec_net.N_E*(1.0-rec_net.frac_nonresponsive)*(1.0-rec_net.frac_A_only))] = 1.0
    rec_net.visual_responsive[int(rec_net.N_E*(1.0-rec_net.frac_nonresponsive)*(1.0-rec_net.frac_AV)):int(rec_net.N_E*(1.0-rec_net.frac_nonresponsive)*(1.0-rec_net.frac_A_only))] = 1.0
    rec_net.auditory_responsive = np.zeros(rec_net.N_E)
    rec_net.auditory_responsive[:int(rec_net.N_E*(1.0-rec_net.frac_nonresponsive))] = 1.0
    rec_net.auditory_responsive[:int(rec_net.N_E*(1.0-rec_net.frac_nonresponsive)*rec_net.frac_V_only)] = 0.0
    rec_net.auditory_responsive[int(rec_net.N_E*(1.0-rec_net.frac_nonresponsive)*rec_net.frac_V_only):int(rec_net.N_E*(1.0-rec_net.frac_nonresponsive)*rec_net.frac_AV)] = 1.0

    #rec_net.auditory_responsive[:rec_net.N_E*0.5] = 0.0
    #rec_net.auditory_responsive[rec_net.N_E*0.5:rec_net.N_E*0.75] = 0.5

    _secondary_groups = np.array([[x]*(sum(rec_net.auditory_responsive>0)/rec_net.N_secondary_groups) for x in xrange(rec_net.N_secondary_groups)]).flatten()
    _visual_groups = np.array([[x]*(sum(rec_net.visual_responsive>0)/rec_net.N_orientations) for x in xrange(rec_net.N_orientations)]).flatten()

    np.random.shuffle(_secondary_groups)
    np.random.shuffle(_visual_groups)
    sg = np.ones(rec_net.N_E)*-1
    sg[rec_net.auditory_responsive>0] = _secondary_groups
    vg = np.ones(rec_net.N_E)*-1
    vg[rec_net.visual_responsive>0] = _visual_groups
    rec_net.secondary_groups = sg
    rec_net.visual_groups = vg


    #for g_idx in rec_net.H_secondary_variable:
    #    if rec_net.H_secondary_variable[g_idx] == 0.0:
    #        rec_net.auditory_responsive[rec_net.secondary_groups == g_idx] = 0

    T_static = 50000
    T_measure = 10000

    prob_prune = 1.0
    rec_net.prune_threshold = 0.2

    T = T_static+T_plastic+T_prune+T_pairstim

    rec_net.T_input_gen = min(100000,T_plastic/10)

    rec_net.N_sample = rec_net.N_E
    rec_net.sample_res = 1000
    rec_net.checkpoint_res = 1e4

    for key in rec_net_pars.keys():
        setattr(rec_net,key,rec_net_pars[key])

    if rec_net_pars.has_key('init_Winh'):
        W[:rec_net.N_E,rec_net.N_E:] = rec_net_pars['init_Winh']
    if rec_net_pars.has_key('init_Wexc'):
        W[:rec_net.N_E,:rec_net.N_E] = rec_net_pars['init_Wexc']

    stgen_drive = stgen.StGen()
    rec_net.OU_drive = stgen_drive.OU_generator(1.,10.0,H_0,0.,0.,T).signal

    pop_rate = []

    x = np.zeros((rec_net.N_E+rec_net.N_I,1))

    x,W,pop_rate_static,sample_rates_static,sample_inh_rates_static,_inputs = rec_net.run_net_static_input_type(x,W,T_static,rec_net.N_sample)

    x_static = x.copy()
    W_static = W.copy()

    y_bar = x.copy()[:rec_net.N_E]
    rec_net.theta_BCM = np.ones(rec_net.N_E)*rec_net.BCM_target*1.10

    print 'theta_BCM', rec_net.theta_BCM

    #x,W,pop_rate_plastic,sample_rates_plastic,sample_weights,sample_inh_weights,mean_incoming_weight,theta_BCM,sample_theta_BCM,sample_inh_rates_plastic = rec_net.run_net_plastic_BCM_and_inh_plasticity(x,W,rec_net.theta_BCM,T_plastic,rec_net.N_sample,'orientation_and_secondary_dynamic',rec_net.N_orientations)
    #x,W,pop_rate_plastic,sample_rates_plastic,sample_weights,sample_inh_weights,mean_incoming_weight,theta_BCM,sample_theta_BCM,sample_inh_rates_plastic = rec_net.run_net_static_threshold_exc_and_inh_plasticity(x,W,rec_net.theta_BCM,T_plastic,rec_net.N_sample,'audiovisual_random',rec_net.N_orientations)
    x,W,pop_rate_plastic,sample_rates_plastic,sample_weights,sample_inh_weights,mean_incoming_weight,theta_BCM,sample_theta_BCM,sample_inh_rates_plastic = rec_net.run_net_plastic_exc_BCM_and_inh_plasticity(x,W,rec_net.theta_BCM,T_plastic,rec_net.N_sample,'audiovisual_random',rec_net.N_orientations,0,savefile)
    #x,W,pop_rate_plastic,sample_rates_plastic,sample_weights,mean_incoming_weight,theta_BCM,sample_theta_BCM,sample_inh_rates_plastic = rec_net.run_net_plastic_sliding_threshold(x,W,rec_net.theta_BCM,T_plastic,rec_net.N_sample,'audiovisual_random',rec_net.N_orientations)
    #x,W,pop_rate_plastic,sample_rates_plastic,sample_weights,sample_inh_weights,theta_BCM,sample_theta_BCM,mean_incoming_weight,sample_inh_rates_plastic = rec_net.run_net_pure_Hebbian_EE_and_inh_plasticity(x,W,rec_net.theta_BCM,T_plastic,rec_net.N_sample,'audiovisual_random',rec_net.N_orientations)
    #x,W,pop_rate_plastic,sample_rates_plastic,sample_weights,mean_incoming_weight,theta_BCM,sample_theta_BCM,sample_inh_rates_plastic = rec_net.run_net_plastic_sliding_threshold_taro_fluct(x,W,rec_net.theta_BCM,T_plastic,rec_net.N_sample,'audiovisual_random',rec_net.N_orientations)
    #x,W,pop_rate_plastic,sample_rates_plastic,sample_weights,sample_inh_weights,mean_incoming_weight,theta_BCM,sample_theta_BCM,sample_inh_rates_plastic = rec_net.run_net_static_threshold_exc_and_inh_plasticity_taro_fluct(x,W,rec_net.theta_BCM,T_plastic,rec_net.N_sample,'audiovisual_random',rec_net.N_orientations)

    x_plastic = x.copy()
    W_plastic = W.copy()
    theta_BCM_plastic = theta_BCM.copy()

    # Pruning synapses which are below 0.1 W_max
    W_plastic_pruned = W_plastic.copy()[:,:rec_net.N_E]<rec_net.W_max*rec_net.prune_threshold
    W_plastic_pruned[W_plastic_pruned==1] = np.random.binomial(1,prob_prune,W_plastic_pruned[W_plastic_pruned==1].shape)
    rec_net.W_pruned = W_plastic_pruned
    rec_net.pruned_synapses = True

    print 'pruned with threshold ', rec_net.prune_threshold

    #x,W,pop_rate_pruned,sample_rates_pruned,sample_weights_pruned,sample_inh_weights_pruned,mean_incoming_weight,theta_BCM,sample_theta_BCM_pruned,sample_inh_rates_pruned = rec_net.run_net_plastic_BCM_and_inh_plasticity(x,W,rec_net.theta_BCM,T_prune,rec_net.N_sample,'orientation_and_secondary_dynamic',rec_net.N_orientations)
    #x,W,pop_rate_pruned,sample_rates_pruned,sample_weights_pruned,sample_inh_weights_pruned,mean_incoming_weight,theta_BCM,sample_theta_BCM_pruned,sample_inh_rates_pruned = rec_net.run_net_static_threshold_exc_and_inh_plasticity(x,W,rec_net.theta_BCM,T_prune,rec_net.N_sample,'audiovisual_random',rec_net.N_orientations)
    x,W,pop_rate_pruned,sample_rates_pruned,sample_weights_pruned,sample_inh_weights_pruned,mean_incoming_weight,theta_BCM,sample_theta_BCM_pruned,sample_inh_rates_pruned = rec_net.run_net_plastic_exc_BCM_and_inh_plasticity(x,W,rec_net.theta_BCM,T_prune,rec_net.N_sample,'audiovisual_random',rec_net.N_orientations)
    #x,W,pop_rate_pruned,sample_rates_pruned,sample_weights_pruned,mean_incoming_weight,theta_BCM,sample_theta_BCM_pruned,sample_inh_rates_pruned = rec_net.run_net_plastic_sliding_threshold(x,W,rec_net.theta_BCM,T_prune,rec_net.N_sample,'audiovisual_random',rec_net.N_orientations)
    #x,W,pop_rate_pruned,sample_rates_pruned,sample_weights_pruned,sample_inh_weights_pruned,theta_BCM,sample_theta_BCM_pruned,mean_incoming_weight,sample_inh_rates_pruned = rec_net.run_net_pure_Hebbian_EE_and_inh_plasticity(x,W,rec_net.theta_BCM,T_prune,rec_net.N_sample,'audiovisual_random',rec_net.N_orientations)
    #x,W,pop_rate_pruned,sample_rates_pruned,sample_weights_pruned,mean_incoming_weight,theta_BCM,sample_theta_BCM_pruned,sample_inh_rates_pruned = rec_net.run_net_plastic_sliding_threshold_taro_fluct(x,W,rec_net.theta_BCM,T_prune,rec_net.N_sample,'audiovisual_random',rec_net.N_orientations)
    #x,W,pop_rate_pruned,sample_rates_pruned,sample_weights_pruned,sample_inh_weights_pruned,mean_incoming_weight,theta_BCM,sample_theta_BCM_pruned,sample_inh_rates_pruned = rec_net.run_net_static_threshold_exc_and_inh_plasticity_taro_fluct(x,W,rec_net.theta_BCM,T_prune,rec_net.N_sample,'audiovisual_random',rec_net.N_orientations)


    x_pruned = x.copy()
    W_pruned = W.copy()
    theta_BCM_pruned  = theta_BCM.copy()

    #rec_net.H_max = 0
    #rec_net.H_min = 1.0

    rec_net.secondary_paired_idx = [0,None,None,None,None]

    #x,W,pop_rate_depriv,sample_rates_depriv,sample_weights_depriv,sample_inh_weights_depriv,mean_incoming_weight_depriv,theta_BCM,sample_theta_BCM_depriv,sample_inh_rates_depriv = rec_net.run_net_plastic_BCM_and_inh_plasticity(x,W,theta_BCM,T_depriv,rec_net.N_sample,'orientation_and_secondary_dynamic',rec_net.N_orientations)
    #x,W,pop_rate_depriv,sample_rates_depriv,sample_weights_depriv,sample_inh_weights_depriv,mean_incoming_weight_depriv,theta_BCM,sample_theta_BCM_depriv,sample_inh_rates_depriv = rec_net.run_net_static_threshold_exc_and_inh_plasticity(x,W,theta_BCM,T_pairstim,rec_net.N_sample,'audiovisual_paired',rec_net.N_orientations)
    #x,W,pop_rate_depriv,sample_rates_depriv,sample_weights_depriv,sample_inh_weights_depriv,mean_incoming_weight_depriv,theta_BCM,sample_theta_BCM_depriv,sample_inh_rates_depriv = rec_net.run_net_plastic_exc_BCM_and_inh_plasticity(x,W,theta_BCM,T_pairstim,rec_net.N_sample,'audiovisual_paired_only_0',rec_net.N_orientations)
    x,W,pop_rate_depriv,sample_rates_depriv,sample_weights_depriv,sample_inh_weights_depriv,mean_incoming_weight_depriv,theta_BCM,sample_theta_BCM_depriv,sample_inh_rates_depriv = rec_net.run_net_plastic_exc_BCM_and_inh_plasticity(x,W,theta_BCM,T_pairstim,rec_net.N_sample,'audiovisual_repeating_only_tone_0',rec_net.N_orientations)
    #x,W,pop_rate_depriv,sample_rates_depriv,sample_weights_depriv,sample_inh_weights_depriv,mean_incoming_weight_depriv,theta_BCM,sample_theta_BCM_depriv,sample_inh_rates_depriv = rec_net.run_net_static_threshold_exc_and_inh_plasticity(x,W,theta_BCM,T_pairstim,rec_net.N_sample,'audiovisual_paired_only_0',rec_net.N_orientations)
    #x,W,pop_rate_depriv,sample_rates_depriv,sample_weights_depriv,sample_inh_weights_depriv,mean_incoming_weight_depriv,theta_BCM,sample_theta_BCM_depriv,sample_inh_rates_depriv = rec_net.run_net_static_threshold_exc_and_inh_plasticity(x,W,theta_BCM,T_pairstim,rec_net.N_sample,'audiovisual_repeating_only_tone_0',rec_net.N_orientations)
    #x,W,pop_rate_depriv,sample_rates_depriv,sample_weights_depriv,mean_incoming_weight_depriv,theta_BCM,sample_theta_BCM_depriv,sample_inh_rates_depriv = rec_net.run_net_plastic_sliding_threshold(x,W,theta_BCM,T_pairstim,rec_net.N_sample,'audiovisual_paired',rec_net.N_orientations)
    #x,W,pop_rate_depriv,sample_rates_depriv,sample_weights_depriv,sample_inh_weights_depriv,theta_BCM,sample_theta_BCM_depriv,mean_incoming_weight_depriv,sample_inh_rates_depriv = rec_net.run_net_pure_Hebbian_EE_and_inh_plasticity(x,W,rec_net.theta_BCM,T_pairstim,rec_net.N_sample,'audiovisual_paired',rec_net.N_orientations)
    #x,W,pop_rate_depriv,sample_rates_depriv,sample_weights_depriv,mean_incoming_weight_depriv,theta_BCM,sample_theta_BCM_depriv,sample_inh_rates_depriv = rec_net.run_net_plastic_sliding_threshold_taro_fluct(x,W,theta_BCM,T_pairstim,rec_net.N_sample,'audiovisual_paired',rec_net.N_orientations)
    #x,W,pop_rate_depriv,sample_rates_depriv,sample_weights_depriv,sample_inh_weights_depriv,mean_incoming_weight_depriv,theta_BCM,sample_theta_BCM_depriv,sample_inh_rates_depriv = rec_net.run_net_static_threshold_exc_and_inh_plasticity_taro_fluct(x,W,theta_BCM,T_pairstim,rec_net.N_sample,'audiovisual_paired',rec_net.N_orientations)

    x_depriv = x.copy()
    W_depriv = W.copy()

    results = {
        'pop_rate_plastic': pop_rate_plastic,
        'pop_rate_depriv': pop_rate_depriv,
        'x_static': x_static,
        'x_pruned': x_pruned,
        'x_plastic': x_plastic,
        'x_depriv': x_depriv,
        'W_static': W_static,
        'W_plastic': W_plastic,
        'W_plastic_pruned': W_plastic_pruned,
        'W_depriv': W_depriv,
        'sample_weights': sample_weights,
        'sample_weights_depriv': sample_weights_depriv,
        'sample_theta_BCM': sample_theta_BCM,
        'sample_theta_BCM_depriv': sample_theta_BCM_depriv,
        'sample_rates_plastic': sample_rates_plastic,
        'sample_rates_static': sample_rates_static,
        'sample_rates_depriv': sample_rates_depriv,
        'sample_inh_rates_plastic': sample_inh_rates_plastic,
        'sample_inh_rates_depriv': sample_inh_rates_depriv,
        'pop_rate_pruned': pop_rate_pruned,
        'W_pruned': W_pruned,
        'W_plastic_pruned': W_plastic_pruned,
        'sample_weights_pruned': sample_weights_pruned,
        'sample_theta_BCM_pruned': sample_theta_BCM_pruned,
        'sample_rates_pruned': sample_rates_pruned,
        'sample_inh_rates_pruned': sample_inh_rates_pruned,
        'sample_inh_weights': sample_inh_weights,
        'sample_inh_weights_pruned': sample_inh_weights_pruned,
        'sample_inh_weights_depriv': sample_inh_weights_depriv,
        'secondary_groups': rec_net.secondary_groups,
        'visual_groups': rec_net.visual_groups,
        'rec_net_pars': rec_net_pars
    }

    if not savefile == None:
        py_scripts_yann.save_pickle_safe('pkl_results/'+savefile, results)

    return results


def get_responses(frozen_results,W,T,rec_net_pars={},spont_activity=False,do_pairing_experiments=False,T_pairstim
                   =int(1e4),simultaneous_presentation=False):
    rec_net.dt = .05

    rec_net.N_E = 80
    rec_net.N_I = 20

    rec_net.r_max= 20.0

    rec_net.ext_OU_noise = True
    rec_net.ext_OU_tau = 50.0
    rec_net.ext_OU_sigma = 0.0

    H_0 = 3.0
    rec_net.H = np.ones((rec_net.N_E+rec_net.N_I))*H_0
    rec_net.H_max = 7.0 #*H_0
    rec_net.H_min = H_0

    for key in rec_net_pars.keys():
        setattr(rec_net,key,rec_net_pars[key])

    x = np.zeros((rec_net.N_E+rec_net.N_I,1))

    rec_net.visual_responsive = np.zeros(rec_net.N_E)
    rec_net.visual_responsive[:int(rec_net.N_E*(1.0-rec_net.frac_nonresponsive)*(1.0-rec_net.frac_A_only))] = 1.0
    rec_net.visual_responsive[int(rec_net.N_E*(1.0-rec_net.frac_nonresponsive)*(1.0-rec_net.frac_AV)):int(rec_net.N_E*(1.0-rec_net.frac_nonresponsive)*(1.0-rec_net.frac_A_only))] = 0.5
    rec_net.auditory_responsive = np.zeros(rec_net.N_E)
    rec_net.auditory_responsive[:int(rec_net.N_E*(1.0-rec_net.frac_nonresponsive))] = 1.0
    rec_net.auditory_responsive[:int(rec_net.N_E*(1.0-rec_net.frac_nonresponsive)*rec_net.frac_V_only)] = 0.0
    rec_net.auditory_responsive[int(rec_net.N_E*(1.0-rec_net.frac_nonresponsive)*rec_net.frac_V_only):int(rec_net.N_E*(1.0-rec_net.frac_nonresponsive)*rec_net.frac_AV)] = 0.5

    #rec_net.auditory_responsive[:rec_net.N_E*0.5] = 0.0
    #rec_net.auditory_responsive[rec_net.N_E*0.5:rec_net.N_E*0.75] = 0.
    rec_net.H_visual_baseline = rec_net.H_max*0.0
    rec_net.H_auditory_baseline = rec_net.H_secondary*0.0

    rec_net.secondary_groups = frozen_results['secondary_groups']
    rec_net.visual_groups = frozen_results['visual_groups']
    #rec_net.N_secondary_groups = 2
    rec_net.N_orientations = 4



    rec_net.T_input_gen = T # min(100000,T/10)

    rec_net.N_sample = rec_net.N_E
    rec_net.sample_res = 1
    rec_net.checkpoint_res = 1e6

    if do_pairing_experiments:
        rec_net.secondary_paired_idx = [0,None,None,None,None]

        if rec_net_pars['pairing_pars']['prune_synapses']:
            W_plastic_pruned = W.copy()[:, :rec_net.N_E] < rec_net.W_max * rec_net_pars['pairing_pars']['prune_threshold']
            rec_net.W_pruned = W_plastic_pruned
            rec_net.pruned_synapses = True
        if rec_net_pars['pairing_pars']['set_theta_BCM']:
            rec_net.theta_BCM = np.ones(rec_net.N_E)*rec_net_pars['pairing_pars']['theta_BCM']
            frozen_results['sample_theta_BCM'][-1] = np.ones(rec_net.N_E)*rec_net_pars['pairing_pars']['theta_BCM']

        x_depriv,W_interleave,pop_rate_depriv,sample_rates_depriv,sample_weights_depriv,sample_inh_weights_depriv,mean_incoming_weight_depriv,theta_BCM,sample_theta_BCM_depriv,sample_inh_rates_depriv = rec_net.run_net_static_threshold_exc_and_inh_plasticity(frozen_results['x_plastic'],W.copy(),frozen_results['sample_theta_BCM'][-1],T_pairstim,rec_net.N_sample,'audiovisual_interleaved_paired_0_1_unpaired_tone_1_unpaired_grating_1',rec_net.N_orientations)
        #x_depriv,W_interleave,pop_rate_depriv,sample_rates_depriv,sample_weights_depriv,sample_inh_weights_depriv,mean_incoming_weight_depriv,theta_BCM,sample_theta_BCM_depriv,sample_inh_rates_depriv = rec_net.run_net_static_threshold_exc_and_inh_plasticity(frozen_results['x_plastic'],W.copy(),frozen_results['sample_theta_BCM'][-1],T_pairstim,rec_net.N_sample,'audiovisual_paired_only_0',rec_net.N_orientations)

        #x_depriv,W_pair,pop_rate_depriv,sample_rates_depriv,sample_weights_depriv,sample_inh_weights_depriv,mean_incoming_weight_depriv,theta_BCM,sample_theta_BCM_depriv,sample_inh_rates_depriv = rec_net.run_net_static_threshold_exc_and_inh_plasticity(frozen_results['x_plastic'],W.copy(),frozen_results['sample_theta_BCM'][-1],T_pairstim,rec_net.N_sample,'audiovisual_paired_only_0',rec_net.N_orientations)
        #x_depriv,W_pair,pop_rate_depriv,sample_rates_depriv,sample_weights_depriv,sample_inh_weights_depriv,mean_incoming_weight_depriv,theta_BCM,sample_theta_BCM_depriv,sample_inh_rates_depriv = rec_net.run_net_plastic_exc_BCM_and_inh_plasticity(frozen_results['x_plastic'],W.copy(),frozen_results['sample_theta_BCM'][-1],T_pairstim,rec_net.N_sample,'audiovisual_paired_only_0',rec_net.N_orientations)
        x,W,pop_rate,sample_rates_pair,sample_inh_rates,inputs_pair = rec_net.run_net_static_input_type(x_depriv,W_interleave,T,rec_net.N_sample,'audiovisual_random_separate',rec_net.N_orientations)

        #x_depriv,W_repeat,pop_rate_depriv,sample_rates_depriv,sample_weights_depriv,sample_inh_weights_depriv,mean_incoming_weight_depriv,theta_BCM,sample_theta_BCM_depriv,sample_inh_rates_depriv = rec_net.run_net_static_threshold_exc_and_inh_plasticity(frozen_results['x_plastic'],W.copy(),frozen_results['sample_theta_BCM'][-1],T_pairstim,rec_net.N_sample,'audiovisual_repeating_only_tone_0',rec_net.N_orientations)
        #x_depriv,W_repeat,pop_rate_depriv,sample_rates_depriv,sample_weights_depriv,sample_inh_weights_depriv,mean_incoming_weight_depriv,theta_BCM,sample_theta_BCM_depriv,sample_inh_rates_depriv = rec_net.run_net_plastic_exc_BCM_and_inh_plasticity(frozen_results['x_plastic'],W.copy(),frozen_results['sample_theta_BCM'][-1],T_pairstim,rec_net.N_sample,'audiovisual_repeating_only_tone_0',rec_net.N_orientations)
        x,W,pop_rate,sample_rates_repeat,sample_inh_rates,inputs_repeat = rec_net.run_net_static_input_type(x_depriv,W_interleave,T,rec_net.N_sample,'audiovisual_random_separate',rec_net.N_orientations)
        #return sample_rates_pair,inputs_pair, sample_rates_repeat,inputs_repeat, W_pair, W_repeat
        return sample_rates_pair,inputs_pair, sample_rates_repeat,inputs_repeat, W_interleave, W_interleave
    elif spont_activity:
        rec_net.ext_OU_sigma = 2.5
        x,W,pop_rate,sample_rates,sample_inh_rates,inputs = rec_net.run_net_static_input_type(frozen_results['x_depriv'],W,T,rec_net.N_sample,'random_dynamic',rec_net.N_orientations,0.0)
    elif simultaneous_presentation:
        rec_net.secondary_paired_idx = [0,None,None,None,None]
        x,W,pop_rate,sample_rates,sample_inh_rates,inputs = rec_net.run_net_static_input_type(frozen_results['x_depriv'],W,T,rec_net.N_sample,'audiovisual_interleaved_paired_0_1_unpaired_tone_1',rec_net.N_orientations,0.0)
    else:
        x,W,pop_rate,sample_rates,sample_inh_rates,inputs = rec_net.run_net_static_input_type(frozen_results['x_depriv'],W,T,rec_net.N_sample,'audiovisual_random_separate',rec_net.N_orientations)

    return sample_rates,inputs
