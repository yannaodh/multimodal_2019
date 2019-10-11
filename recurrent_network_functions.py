
# coding: utf-8

# In[1]:

import numpy as np
import itertools

from matplotlib import pyplot as plt
from scipy import stats

import py_scripts_yann
from pdb import set_trace as bp

#get_ipython().magic(u'matplotlib qt')

N_E = 80
N_I = 20

w_theta = 1.3/N_E

cbar = 1.05/N_E #original value from Okun (2015)
#cbar = 5.0/N_E

a = cbar
b = 5.0*cbar

r_0 = 1.0
r_max = 100.0 #maybe should set to more realistic value (e.g. =20)?

dt = .001

alpha = 0.01
BCM_lambda = 0.9

theta_BCM = 1.0

theta_BCM = np.ones(N_E)*1.0

BCM_target = 2.0
theta_BCM_dt = .001

W_max = 1.0

pruned_synapses = False
continual_pruning = False

synaptic_scaling = False

# In[2]:

#%%px

def generate_connectivity_matrix(N_E,N_I,C_E_dist='uniform'):

    # uniform values
    if C_E_dist=='uniform':
        C_E = np.ones(N_E)*cbar
    elif C_E_dist=='random uniform':
        # random uniform distribution
        C_E = np.random.uniform(0,2*cbar,N_E)
    elif C_E_dist=='gamma':
        # starting with only gamma distribution initially
        C_E = np.random.gamma(1,cbar,N_E)

    # sort C_E
    #C_E.sort()

    # less skewed gamma
    #C_E = np.random.gamma(5,cbar,N_E)

    print 'NE  ', N_E

    # addition of gamma and beta distribution
    # gamma_beta_ratio = 0.5
    #C_E = np.random.beta(1,cbar*(1-gamma_beta_ratio),N_E)+np.random.gamma(1,cbar,gamma_beta_ratio,N_E)

    # initially all excitatory neuron have same preferred orientation of 0 (or 90) degrees
    #theta_i = np.zeros(N_E)
    #theta_i = np.ones(N_E)*np.pi/4

    # excitatory neuron have random orientation on uniform distribution between 0 and 180 degrees
    #theta_i = np.random.uniform(0,np.pi,N_E)

    # testing; with first half  preferred 0 degrees, second half preferred 180
    #theta_i = np.append(np.zeros(N_E/2),np.pi/2*(np.ones(N_E/2)))

    #N_orientations = 8
    orientation_n = [np.pi*x/(N_orientations) for x in xrange(N_orientations)]
    theta_i  = np.array([np.ones(N_E/N_orientations)*x for x in orientation_n]).flatten()

    # assuming all neurons have same orientation
    Theta = w_theta*np.ones((N_E,N_E))

    # if neurons have different orientations, get differences:
    iter_ij = itertools.permutations(range(N_E),2)
    for (i,j) in iter_ij:
        Theta[i,j] = theta_i[i]-theta_i[j]
    #Theta = w_theta*np.cos(Theta)
    # setting so that Theta is always positive
    Theta = w_theta*(np.cos(Theta)+1.0)

    C_I = np.ones(N_I)*cbar

    C = np.append(C_E,C_I)

    #lp_noise_scale = 2.5/N_E
    lp_noise_scale = .05/N_E

    # will start with no noise, as this leads to negative E-E weights
    lp_noise_scale = 0

    lp_noise = (np.random.normal(0,1,(N_E,N_E))**3)*lp_noise_scale

    Theta = Theta + lp_noise

    W = np.zeros((N_E+N_I,N_E+N_I))

    exc_idx = range(N_E)
    inh_idx = range(N_E,N_E+N_I)

    iter_EE = itertools.product(range(N_E),range(N_E))
    for (i,j) in iter_EE:
        W[i,j] = C_E[i] + Theta[i,j]

    iter_EI = itertools.product(inh_idx,range(N_E))
    for (i,j) in iter_EI:
        W[i,j] = C[i]

    iter_IE = itertools.product(range(N_E),inh_idx)
    for (i,j) in iter_IE:
        W[i,j] = -a - b*C_E[i]

    iter_II = itertools.product(range(N_I),inh_idx)
    for (i,j) in iter_II:
        W[i,j] = -a - b*C[i]

    #iter_inh_pre = itertools.product(range(N_E+N_I),range(N_I))

    return W,Theta,C_E,theta_i

def get_rate(x):
    if x>0:
        return r_0*np.tanh(x/r_0)
    else:
        return (r_max-r_0)*np.tanh(x/(r_max-r_0))

def update_rates(x):
    #rates = x
    x[x<=0] = r_0*np.tanh(x[x<=0]/r_0)
    x[x>0] = (r_max-r_0)*np.tanh(x[x>0]/(r_max-r_0))

    return x

    # linear
    #print 'max rate x ', max(x)
    #return x


# BCM: tau_w*(dw/dt) = xy(y-theta_BCM)

def update_weights(_x,_W,_theta_BCM):
    _x=_x[:N_E]
    # only half of excitatory neurons are plastic
    #x=x[:N_E/2]

    #print theta_BCM
    #x=x.reshape(N_E,1)

    # if i=pre,j=post (which it isn't .. ?)
    #_W[:N_E,:N_E] += alpha*_x*_x.transpose()*(_x.transpose()-_theta_BCM)

    # if j=pre,i=post ; which it is (?)
    _W[:N_E,:N_E] += alpha*_x.transpose()*_x*(_x-_theta_BCM.reshape(N_E,1))

    # if j=pre,i=post ; which it is (?), artificial speedup
    #_W[:N_E,:N_E] += alpha*_x.transpose()*_x*(_x-_theta_BCM.reshape(N_E,1))*(1.0/theta_BCM.reshape(N_E,1))

    # only half of excitatory neurons are plastic
    #W[:N_E,:N_E/2] += alpha*x.transpose()*x*(x-theta_BCM[:N_E/2])
    #dW = alpha*x.transpose()*x*(x-theta_BCM)
    #W[:N_E,:N_E/2] += dW[:N_E,:N_E/2]

    # bounding weights to be positive
    _W[:N_E,:N_E] = _W[:N_E,:N_E]*(0.5 * (np.sign(_W[:N_E,:N_E]) + 1))

    # bounding weights below max value
    _W[_W>W_max] = W_max

    # no self-connections
    np.fill_diagonal(_W,0.0)

    # or, have weight decay
    _W[:N_E,:N_E] = _W[:N_E,:N_E]*BCM_lambda

    # set pruned synapses to zero again
    if pruned_synapses:
        _W[:N_E,:N_E][W_pruned[:N_E,:N_E]==1] = 0

    return _W

def update_weights_EE_EI(_x,_W,_theta_BCM):
    #_x=_x[:N_E]
    # only half of excitatory neurons are plastic
    #x=x[:N_E/2]

    #print theta_BCM
    #x=x.reshape(N_E,1)

    # if i=pre,j=post (which it isn't .. ?)
    #_W[:N_E,:N_E] += alpha*_x*_x.transpose()*(_x.transpose()-_theta_BCM)

    # if j=pre,i=post ; which it is (?)
    _W[:,:N_E] += alpha*_x[:N_E].transpose()*_x*(_x-_theta_BCM.reshape(N_E+N_I,1))
    # if j=pre,i=post ; which it is (?), artificial speedup
    #_W[:N_E,:N_E] += alpha*_x.transpose()*_x*(_x-_theta_BCM.reshape(N_E,1))*(1.0/theta_BCM.reshape(N_E,1))

    # only half of excitatory neurons are plastic
    #W[:N_E,:N_E/2] += alpha*x.transpose()*x*(x-theta_BCM[:N_E/2])
    #dW = alpha*x.transpose()*x*(x-theta_BCM)
    #W[:N_E,:N_E/2] += dW[:N_E,:N_E/2]

    # bounding weights to be positive
    _W[:,:N_E] = _W[:,:N_E]*(0.5 * (np.sign(_W[:,:N_E]) + 1))

    # bounding weights below max value
    _W[_W>W_max] = W_max

    # no self-connections
    np.fill_diagonal(_W,0.0)

    # or, have weight decay
    _W[:,:N_E] = _W[:,:N_E]*BCM_lambda

    # set pruned synapses to zero again
    if pruned_synapses:
        _W[:,:N_E][W_pruned[:,:N_E]==1] = 0

    return _W

def update_inh_weights(_x,_W):
    # if j=pre,i=post ; which it is (?)
    _W[:N_E,N_E:] += -1*eta*(_x[N_E:].transpose()*(_x[:N_E]-BCM_target))

    # bounding weights to be negative
    _W[:N_E:,N_E:] = _W[:N_E,N_E:]*(0.5 * (-1*np.sign(_W[:N_E,N_E:]) + 1))

    # bounding weights below max value
    _W[_W<W_inh_min] = W_inh_min

    return _W

def update_weights_pure_Hebbian(_x,_W):
    _x=_x[:N_E]
    # only half of excitatory neurons are plastic
    #x=x[:N_E/2]

    #print theta_BCM
    #x=x.reshape(N_E,1)

    # if i=pre,j=post (which it isn't .. ?)
    #_W[:N_E,:N_E] += alpha*_x*_x.transpose()*(_x.transpose()-_theta_BCM)

    # if j=pre,i=post ; which it is (?)
    _W[:N_E,:N_E] += alpha*_x.transpose()*_x

    # if j=pre,i=post ; which it is (?), artificial speedup
    #_W[:N_E,:N_E] += alpha*_x.transpose()*_x*(_x-_theta_BCM.reshape(N_E,1))*(1.0/theta_BCM.reshape(N_E,1))

    # only half of excitatory neurons are plastic
    #W[:N_E,:N_E/2] += alpha*x.transpose()*x*(x-theta_BCM[:N_E/2])
    #dW = alpha*x.transpose()*x*(x-theta_BCM)
    #W[:N_E,:N_E/2] += dW[:N_E,:N_E/2]

    # bounding weights to be positive
    _W[:N_E,:N_E] = _W[:N_E,:N_E]*(0.5 * (np.sign(_W[:N_E,:N_E]) + 1))

    # bounding weights below max value
    _W[_W>W_max] = W_max

    # no self-connections
    np.fill_diagonal(_W,0.0)

    # or, have weight decay
    _W[:N_E,:N_E] = _W[:N_E,:N_E]*BCM_lambda

    # set pruned synapses to zero again
    if pruned_synapses:
        _W[:N_E,:N_E][W_pruned[:N_E,:N_E]==1] = 0

    return _W

def update_weights_taro_fluct(_x,_W,_theta_BCM):
    _x=_x[:N_E]
    # only half of excitatory neurons are plastic
    #x=x[:N_E/2]

    #print theta_BCM
    #x=x.reshape(N_E,1)

    # if i=pre,j=post (which it isn't .. ?)
    #_W[:N_E,:N_E] += alpha*_x*_x.transpose()*(_x.transpose()-_theta_BCM)

    # if j=pre,i=post ; which it is (?)
    _W[:N_E,:N_E] += alpha*_x.transpose()*_x*(_x-_theta_BCM.reshape(N_E,1))

    # if j=pre,i=post ; which it is (?), artificial speedup
    #_W[:N_E,:N_E] += alpha*_x.transpose()*_x*(_x-_theta_BCM.reshape(N_E,1))*(1.0/theta_BCM.reshape(N_E,1))

    # only half of excitatory neurons are plastic
    #W[:N_E,:N_E/2] += alpha*x.transpose()*x*(x-theta_BCM[:N_E/2])
    #dW = alpha*x.transpose()*x*(x-theta_BCM)
    #W[:N_E,:N_E/2] += dW[:N_E,:N_E/2]

    # weight-dependent fluctuations
    _W[:N_E,:N_E] += stats.norm.rvs(scale=0.000001+_W[:N_E,:N_E]*0.00005,size=(N_E,N_E))

    # bounding weights to be positive
    _W[:N_E,:N_E] = _W[:N_E,:N_E]*(0.5 * (np.sign(_W[:N_E,:N_E]) + 1))

    # bounding weights below max value
    _W[_W>W_max] = W_max

    # no self-connections
    np.fill_diagonal(_W,0.0)

    # or, have weight decay
    _W[:N_E,:N_E] = _W[:N_E,:N_E]*BCM_lambda

    # set pruned synapses to zero again
    if pruned_synapses:
        _W[:N_E,:N_E][W_pruned[:N_E,:N_E]==1] = 0

    return _W

def update_W_pruned(_W):
    _W = _W.copy()
    _W[W_pruned==1] = 1.0
    W_pruned[_W<W_max*prune_threshold] = 1

def update_theta_BCM(x,theta_BCM):
    x_arr = x.reshape(x.size,)
    theta_BCM += theta_BCM_dt*((x_arr[:N_E]/BCM_target)*x_arr[:N_E] - theta_BCM)

    return theta_BCM

def update_theta_BCM_E_I(x,theta_BCM):
    x_arr = x.reshape(x.size,)
    theta_BCM += theta_BCM_dt*((x_arr/BCM_target)*x_arr - theta_BCM)

    return theta_BCM

def do_synaptic_scaling(W,x,y_bar,y_0,tau_scaling,settling=False):
    y_bar += (1.0/tau_y)*(-y_bar + x[:N_E])

    #mean_incoming_weight[i]=np.mean(W,axis=1)[:N_E]

    if not settling:
        for i in xrange(N_E):
            W[:,i][:N_E] += (1.0/tau_scaling)*(W[:,i][:N_E]*(y_0-y_bar[i]))

    return W,y_bar

def update_weight_toyoizumi(x,W,H,theta_BCM,y_bar,y_0,gamma,w_max,w_min,tau_y,settling=False):
    x.resize(N_E,1)
    y_bar.resize(N_E,1)

    # if i=pre,j=post (which it isn't .. ?)
    #W[:N_E,:N_E] += alpha*x*x.transpose()*(x.transpose()-theta_BCM)

    # if j=pre,i=post ; which it is (?)

    xy_theta = (x.transpose()*x - theta_BCM)
    xy_theta = xy_theta*(0.5 * (np.sign(xy_theta) + 1))

    theta_xy = (theta_BCM - x.transpose()*x)
    theta_xy = theta_xy*(0.5 * (np.sign(theta_xy) + 1))

    w_max_W = w_max - W[:N_E,:N_E]
    w_max_W = w_max_W*(0.5 * (np.sign(w_max_W) + 1))

    W_w_min = W[:N_E,:N_E] - w_min
    W_w_min = W_w_min*(0.5 * (np.sign(W_w_min) + 1))

    y_bar += (1.0/tau_y)*(-y_bar + x)

    if not settling:
        W[:N_E,:N_E] += alpha*(w_max_W*xy_theta - W_w_min*theta_xy) + gamma*W[:N_E,:N_E]*(1-y_bar/y_0)
    return W,y_bar

def update_state_toyoziumu(x,W,H,theta_BCM,y_bar,y_0,gamma,w_max,w_min,tau_y,settling=False):
    x += dt*(-1*x + np.dot(W,update_rates(x)) + H)

    # only allow positive firing rates
    x = x*(0.5 * (np.sign(x) + 1))

    x_exc = x[:N_E]

    W,y_bar = update_weight_toyoizumi(x_exc,W,H,theta_BCM,y_bar,y_0,gamma,w_max,w_min,tau_y,settling)

    return x,W,y_bar


def update_state_sliding_threshold(x,W,H,theta_BCM):
    H.resize(H.size,1)
    x.resize(x.size,1)

    x += dt*(-1*x + np.dot(W,update_rates(x)) + H)

    # only allow positive firing rates
    x = x*(0.5 * (np.sign(x) + 1))

    W = update_weights(x,W,theta_BCM)

    theta_BCM = update_theta_BCM(x,theta_BCM)

    return x.reshape(x.size,1),W,theta_BCM

def update_state_sliding_threshold_EE_EI(x,W,H,theta_BCM):
    H.resize(H.size,1)
    x.resize(x.size,1)

    x += dt*(-1*x + np.dot(W,update_rates(x)) + H)

    # only allow positive firing rates
    x = x*(0.5 * (np.sign(x) + 1))

    W = update_weights_EE_EI(x,W,theta_BCM)

    theta_BCM = update_theta_BCM_E_I(x,theta_BCM)

    return x.reshape(x.size,1),W,theta_BCM

def update_state_sliding_threshold_scaling(x,W,H,theta_BCM,y_bar,y_0,tau_scaling,settling=False):
    x += dt*(-1*x + np.dot(W,update_rates(x)) + H)

    # only allow positive firing rates
    x = x*(0.5 * (np.sign(x) + 1))

    W = update_weights(x,W)

    theta_BCM = update_theta_BCM(x,theta_BCM)

    W,y_bar = do_synaptic_scaling(W,x,y_bar,y_0,tau_scaling,settling)

    return x,W,theta_BCM,y_bar

def update_state(x,W,H):
    x += dt*(-1*x + np.dot(W,update_rates(x)) + H)

    # only allow positive firing rates
    x = x*(0.5 * (np.sign(x) + 1))

    W = update_weights(x,W)

    return x,W

def update_state_inh_plasticity(x,W,H):
    H.resize(H.size,1)
    x.resize(x.size,1)

    x += dt*(-1*x + np.dot(W,update_rates(x)) + H)

    # only allow positive firing rates
    x = x*(0.5 * (np.sign(x) + 1))

    W = update_inh_weights(x,W)

    #theta_BCM = update_theta_BCM_E_I(x,theta_BCM)

    return x.reshape(x.size,1),W,theta_BCM

def update_state_sliding_threshold_EE_EI_and_inh_plasticity(x,W,H,theta_BCM):
    H.resize(H.size,1)
    x.resize(x.size,1)

    x += dt*(-1*x + np.dot(W,update_rates(x)) + H)

    # only allow positive firing rates
    x = x*(0.5 * (np.sign(x) + 1))

    W = update_weights_EE_EI(x,W,theta_BCM)
    W = update_inh_weights(x,W)

    theta_BCM = update_theta_BCM_E_I(x,theta_BCM)

    return x.reshape(x.size,1),W,theta_BCM

def update_state_sliding_threshold_EE_and_inh_plasticity(x,W,H,theta_BCM):
    H.resize(H.size,1)
    x.resize(x.size,1)

    x += dt*(-1*x + np.dot(W,update_rates(x)) + H)

    # only allow positive firing rates
    x = x*(0.5 * (np.sign(x) + 1))

    W = update_weights(x,W,theta_BCM)
    W = update_inh_weights(x,W)

    theta_BCM = update_theta_BCM(x,theta_BCM)

    return x.reshape(x.size,1),W,theta_BCM

def update_state_static_threshold_EE_and_inh_plasticity(x,W,H,theta_BCM):
    H.resize(H.size,1)
    x.resize(x.size,1)

    x += dt*(-1*x + np.dot(W,update_rates(x)) + H)

    # only allow positive firing rates
    x = x*(0.5 * (np.sign(x) + 1))

    W = update_weights(x,W,theta_BCM)
    W = update_inh_weights(x,W)

    return x.reshape(x.size,1),W,theta_BCM

def update_state_pure_Hebbian_EE_and_inh_plasticity(x,W,H):
    H.resize(H.size,1)
    x.resize(x.size,1)

    x += dt*(-1*x + np.dot(W,update_rates(x)) + H)

    # only allow positive firing rates
    x = x*(0.5 * (np.sign(x) + 1))

    W = update_weights_pure_Hebbian(x,W)
    W = update_inh_weights(x,W)

    return x.reshape(x.size,1),W

def update_state_no_plasticity(x,W,H):
    H.resize(H.size,1)
    x.resize(x.size,1)

    x += dt*(-1*x + np.dot(W,update_rates(x)) + H)

    # only allow positive firing rates
    x = x*(0.5 * (np.sign(x) + 1))

    return x.reshape(x.size,1)


    #if ext_OU_noise:
    #    ext_OU = np.zeros((N,T))
    #    for n_idx in xrange(N):
    #        ext_OU[n_idx] = stgen.OU_generator(0.1,10,ext_OU_sigma,0,0,int(T/10),True)[0]
    #    ext_OU = np.transpose(ext_OU)

def update_state_sliding_threshold_taro_fluct(x,W,H,theta_BCM):
    H.resize(H.size,1)
    x.resize(x.size,1)

    x += dt*(-1*x + np.dot(W,update_rates(x)) + H)

    # only allow positive firing rates
    x = x*(0.5 * (np.sign(x) + 1))

    W = update_weights_taro_fluct(x,W,theta_BCM)

    theta_BCM = update_theta_BCM(x,theta_BCM)

    return x.reshape(x.size,1),W,theta_BCM

def update_state_static_threshold_EE_and_inh_plasticity_taro_fluct(x,W,H,theta_BCM):
    H.resize(H.size,1)
    x.resize(x.size,1)

    x += dt*(-1*x + np.dot(W,update_rates(x)) + H)

    # only allow positive firing rates
    x = x*(0.5 * (np.sign(x) + 1))

    W = update_weights_taro_fluct(x,W,theta_BCM)
    W = update_inh_weights(x,W)

    return x.reshape(x.size,1),W,theta_BCM

def update_state_taro_mult(_x,_P,_H,_H_taro): #theta_BCM ~ H
    _H.resize(_H.size,1)
    _x.resize(_x.size,1)

    _x += dt*(-1*x + np.dot(_H_taro*_P,update_rates(_x)) + _H) # weight = p*H (where p is denoted W)

    # only allow positive firing rates
    _x = _x*(0.5 * (np.sign(_x) + 1))

    _P += (W_max - _P[:N_E,:N_E])*max(0,_x.transpose()*_x - BCM_target) - (P[:N_E,:N_E] - 0.0)*max(BCM_target - _x.tranpose()*_x)

    x_arr = _x.reshape(_x.size,)
    _H_taro += theta_BCM_dt*_H_taro*(1.0 - x_arr[:N_E]/BCM_target)

    return _x,_P,_H_taro


from scipy import linalg

def partial_corr(x,y,z,plot=True,color='b',xlabel='population correlation',ylabel='firing rate',title='network'):
    beta_i = stats.linregress(z, x)
    beta_j = stats.linregress(z, y)

    line_x = beta_i[1] + np.multiply(beta_i[0],z)
    line_y =  beta_j[1] + np.multiply(beta_j[0],z)

    #print beta_i,beta_j

    #plt.scatter(x,line_x)
    #plt.scatter(y,line_y,color='r')

    res_j = np.subtract(y,line_y)
    res_i = np.subtract(x,line_x)

    corr = stats.spearmanr(res_i, res_j)
    print 'partial correlation', corr

    if plot:
        fig,axes=plt.subplots()
        plt.scatter(res_j,res_i,color=color)
        plt.title('Residuals ,' + title + ' ; ' + str(corr[0])+ ' ' + str(corr[1]))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    #plt.scatter(z,res_j,color='r')


# In[3]:

#%%px

def prep_net_run(T):
    from NeuroTools import stgen

    if ext_OU_noise:
        stgen = stgen.StGen()
        ext_OU = np.zeros((N_E+N_I,T))
        for n_idx in xrange(N_E+N_I):
            ext_OU[n_idx] = stgen.OU_generator(1,ext_OU_tau,ext_OU_sigma,0,0,T).signal
        ext_OU = np.transpose(ext_OU)

    return ext_OU

def input_generator(T,N_E,N_I,H_min,H_max,input_type='constant',t_change=500,noise_type=None,OU_drive_tau=10.0,OU_drive_sigma=0.0):
    _H = np.ones((T,N_E+N_I))*H_min
    _orientations = np.zeros((T,N_orientations))
    _secondary_inputs = np.zeros((T,N_secondary_groups))

    OU_drive = np.zeros(T)
    if noise_type=='OU_drive':
        from NeuroTools import stgen

        stgen = stgen.StGen()
        ext_OU = np.zeros((N_E+N_I,T))
        OU_drive = stgen.OU_generator(1,OU_drive_tau,OU_drive_sigma,0,0,T).signal

    #print 'H_min input gen = ', H_min

    #if input_type == 'orientation_and_secondary_dynamic':
    #    secondary_groups = np.array([[x]*(N_E/N_secondary_groups) for x in xrange(N_secondary_groups)]).flatten()
    #    np.random.shuffle(_secondary_groups)

    #plt.figure()

    paired_trial = True
    gap_trial = 5

    for i in xrange(T):
        if input_type == 'random_dynamic':
            # randomly chosen subset of neurons receieve high input
            if i%t_change == 0:
                _H[i] = np.ones(N_E+N_I)*H_min
            if i%t_change*2 == 0:
                _H[i][np.random.randint(0,N_E,np.random.randint(N_E/4,N_E/2))]=H_max
            elif not i%t_change == 0:
                _H[i] = _H[i-1]
            #H = H+OU_drive[i]
        elif input_type == 'oriented_dynamic':
            # randomly chosen orientation of neurons receieve high input
            if i%t_change == 0:
                _H[i] = np.ones(N_E+N_I)*H_min
                orientation_i=np.random.randint(N_orientations)
                #orientation_i=0
            _H[i][orientation_i*(N_E/N_orientations):(orientation_i*(N_E/N_orientations)+N_E/N_orientations)]=H_max
            _orientations[i][orientation_i]=1
            #plt.plot(_H[i])
            #H = H+OU_drive[i]
        elif input_type == 'oriented_dynamic_inh':
            # randomly chosen orientation of neurons receieve high input
            if i%t_change == 0:
                _H[i] = np.ones(N_E+N_I)*H_min
                orientation_i=np.random.randint(N_orientations)
                #orientation_i=0
            _H[i][orientation_i*(N_E/N_orientations):(orientation_i*(N_E/N_orientations)+N_E/N_orientations)]=H_max
            _H[i][N_E+orientation_i*(N_I/N_orientations):N_E+(orientation_i*(N_I/N_orientations)+N_I/N_orientations)]=H_max
            _orientations[i][orientation_i]=1
            #plt.plot(_H[i])
            #H = H+OU_drive[i]
        elif input_type == 'oriented_dynamic_half_groups':
            # randomly chosen orientation of neurons receieve high input
            if i%t_change == 0:
                _H[i] = np.ones(N_E+N_I)*H_min
                orientation_i=np.random.randint(N_orientations/2)
                _H[i][orientation_i*(N_E/N_orientations):(orientation_i*(N_E/N_orientations)+N_E/N_orientations)]=H_max
            if i%t_change == 0:
                _H[i][np.random.randint(N_E/2,N_E,N_E/4)]=H_max
            elif not i%t_change == 0:
                _H[i] = _H[i-1]
                #orientation_i=0
            _orientations[i][orientation_i]=1
            #plt.plot(_H[i])
            #H = H+OU_drive[i]
            #if i%100 == 0:
            #    print _H[i]
        elif input_type == 'oriented_dynamic_half_groups_pairs':
            # randomly chosen orientation of neurons receieve high input
            if i%t_change == 0:
                _H[i] = np.ones(N_E+N_I)*H_min
                orientation_i=np.random.randint(N_orientations/2)
                orientation_j=orientation_i+N_orientations/2
                #orientation_i=0
            _H[i][orientation_i*(N_E/N_orientations):(orientation_i*(N_E/N_orientations)+N_E/N_orientations)]=H_max
            _H[i][orientation_j*(N_E/N_orientations):(orientation_j*(N_E/N_orientations)+N_E/N_orientations)]=H_max
            _orientations[i][orientation_i]=1
            _orientations[i][orientation_j]=1
            #plt.plot(_H[i])
            #H = H+OU_drive[i]
        elif input_type == 'oriented_dynamic_spaced':
            # randomly chosen orientation of neurons receieve high input
            if i%t_change == 0:
                _H[i] = np.ones(N_E+N_I)*H_min
                orientation_i=np.random.randint(N_orientations)
                #print 'orientation_i = ', orientation_i
            _H[i][orientation_i::(N_E/N_orientations)]=H_max
            _orientations[i][orientation_i]=orientation_i
        elif input_type == 'oriented_dynamic_alternate_double':
            # randomly chosen orientation of neurons receieve high input
            if i%(t_change*2) == 0:
                _H[i] = np.ones(N_E+N_I)*H_min
                orientation_j=np.random.randint(N_orientations/2)
            elif i%t_change == 0:
                _H[i] = np.ones(N_E+N_I)*H_min
                orientation_i=np.random.randint(N_orientations)
                orientation_j = None
                #orientation_i=0

            if not orientation_j == None:
                _H[i][orientation_j*(N_E/(N_orientations/2)):(orientation_j*(N_E/(N_orientations/2))+N_E/(N_orientations/2))]=H_max
                _orientations[i][orientation_j]=1
            else:
                _H[i][orientation_i*(N_E/N_orientations):(orientation_i*(N_E/N_orientations)+N_E/N_orientations)]=H_max
                _orientations[i][orientation_i]=1
            #if i%100 == 0:
            #    print _H[i]
        elif input_type == 'orientation_and_random_dynamic':
            if i%t_change == 0:
                _H[i]=np.random.uniform(H_min,H_max,N_E+N_I)
                orientation_i=np.random.randint(N_orientations)
                #print 'orientation_i = ', orientation_i
            _H[i][orientation_i*(N_E/N_orientations):(orientation_i*(N_E/N_orientations)+N_E/N_orientations)]=H_max
            #H = H+OU_drive[i]
        elif input_type == 'orientation_and_secondary_dynamic':
            if i%t_change == 0:
                _H[i]=np.ones(N_E+N_I)*H_min
                orientation_i=np.random.randint(N_orientations)
                secondary_i = np.random.randint(N_secondary_groups)
                #print 'orientation_i = ', orientation_i
            _H[i][orientation_i*(N_E/N_orientations):(orientation_i*(N_E/N_orientations)+N_E/N_orientations)]+=H_max
            _H[i][secondary_groups == secondary_i] += H_secondary
            #H = H+OU_drive[i]
            #if i%100 == 0:
            #    print _H[i]
        elif input_type == 'orientation_and_variable_secondary_dynamic':
            if i%t_change == 0:
                _H[i]=np.ones(N_E+N_I)*H_min
                orientation_i=np.random.randint(N_orientations)
                secondary_i = np.random.randint(N_secondary_groups)
                #print 'orientation_i = ', orientation_i
            _H[i][orientation_i*(N_E/N_orientations):(orientation_i*(N_E/N_orientations)+N_E/N_orientations)]+=H_max_variable[orientation_i]
            _H[i][secondary_groups == secondary_i] += H_secondary_variable[secondary_i]
            #H = H+OU_drive[i]
            #if i%100 == 0:
            #    print _H[i]
        elif input_type == 'orientation_and_secondary_dynamic_paired':
            if i%t_change == 0:
                _H[i]=np.ones(N_E+N_I)*H_min
                orientation_i=np.random.randint(N_orientations)
                if secondary_paired_idx[orientation_i] == None:
                    secondary_i = np.random.randint(N_secondary_groups)
                else:
                    secondary_i = secondary_paired_idx[orientation_i]
                #print 'orientation_i = ', orientation_i
            _H[i][orientation_i*(N_E/N_orientations):(orientation_i*(N_E/N_orientations)+N_E/N_orientations)]+=H_max
            _H[i][secondary_groups == secondary_i] += H_secondary
            #H = H+OU_drive[i]
            #if i%250 == 0:
            #    print 'p,s ',orientation_i,secondary_i
            #    print _H[i]
        elif input_type == 'orientation_and_variable_secondary_dynamic_paired':
            if i%t_change == 0:
                _H[i]=np.ones(N_E+N_I)*H_min
                orientation_i=np.random.randint(N_orientations)
                if secondary_paired_idx[orientation_i] == None:
                    secondary_i = np.random.randint(N_secondary_groups)
                else:
                    secondary_i = secondary_paired_idx[orientation_i]
                #print 'orientation_i = ', orientation_i
            _H[i][orientation_i*(N_E/N_orientations):(orientation_i*(N_E/N_orientations)+N_E/N_orientations)]+=H_max_variable[orientation_i]
            _H[i][secondary_groups == secondary_i] += H_secondary_variable[secondary_i]
            #H = H+OU_drive[i]
            #if i%250 == 0:
            #    print 'p,s ',orientation_i,secondary_i
            #    print _H[i]
        elif input_type == 'orientation_and_context_groups':
            if i%t_change == 0:
                _H[i]=np.ones(N_E+N_I)*H_min
                orientation_i=np.random.randint(N_orientations)
                #active_contexts = [np.random.binomial(1,k) for k in context_probs]
                context_rates = np.zeros(N_E+N_I)
                for k in xrange(len(context_probs)):
                    if np.random.binomial(1,context_probs[k]):
                        context_rates[context_groups[k]] += H_context
            _H[i][orientation_i*(N_E/N_orientations):(orientation_i*(N_E/N_orientations)+N_E/N_orientations)]=H_max
            _H[i][N_E+orientation_i*(N_I/N_orientations):N_E+(orientation_i*(N_I/N_orientations)+N_I/N_orientations)]=H_max
            _H[i] += context_rates
        elif input_type == 'context_groups':
            if i%t_change == 0:
                _H[i]=np.ones(N_E+N_I)*H_min
                #active_contexts = [np.random.binomial(1,k) for k in context_probs]
                context_rates = np.zeros(N_E+N_I)
                for k in xrange(len(context_probs)):
                    if np.random.binomial(1,context_probs[k]):
                        context_rates[context_groups[k]] += H_context
            _H[i] += context_rates
            #if i%250 == 0:
            #    print 'p,s ',orientation_i,context_rates
            #    print _H[i]
        elif input_type == 'audiovisual_random':
            if i%t_change == 0:
                _H[i]=np.ones(N_E+N_I)*H_min
                orientation_i = np.random.randint(N_orientations)
                secondary_i = np.random.randint(N_secondary_groups)
                #print 'orientation_i = ', orientation_i
            #_H[i][visual_responsive>0] += H_visual_baseline
            #_H[i][auditory_responsive>0] += H_auditory_baseline
            _H[i][:N_E] += H_visual_baseline*visual_responsive
            _H[i][:N_E] += H_auditory_baseline*auditory_responsive
            #_H[i][orientation_i*(N_E/N_orientations):(orientation_i*(N_E/N_orientations)+N_E/N_orientations)]+=H_max_variable[orientation_i]

            #_H[i][visual_groups == orientation_i] += H_max_variable[orientation_i]
            #_H[i][secondary_groups == secondary_i] += H_secondary_variable[secondary_i]
            _H[i][:N_E][visual_groups == orientation_i] += H_max
            _H[i][:N_E][secondary_groups == secondary_i] += H_secondary
            #if i%50000 == 0:
            #    print 'p,s ',orientation_i,context_rates
            #    print _H[i]
            _orientations[i][0]=orientation_i
            _orientations[i][1]=secondary_i
        elif input_type == 'audiovisual_random_separate':
            if i%t_change == 0:
                _H[i]=np.ones(N_E+N_I)*H_min
                if np.random.uniform(0.0,1.0) < 0.5:
                    secondary_i = -99
                    orientation_i = np.random.randint(N_orientations)
                else:
                    orientation_i = -99
                    secondary_i = np.random.randint(N_secondary_groups)
                #print 'orientation_i = ', orientation_i
            #_H[i][visual_responsive>0] += H_visual_baseline
            #_H[i][auditory_responsive>0] += H_auditory_baseline
            _H[i][:N_E] += H_visual_baseline*visual_responsive
            _H[i][:N_E] += H_auditory_baseline*auditory_responsive
            #_H[i][orientation_i*(N_E/N_orientations):(orientation_i*(N_E/N_orientations)+N_E/N_orientations)]+=H_max_variable[orientation_i]

            #_H[i][visual_groups == orientation_i] += H_max_variable[orientation_i]
            #_H[i][secondary_groups == secondary_i] += H_secondary_variable[secondary_i]
            _H[i][:N_E][visual_groups == orientation_i] += H_max
            _H[i][:N_E][secondary_groups == secondary_i] += H_secondary
            #if i%50000 == 0:
            #    print 'p,s ',orientation_i,secondary_i
            #    print _H[i]
            _orientations[i][0]=orientation_i
            _orientations[i][1]=secondary_i
        elif input_type == 'audiovisual_paired':
            if i%t_change == 0:
                _H[i]=np.ones(N_E+N_I)*H_min
                orientation_i = np.random.randint(N_orientations)
                if secondary_paired_idx[orientation_i] == None:
                    secondary_i = -2 # np.random.randint(N_secondary_groups)
                else:
                    secondary_i = secondary_paired_idx[orientation_i]
                #print 'orientation_i = ', orientation_i
            #_H[i][visual_responsive>0] += H_visual_baseline
            #_H[i][auditory_responsive>0] += H_auditory_baseline
            _H[i][:N_E] += H_visual_baseline*visual_responsive
            _H[i][:N_E] += H_auditory_baseline*auditory_responsive
            #_H[i][orientation_i*(N_E/N_orientations):(orientation_i*(N_E/N_orientations)+N_E/N_orientations)]+=H_max_variable[orientation_i]

            #_H[i][visual_groups == orientation_i] += H_max_variable[orientation_i]
            #_H[i][secondary_groups == secondary_i] += H_secondary_variable[secondary_i]

            _H[i][:N_E][visual_groups == orientation_i] += H_max
            _H[i][:N_E][secondary_groups == secondary_i] += H_secondary
            _orientations[i][0]=orientation_i
            _orientations[i][1]=secondary_i

        elif input_type == 'audiovisual_paired_only_0':
            if i%t_change == 0:
                _H[i]=np.ones(N_E+N_I)*H_min
                orientation_i = secondary_paired_idx[0]
                secondary_i = 0
                #print 'orientation_i = ', orientation_i
            #_H[i][visual_responsive>0] += H_visual_baseline
            #_H[i][auditory_responsive>0] += H_auditory_baseline
            _H[i][:N_E] += H_visual_baseline*visual_responsive
            _H[i][:N_E] += H_auditory_baseline*auditory_responsive
            #_H[i][orientation_i*(N_E/N_orientations):(orientation_i*(N_E/N_orientations)+N_E/N_orientations)]+=H_max_variable[orientation_i]

            #_H[i][visual_groups == orientation_i] += H_max_variable[orientation_i]
            #_H[i][secondary_groups == secondary_i] += H_secondary_variable[secondary_i]

            _H[i][:N_E][visual_groups == orientation_i] += H_max
            _H[i][:N_E][secondary_groups == secondary_i] += H_secondary
            _orientations[i][0]=orientation_i
            _orientations[i][1]=secondary_i

        elif input_type == 'audiovisual_repeating_only_tone_0':
            if i%t_change == 0:
                _H[i]=np.ones(N_E+N_I)*H_min
                orientation_i = -99
                secondary_i = 0
                #print 'orientation_i = ', orientation_i
            #_H[i][visual_responsive>0] += H_visual_baseline
            #_H[i][auditory_responsive>0] += H_auditory_baseline
            _H[i][:N_E] += H_visual_baseline*visual_responsive
            _H[i][:N_E] += H_auditory_baseline*auditory_responsive
            #_H[i][orientation_i*(N_E/N_orientations):(orientation_i*(N_E/N_orientations)+N_E/N_orientations)]+=H_max_variable[orientation_i]

            #_H[i][visual_groups == orientation_i] += H_max_variable[orientation_i]
            #_H[i][secondary_groups == secondary_i] += H_secondary_variable[secondary_i]

            _H[i][:N_E][visual_groups == orientation_i] += H_max
            _H[i][:N_E][secondary_groups == secondary_i] += H_secondary
            _orientations[i][0]=orientation_i
            _orientations[i][1]=secondary_i

        elif input_type == 'audiovisual_interleaved_paired_0_1_unpaired_tone_1':
            if i%t_change == 0:
                paired_trial = np.random.uniform()>0.75
                if paired_trial:
                    _H[i]=np.ones(N_E+N_I)*H_min
                    orientation_i = secondary_paired_idx[0]
                    secondary_i = 0
                else:
                    if np.random.uniform()>0.5:
                        _H[i]=np.ones(N_E+N_I)*H_min
                        orientation_i = np.random.randint(N_orientations)
                        orientation_i = np.random.randint(N_orientations)
                        while orientation_i == secondary_paired_idx[0]:
                            orientation_i = np.random.randint(N_orientations)
                        secondary_i = -99
                    else:
                        _H[i]=np.ones(N_E+N_I)*H_min
                        orientation_i = -99
                        secondary_i = 1
                #print 'orientation_i = ', orientation_i
            #_H[i][visual_responsive>0] += H_visual_baseline
            #_H[i][auditory_responsive>0] += H_auditory_baseline
            _H[i][:N_E] += H_visual_baseline*visual_responsive
            _H[i][:N_E] += H_auditory_baseline*auditory_responsive
            #_H[i][orientation_i*(N_E/N_orientations):(orientation_i*(N_E/N_orientations)+N_E/N_orientations)]+=H_max_variable[orientation_i]

            #_H[i][visual_groups == orientation_i] += H_max_variable[orientation_i]
            #_H[i][secondary_groups == secondary_i] += H_secondary_variable[secondary_i]

            _H[i][:N_E][visual_groups == orientation_i] += H_max
            _H[i][:N_E][secondary_groups == secondary_i] += H_secondary
            _orientations[i][0]=orientation_i
            _orientations[i][1]=secondary_i


        elif input_type == 'audiovisual_interleaved_paired_0_1_unpaired_tone_1_unpaired_grating_1':
            if i % t_change == 0:
                paired_trial = np.random.uniform() > 0.5
                if gap_trial<5:
                    gap_trial += 1
                    _H[i] = np.ones(N_E + N_I) * H_min
                    orientation_i = -99
                    secondary_i = -99
                elif paired_trial:
                    gap_trial = 0
                    _H[i] = np.ones(N_E + N_I) * H_min
                    orientation_i = secondary_paired_idx[0]
                    secondary_i = 0
                else:
                    gap_trial = 0
                    if np.random.uniform() > 0.5:
                        _H[i] = np.ones(N_E + N_I) * H_min
                        orientation_i = 1
                        secondary_i = -99
                    else:
                        _H[i] = np.ones(N_E + N_I) * H_min
                        orientation_i = -99
                        secondary_i = 1

                        # print 'orientation_i = ', orientation_i
            # _H[i][visual_responsive>0] += H_visual_baseline
            # _H[i][auditory_responsive>0] += H_auditory_baseline
            _H[i][:N_E] += H_visual_baseline * visual_responsive
            _H[i][:N_E] += H_auditory_baseline * auditory_responsive
            # _H[i][orientation_i*(N_E/N_orientations):(orientation_i*(N_E/N_orientations)+N_E/N_orientations)]+=H_max_variable[orientation_i]

            # _H[i][visual_groups == orientation_i] += H_max_variable[orientation_i]
            # _H[i][secondary_groups == secondary_i] += H_secondary_variable[secondary_i]

            _H[i][:N_E][visual_groups == orientation_i] += H_max
            _H[i][:N_E][secondary_groups == secondary_i] += H_secondary
            _orientations[i][0] = orientation_i
            _orientations[i][1] = secondary_i

        _H[i] = _H[i]+OU_drive[i]

    return _H, _orientations


def run_net_static(x,W,T=1000,N_sample=N_E):
    pop_rate = []

    ext_OU = prep_net_run(T)

    sample_rates = np.ones((T,N_sample))

    _H = np.ones((T,N_E+N_I))*H_min

    for i in xrange(T):
        H_noisy = _H+ext_OU[i]+OU_drive[i]

        x = update_state_no_plasticity(x,W,H_noisy)

        pop_rate.append(np.mean(x))
        sample_rates[i]=x.reshape(x.size,)[:N_sample]

    return x,pop_rate,sample_rates

def run_net_static_input_type(x,W,T=1000,N_sample=N_E,input_type='random',N_orientations=8,input_OU_sigma=0):
    pop_rate = []

    ext_OU = prep_net_run(T)

    sample_rates = np.ones((T/sample_res,N_sample))
    sample_inh_rates = np.ones((T/sample_res,N_I))

    #H,orientations = input_generator(T,N_E,N_I,H_min,H_max,input_type,500,'OU_drive',10.0,input_OU_sigma)

    for j in xrange(T):
        #H_noisy = H+ext_OU[i]+OU_drive[i]

        if j%T_input_gen == 0:
            H,orientations = input_generator(T_input_gen,N_E,N_I,H_min,H_max,input_type,500,'OU_drive',10.0,input_OU_sigma)
            ext_OU = prep_net_run(T_input_gen)

            H.resize(H.shape[0],H.shape[1],1)
            ext_OU.resize(ext_OU.shape[0],ext_OU.shape[1],1)

        i = j%T_input_gen

        H_noisy = H[i]+ext_OU[i]

        #if i%1000 == 0:
        #    W_orig = W.copy()

        x = update_state_no_plasticity(x,W,H_noisy)

        if j%sample_res == 0:
            pop_rate.append(np.mean(x))
            sample_rates[j/sample_res]=x.reshape(x.size,)[:N_sample]
            sample_inh_rates[j/sample_res]=x.reshape(x.size,)[N_E:]

        #if i%1000 == 0:
        #    print x
        #    print theta_BCM
        #    plt.figure()
        #    plt.pcolor(W-W_orig)

    return x,W,pop_rate,sample_rates,sample_inh_rates,orientations

def run_net_plastic(x,W,T=1000,N_sample=N_E):
    pop_rate = []

    ext_OU = prep_net_run(T)

    sample_rates = np.ones((T,N_sample))
    sample_weights = np.ones((T,N_sample))
    mean_incoming_weight = np.ones((T,N_sample))

    for i in xrange(T):
        H_noisy = H+ext_OU[i]+OU_drive[i]

        x,W = update_state(x,W,H_noisy)

        pop_rate.append(np.mean(x))
        sample_rates[i]=x.resize(x.size,)[:N_sample]
        mean_incoming_weight[i]=np.sum(W,axis=1)[:N_E]
        for j in xrange(N_sample):
            sample_weights[i][j]=W[j,0]

    return x,W,pop_rate,sample_rates,sample_weights,mean_incoming_weight

def run_net_plastic_sliding_threshold(x,W,theta_BCM,T=1000,N_sample=N_E,input_type='random',N_orientations=8,input_OU_sigma=0):
    pop_rate = []

    #ext_OU = prep_net_run(T)

    sample_rates = np.ones((T/sample_res,N_sample))
    sample_inh_rates = np.ones((T/sample_res,N_I))
    sample_weights = np.ones((T/sample_res,N_sample))
    mean_incoming_weight = np.ones((T/sample_res,N_sample))
    sample_theta_BCM = np.ones((T/sample_res,N_sample))

    #H,orientations = input_generator(T,N_E,N_I,H_min,H_max,input_type,500,'OU_drive',10.0,input_OU_sigma)

    for j in xrange(T):
        #H_noisy = H+ext_OU[i]+OU_drive[i]

        if j%T_input_gen == 0:
            H,orientations = input_generator(T_input_gen,N_E,N_I,H_min,H_max,input_type,500,'OU_drive',10.0,input_OU_sigma)
            ext_OU = prep_net_run(T_input_gen)

            H.resize(H.shape[0],H.shape[1],1)
            ext_OU.resize(ext_OU.shape[0],ext_OU.shape[1],1)

        i = j%T_input_gen

        H_noisy = H[i]+ext_OU[i]

        #if i%1000 == 0:
        #    W_orig = W.copy()

        x,W,theta_BCM = update_state_sliding_threshold(x,W,H_noisy,theta_BCM)

        if j%sample_res == 0:
            pop_rate.append(np.mean(x))
            sample_rates[j/sample_res]=x.reshape(x.size,)[:N_sample]
            sample_inh_rates[j/sample_res]=x.reshape(x.size,)[N_E:]
            mean_incoming_weight[j/sample_res]=np.sum(W,axis=1)[:N_sample]
            for k in xrange(N_sample):
                sample_weights[j/sample_res][k]=W[k,0]
            sample_theta_BCM[j/sample_res] =theta_BCM[:N_sample]

        #if i%1000 == 0:
        #    print x
        #    print theta_BCM
        #    plt.figure()
        #    plt.pcolor(W-W_orig)

    return x,W,pop_rate,sample_rates,sample_weights,mean_incoming_weight,theta_BCM,sample_theta_BCM,sample_inh_rates

def run_net_plastic_sliding_threshold_EE_EI(x,W,theta_BCM,T=1000,N_sample=N_E,input_type='random',N_orientations=8,input_OU_sigma=0):
    pop_rate = []

    #ext_OU = prep_net_run(T)

    sample_rates = np.ones((T/sample_res,N_sample))
    sample_inh_rates = np.ones((T/sample_res,N_I))
    sample_weights = np.ones((T/sample_res,N_sample))
    mean_incoming_weight = np.ones((T/sample_res,N_sample))
    sample_theta_BCM = np.ones((T/sample_res,N_sample))

    #H,orientations = input_generator(T,N_E,N_I,H_min,H_max,input_type,500,'OU_drive',10.0,input_OU_sigma)

    for j in xrange(T):
        #H_noisy = H+ext_OU[i]+OU_drive[i]

        if j%T_input_gen == 0:
            H,orientations = input_generator(T_input_gen,N_E,N_I,H_min,H_max,input_type,500,'OU_drive',10.0,input_OU_sigma)
            ext_OU = prep_net_run(T_input_gen)

            H.resize(H.shape[0],H.shape[1],1)
            ext_OU.resize(ext_OU.shape[0],ext_OU.shape[1],1)

        i = j%T_input_gen

        H_noisy = H[i]+ext_OU[i]

        #if i%1000 == 0:
        #    W_orig = W.copy()

        x,W,theta_BCM = update_state_sliding_threshold_EE_EI(x,W,H_noisy,theta_BCM)

        if j%sample_res == 0:
            pop_rate.append(np.mean(x))
            sample_rates[j/sample_res]=x.reshape(x.size,)[:N_sample]
            sample_inh_rates[j/sample_res]=x.reshape(x.size,)[N_E:]
            mean_incoming_weight[j/sample_res]=np.sum(W,axis=1)[:N_E]
            for k in xrange(N_sample):
                sample_weights[j/sample_res][k]=W[k,0]
            sample_theta_BCM[j/sample_res] =theta_BCM[:N_sample]

        #if i%1000 == 0:
        #    print x
        #    print theta_BCM
        #    plt.figure()
        #    plt.pcolor(W-W_orig)

    return x,W,pop_rate,sample_rates,sample_weights,mean_incoming_weight,theta_BCM,sample_theta_BCM,sample_inh_rates

def run_net_plastic_sliding_threshold_scaling(x,W,theta_BCM,y_bar,y_0,tau_scaling,T=1000,T_settle=1000,N_sample=N_E,input_type='random',N_orientations=8,input_OU_sigma=0):
    print 'input type' , input_type

    pop_rate = []

    ext_OU = prep_net_run(T)

    sample_rates = np.ones((T,N_sample))
    sample_weights = np.ones((T,N_sample))
    mean_incoming_weight = np.ones((T,N_sample))
    sample_theta_BCM = np.ones((T,N_sample))

    H,orientations = input_generator(T,N_E,N_I,H_min,H_max,input_type,500,'OU_drive',10.0,input_OU_sigma)

    settling=True

    plt.figure()

    for i in xrange(T):
        if i == T_settle:
            settling=False

        if i%500 == 0:
            plt.plot(H[i])

        #H_noisy = H+ext_OU[i]+OU_drive[i]
        H_noisy = H[i]+ext_OU[i]

        x,W,theta_BCM,y_bar = update_state_sliding_threshold_scaling(x,W,H_noisy,theta_BCM,y_bar,y_0,tau_scaling,settling)

        pop_rate.append(np.mean(x))
        sample_rates[i]=x[:N_sample]
        mean_incoming_weight[i]=np.sum(W,axis=1)[:N_E]
        for j in xrange(N_sample):
            sample_weights[i][j]=W[j,0]
        sample_theta_BCM[i] =theta_BCM[:N_sample]

    return x,W,pop_rate,sample_rates,sample_weights,mean_incoming_weight,theta_BCM,sample_theta_BCM

def run_net_plastic_taro(x,W,theta_BCM,y_bar,y_0,gamma,w_max,w_min,tau_y,T=1000,T_settle=1000,N_sample=N_E,input_type='random',N_orientations=8):
    pop_rate = []

    ext_OU = prep_net_run(T)

    sample_rates = np.ones((T,N_sample))
    sample_weights = np.ones((T,N_sample))
    mean_incoming_weight = np.ones((T,N_sample))
    sample_y_bar = np.ones((T,N_sample))

    settling=True

    plt.figure()
    for i in xrange(T):
        if input_type == 'dynamic':
            # randomly chosen subset of neurons receieve high input
            if i%500 == 0:
                H = np.ones(N_E+N_I)*H_min
            if i%1000 == 0:
                H[np.random.randint(0,N_E,np.random.randint(N_E/4,N_E/2))]=H_max
            #H = H+OU_drive[i]
        elif input_type == 'oriented_dynamic':
            # randomly chosen orientation of neurons receieve high input
            if i%250 == 0:
                H = np.ones(N_E+N_I)*H_min
                orientation_i=np.random.randint(N_orientations)
                #print 'orientation_i = ', orientation_i
                H[orientation_i*(N_E/N_orientations):(orientation_i*(N_E/N_orientations)+N_E/N_orientations)]=H_max
                plt.plot(H)
        elif input_type == 'oriented_dynamic_spaced':
            # randomly chosen orientation of neurons receieve high input
            if i%250 == 0:
                H = np.ones(N_E+N_I)*H_min
                orientation_i=np.random.randint(N_orientations)
                #print 'orientation_i = ', orientation_i
                H[orientation_i::(N_E/N_orientations)]=H_max
                plt.plot(H)
        elif input_type == 'oriented_dynamic_and_random':
            # randomly chosen orientation of neurons receieve high input
            if i%250 == 0:
                H = np.ones(N_E+N_I)*H_min
                H[np.random.randint(0,N_E,np.random.randint(N_E/4,N_E/2))]=H_max

                orientation_i=np.random.randint(N_orientations)
                #print 'orientation_i = ', orientation_i
                H[orientation_i*(N_E/N_orientations):(orientation_i*(N_E/N_orientations)+N_E/N_orientations)]=H_max
                plt.plot(H)
            #H = H+OU_drive[i]
        if i == T_settle:
            settling=False

        H_noisy = H+ext_OU[i]+OU_drive[i]

        x,W,y_bar = update_state_toyoziumu(x,W,H_noisy,theta_BCM,y_bar,y_0,gamma,w_max,w_min,tau_y,settling)

        pop_rate.append(np.mean(x))
        sample_rates[i]=x[:N_sample]
        mean_incoming_weight[i]=np.sum(W,axis=1)[:N_E]
        for j in xrange(N_sample):
            sample_weights[i][j]=W[j,0]
            sample_y_bar[i][j] = y_bar[j]
    return x,W,pop_rate,sample_rates,sample_weights,mean_incoming_weight,y_bar,sample_y_bar

def run_net_plastic_inh_plasticity(x,W,theta_BCM,T=1000,N_sample=N_E,input_type='random',N_orientations=8,input_OU_sigma=0):
    pop_rate = []

    #ext_OU = prep_net_run(T)

    sample_rates = np.ones((T/sample_res,N_sample))
    sample_inh_rates = np.ones((T/sample_res,N_I))
    sample_weights = np.ones((T/sample_res,N_sample))
    mean_incoming_weight = np.ones((T/sample_res,N_sample))
    sample_theta_BCM = np.ones((T/sample_res,N_sample))

    #H,orientations = input_generator(T,N_E,N_I,H_min,H_max,input_type,500,'OU_drive',10.0,input_OU_sigma)

    for j in xrange(T):
        #H_noisy = H+ext_OU[i]+OU_drive[i]

        if j%T_input_gen == 0:
            H,orientations = input_generator(T_input_gen,N_E,N_I,H_min,H_max,input_type,500,'OU_drive',10.0,input_OU_sigma)
            ext_OU = prep_net_run(T_input_gen)

            H.resize(H.shape[0],H.shape[1],1)
            ext_OU.resize(ext_OU.shape[0],ext_OU.shape[1],1)

        i = j%T_input_gen

        H_noisy = H[i]+ext_OU[i]

        #if i%1000 == 0:
        #    W_orig = W.copy()

        x,W,theta_BCM = update_state_inh_plasticity(x,W,H_noisy)

        if j%sample_res == 0:
            pop_rate.append(np.mean(x))
            sample_rates[j/sample_res]=x.reshape(x.size,)[:N_sample]
            sample_inh_rates[j/sample_res]=x.reshape(x.size,)[N_E:]
            #mean_incoming_weight[j/sample_res]=np.sum(W,axis=1)[:N_E]
            for k in xrange(N_sample):
                sample_weights[j/sample_res][k]=W[0,N_E+k]
            sample_theta_BCM[j/sample_res] =theta_BCM[:N_sample]

        #if i%1000 == 0:
        #    print x
        #    print theta_BCM
        #    plt.figure()
        #    plt.pcolor(W-W_orig)

    return x,W,pop_rate,sample_rates,sample_weights,mean_incoming_weight,theta_BCM,sample_theta_BCM,sample_inh_rates


def run_net_plastic_BCM_and_inh_plasticity(x,W,theta_BCM,T=1000,N_sample=N_E,input_type='random',N_orientations=8,input_OU_sigma=0):
    pop_rate = []

    #ext_OU = prep_net_run(T)

    sample_rates = np.ones((T/sample_res,N_E))
    sample_inh_rates = np.ones((T/sample_res,N_I))
    sample_exc_weights = np.ones((T/sample_res,N_E))
    sample_inh_weights = np.ones((T/sample_res,N_I))
    mean_incoming_weight = np.ones((T/sample_res,N_sample))
    sample_theta_BCM = np.ones((T/sample_res,N_sample))

    #H,orientations = input_generator(T,N_E,N_I,H_min,H_max,input_type,500,'OU_drive',10.0,input_OU_sigma)

    for j in xrange(T):
        #H_noisy = H+ext_OU[i]+OU_drive[i]

        if j%T_input_gen == 0:
            H,orientations = input_generator(T_input_gen,N_E,N_I,H_min,H_max,input_type,500,'OU_drive',10.0,input_OU_sigma)
            ext_OU = prep_net_run(T_input_gen)

            H.resize(H.shape[0],H.shape[1],1)
            ext_OU.resize(ext_OU.shape[0],ext_OU.shape[1],1)

        i = j%T_input_gen

        H_noisy = H[i]+ext_OU[i]

        #if i%1000 == 0:
        #    W_orig = W.copy()

        x,W,theta_BCM = update_state_sliding_threshold_EE_EI_and_inh_plasticity(x,W,H_noisy,theta_BCM)

        if j%sample_res == 0:
            pop_rate.append(np.mean(x))
            sample_rates[j/sample_res]=x.reshape(x.size,)[:N_E]
            sample_inh_rates[j/sample_res]=x.reshape(x.size,)[N_E:]
            #mean_incoming_weight[j/sample_res]=np.sum(W,axis=1)[:N_E]
            for k in xrange(N_E):
                sample_exc_weights[j/sample_res][k]=W[0,k]
                if k < N_I:
                    sample_inh_weights[j/sample_res][k]=W[0,N_E+k]
            sample_theta_BCM[j/sample_res] =theta_BCM[:N_sample]

            if continual_pruning:
                update_W_pruned(W)

        #if i%1000 == 0:
        #    print x
        #    print theta_BCM
        #    plt.figure()
        #    plt.pcolor(W-W_orig)

    return x,W,pop_rate,sample_rates,sample_exc_weights,sample_inh_weights,mean_incoming_weight,theta_BCM,sample_theta_BCM,sample_inh_rates

def run_net_plastic_exc_BCM_and_inh_plasticity(x,W,theta_BCM,T=1000,N_sample=N_E,input_type='random',N_orientations=8,input_OU_sigma=0,checkpoint_path=None):
    #try:
    #    get_ipython().magic(u'matplotlib qt')
    #except:
    #    pass
    #plt.ion()
    #fig,axes = plt.subplots(1,3)

    pop_rate = []

    #ext_OU = prep_net_run(T)

    sample_rates = np.ones((T/sample_res,N_sample))
    sample_inh_rates = np.zeros((T/sample_res,N_I))
    sample_exc_weights = np.ones((T/sample_res,N_sample))
    sample_inh_weights = np.zeros((T/sample_res,N_sample))
    mean_incoming_weight = np.ones((T/sample_res,N_sample))
    sample_theta_BCM = np.ones((T/sample_res,N_sample))

    #H,orientations = input_generator(T,N_E,N_I,H_min,H_max,input_type,500,'OU_drive',10.0,input_OU_sigma)

    for j in xrange(T):
        #H_noisy = H+ext_OU[i]+OU_drive[i]

        if j%T_input_gen == 0:
            H,orientations = input_generator(T_input_gen,N_E,N_I,H_min,H_max,input_type,500,'OU_drive',10.0,input_OU_sigma)
            ext_OU = prep_net_run(T_input_gen)

            H.resize(H.shape[0],H.shape[1],1)
            ext_OU.resize(ext_OU.shape[0],ext_OU.shape[1],1)

        i = j%T_input_gen

        H_noisy = H[i]+ext_OU[i]

        #if i%1000 == 0:
        #    W_orig = W.copy()

        x,W,theta_BCM = update_state_sliding_threshold_EE_and_inh_plasticity(x,W,H_noisy,theta_BCM)

        if j%sample_res == 0:
            pop_rate.append(np.mean(x))
            sample_rates[j/sample_res]=x.reshape(x.size,)[:N_E]
            sample_inh_rates[j/sample_res]=x.reshape(x.size,)[N_E:]
            #mean_incoming_weight[j/sample_res]=np.sum(W,axis=1)[:N_E]
            for k in xrange(N_sample):
                if k < N_I:
                    sample_inh_weights[j/sample_res][k]=W[0,N_E+k]
                sample_exc_weights[j/sample_res][k]=W[0,k]
            sample_theta_BCM[j/sample_res] =theta_BCM[:N_sample]

            #plt.cla()
            #axes[0].plot(sample_exc_weights[:j/sample_res])
            #axes[1].plot(sample_inh_weights[:j/sample_res])
            #axes[2].pcolor(W)
            #plt.draw()
            #plt.pause(0.001)

            if continual_pruning:
                update_W_pruned(W)

        if j%checkpoint_res == 0 and checkpoint_path is not None:
            checkpoint_results = {'sample_exc_weights': sample_exc_weights,'sample_rates':sample_rates,'W':W}
            py_scripts_yann.save_pickle('pkl_results/'+checkpoint_path+'.ckpt',checkpoint_results)


        #if i%1000 == 0:
        #    print x
        #    print theta_BCM
        #    plt.figure()
        #    plt.pcolor(W-W_orig)

    return x,W,pop_rate,sample_rates,sample_exc_weights,sample_inh_weights,mean_incoming_weight,theta_BCM,sample_theta_BCM,sample_inh_rates

def run_net_static_threshold_exc_and_inh_plasticity(x,W,theta_BCM,T=1000,N_sample=N_E,input_type='random',N_orientations=8,input_OU_sigma=0):
    #try:
    #    get_ipython().magic(u'matplotlib qt')
    #except:
    #    pass
    #plt.ion()

    #fig,axes = plt.subplots(1,3)

    pop_rate = []

    #ext_OU = prep_net_run(T)

    sample_rates = np.ones((T/sample_res,N_sample))
    sample_inh_rates = np.ones((T/sample_res,N_I))
    sample_exc_weights = np.ones((T/sample_res,N_sample))
    sample_inh_weights = np.ones((T/sample_res,N_I))
    mean_incoming_weight = np.ones((T/sample_res,N_sample))
    sample_theta_BCM = np.ones((T/sample_res,N_sample))

    #H,orientations = input_generator(T,N_E,N_I,H_min,H_max,input_type,500,'OU_drive',10.0,input_OU_sigma)

    for j in xrange(T):
        #H_noisy = H+ext_OU[i]+OU_drive[i]

        if j%T_input_gen == 0:
            H,orientations = input_generator(T_input_gen,N_E,N_I,H_min,H_max,input_type,500,'OU_drive',10.0,input_OU_sigma)
            ext_OU = prep_net_run(T_input_gen)

            H.resize(H.shape[0],H.shape[1],1)
            ext_OU.resize(ext_OU.shape[0],ext_OU.shape[1],1)

        i = j%T_input_gen

        H_noisy = H[i]+ext_OU[i]

        #resetting rates every stim_change
        if j%500 == 0:
            x = np.zeros(np.shape(x))

        #if i%1000 == 0:
        #    W_orig = W.copy()

        x,W,theta_BCM = update_state_static_threshold_EE_and_inh_plasticity(x,W,H_noisy,theta_BCM)

        if j%sample_res == 0:
            pop_rate.append(np.mean(x))
            sample_rates[j/sample_res]=x.reshape(x.size,)[:N_E]
            sample_inh_rates[j/sample_res]=x.reshape(x.size,)[N_E:]
            #mean_incoming_weight[j/sample_res]=np.sum(W,axis=1)[:N_E]
            for k in xrange(N_sample):
                if k < N_I:
                    sample_inh_weights[j/sample_res][k]=W[0,N_E+k]
                sample_exc_weights[j/sample_res][k]=W[0,k]
            sample_theta_BCM[j/sample_res] =theta_BCM[:N_sample]

            #plt.cla()
            #axes[0].plot(sample_exc_weights[:j/sample_res])
            #axes[1].plot(sample_inh_weights[:j/sample_res])
            #axes[2].pcolor(W)
            #plt.draw()
            #plt.pause(0.001)

        #if i%1000 == 0:
        #    print x
        #    print theta_BCM
        #    plt.figure()
        #    plt.pcolor(W-W_orig)

    return x,W,pop_rate,sample_rates,sample_exc_weights,sample_inh_weights,mean_incoming_weight,theta_BCM,sample_theta_BCM,sample_inh_rates

def run_net_pure_Hebbian_EE_and_inh_plasticity(x,W,theta_BCM,T=1000,N_sample=N_E,input_type='random',N_orientations=8,input_OU_sigma=0):
    try:
        get_ipython().magic(u'matplotlib qt')
    except:
        pass
    plt.ion()
    fig,axes = plt.subplots(1,3)

    pop_rate = []

    #ext_OU = prep_net_run(T)

    sample_rates = np.ones((T/sample_res,N_sample))
    sample_inh_rates = np.ones((T/sample_res,N_I))
    sample_exc_weights = np.ones((T/sample_res,N_sample))
    sample_inh_weights = np.ones((T/sample_res,N_I))
    mean_incoming_weight = np.ones((T/sample_res,N_sample))
    sample_theta_BCM = np.ones((T/sample_res,N_sample))

    #H,orientations = input_generator(T,N_E,N_I,H_min,H_max,input_type,500,'OU_drive',10.0,input_OU_sigma)

    for j in xrange(T):
        #H_noisy = H+ext_OU[i]+OU_drive[i]

        if j%T_input_gen == 0:
            H,orientations = input_generator(T_input_gen,N_E,N_I,H_min,H_max,input_type,500,'OU_drive',10.0,input_OU_sigma)
            ext_OU = prep_net_run(T_input_gen)

            H.resize(H.shape[0],H.shape[1],1)
            ext_OU.resize(ext_OU.shape[0],ext_OU.shape[1],1)

        i = j%T_input_gen

        H_noisy = H[i]+ext_OU[i]

        #if i%1000 == 0:
        #    W_orig = W.copy()

        x,W= update_state_pure_Hebbian_EE_and_inh_plasticity(x,W,H_noisy)




        if j%sample_res == 0:
            pop_rate.append(np.mean(x))
            sample_rates[j/sample_res]=x.reshape(x.size,)[:N_E]
            sample_inh_rates[j/sample_res]=x.reshape(x.size,)[N_E:]
            #mean_incoming_weight[j/sample_res]=np.sum(W,axis=1)[:N_E]
            for k in xrange(N_sample):
                if k < N_I:
                    sample_inh_weights[j/sample_res][k]=W[0,N_E+k]
                sample_exc_weights[j/sample_res][k]=W[0,k]

            plt.cla()
            axes[0].plot(sample_exc_weights[:j/sample_res])
            axes[1].plot(sample_inh_weights[:j/sample_res])
            axes[2].pcolor(W)
            plt.draw()
            plt.pause(0.00001)

        #if i%1000 == 0:
        #    print x
        #    print theta_BCM
        #    plt.figure()
        #    plt.pcolor(W-W_orig)

    return x,W,pop_rate,sample_rates,sample_exc_weights,sample_inh_weights,mean_incoming_weight,theta_BCM,sample_theta_BCM,sample_inh_rates

def run_net_plastic_sliding_threshold_taro_fluct(x,W,theta_BCM,T=1000,N_sample=N_E,input_type='random',N_orientations=8,input_OU_sigma=0):
    try:
        get_ipython().magic(u'matplotlib qt')
    except:
        pass
    plt.ion()
    fig,axes = plt.subplots(1,3)

    pop_rate = []

    #ext_OU = prep_net_run(T)

    sample_rates = np.ones((T/sample_res,N_sample))
    sample_inh_rates = np.ones((T/sample_res,N_I))
    sample_weights = np.ones((T/sample_res,N_sample))
    mean_incoming_weight = np.ones((T/sample_res,N_sample))
    sample_theta_BCM = np.ones((T/sample_res,N_sample))

    #H,orientations = input_generator(T,N_E,N_I,H_min,H_max,input_type,500,'OU_drive',10.0,input_OU_sigma)

    for j in xrange(T):
        #H_noisy = H+ext_OU[i]+OU_drive[i]

        if j%T_input_gen == 0:
            H,orientations = input_generator(T_input_gen,N_E,N_I,H_min,H_max,input_type,500,'OU_drive',10.0,input_OU_sigma)
            ext_OU = prep_net_run(T_input_gen)

            H.resize(H.shape[0],H.shape[1],1)
            ext_OU.resize(ext_OU.shape[0],ext_OU.shape[1],1)

        i = j%T_input_gen

        H_noisy = H[i]+ext_OU[i]

        #if i%1000 == 0:
        #    W_orig = W.copy()

        x,W,theta_BCM = update_state_sliding_threshold(x,W,H_noisy,theta_BCM)

        if j%sample_res == 0:
            pop_rate.append(np.mean(x))
            sample_rates[j/sample_res]=x.reshape(x.size,)[:N_sample]
            sample_inh_rates[j/sample_res]=x.reshape(x.size,)[N_E:]
            mean_incoming_weight[j/sample_res]=np.sum(W,axis=1)[:N_sample]
            for k in xrange(N_sample):
                sample_weights[j/sample_res][k]=W[k,0]
            sample_theta_BCM[j/sample_res] =theta_BCM[:N_sample]

            plt.cla()
            axes[0].plot(sample_weights[:j/sample_res])
            axes[1].hist(W[:N_E,:N_E].flatten(),25)
            axes[2].pcolor(W[:N_E,:N_E])
            plt.draw()
            plt.pause(0.001)

        #if i%1000 == 0:
        #    print x
        #    print theta_BCM
        #    plt.figure()
        #    plt.pcolor(W-W_orig)

    return x,W,pop_rate,sample_rates,sample_weights,mean_incoming_weight,theta_BCM,sample_theta_BCM,sample_inh_rates

def run_net_static_threshold_exc_and_inh_plasticity_taro_fluct(x,W,theta_BCM,T=1000,N_sample=N_E,input_type='random',N_orientations=8,input_OU_sigma=0):
    try:
        get_ipython().magic(u'matplotlib qt')
    except:
        pass
    plt.ion()

    fig,axes = plt.subplots(1,3)

    pop_rate = []

    #ext_OU = prep_net_run(T)

    sample_rates = np.ones((T/sample_res,N_sample))
    sample_inh_rates = np.ones((T/sample_res,N_I))
    sample_exc_weights = np.ones((T/sample_res,N_sample))
    sample_inh_weights = np.ones((T/sample_res,N_I))
    mean_incoming_weight = np.ones((T/sample_res,N_sample))
    sample_theta_BCM = np.ones((T/sample_res,N_sample))

    #H,orientations = input_generator(T,N_E,N_I,H_min,H_max,input_type,500,'OU_drive',10.0,input_OU_sigma)

    for j in xrange(T):
        #H_noisy = H+ext_OU[i]+OU_drive[i]

        if j%T_input_gen == 0:
            H,orientations = input_generator(T_input_gen,N_E,N_I,H_min,H_max,input_type,500,'OU_drive',10.0,input_OU_sigma)
            ext_OU = prep_net_run(T_input_gen)

            H.resize(H.shape[0],H.shape[1],1)
            ext_OU.resize(ext_OU.shape[0],ext_OU.shape[1],1)

        i = j%T_input_gen

        H_noisy = H[i]+ext_OU[i]

        #if i%1000 == 0:
        #    W_orig = W.copy()

        x,W,theta_BCM = update_state_static_threshold_EE_and_inh_plasticity_taro_fluct(x,W,H_noisy,theta_BCM)

        if j%sample_res == 0:
            pop_rate.append(np.mean(x))
            sample_rates[j/sample_res]=x.reshape(x.size,)[:N_E]
            sample_inh_rates[j/sample_res]=x.reshape(x.size,)[N_E:]
            #mean_incoming_weight[j/sample_res]=np.sum(W,axis=1)[:N_E]
            for k in xrange(N_sample):
                if k < N_I:
                    sample_inh_weights[j/sample_res][k]=W[0,N_E+k]
                sample_exc_weights[j/sample_res][k]=W[0,k]
            sample_theta_BCM[j/sample_res] =theta_BCM[:N_sample]

            plt.cla()
            axes[0].plot(sample_exc_weights[:j/sample_res])
            axes[1].plot(sample_inh_weights[:j/sample_res])
            axes[2].pcolor(W)

            #if j == 0:
            #    line_1, = axes[0].plot(sample_exc_weights[:j/sample_res])
            #    line_2, = axes[1].plot(sample_inh_weights[:j/sample_res])
            #   pc = axes[2].pcolor(W)
            #else:
            #    line_1.set_ydata(sample_exc_weights[:j/sample_res])
            #    line_2.set_ydata(sample_inh_weights[:j/sample_res])
            #    pc.set_array(W)
            plt.draw()
            plt.pause(0.001)

        #if i%1000 == 0:
        #    print x
        #    print theta_BCM
        #    plt.figure()
        #    plt.pcolor(W-W_orig)

    return x,W,pop_rate,sample_rates,sample_exc_weights,sample_inh_weights,mean_incoming_weight,theta_BCM,sample_theta_BCM,sample_inh_rates


def measure_population_coupling(x,W,T_measure_corr=1000,settletime=0,return_rates=False,input_type='random_dynamic',input_OU_sigma=50.0):
    from NeuroTools import stgen

    if ext_OU_noise:
        stgen = stgen.StGen()
        ext_OU = np.zeros((N_E+N_I,T_measure_corr))
        for n_idx in xrange(N_E+N_I):
            ext_OU[n_idx] = stgen.OU_generator(1,ext_OU_tau,ext_OU_sigma,0,0,T_measure_corr).signal
        ext_OU = np.transpose(ext_OU)

    pop_rate = []

    N_sample = N_E
    sample_rates = np.ones((T_measure_corr,N_sample))
    sample_weights = np.ones((T_measure_corr,N_sample))
    mean_incoming_weight = np.ones((T_measure_corr,N_sample))

    #H = np.random.normal(1,1,T_measure_corr)

    #H = np.ones(N_E+N_I)*1.0

    H,orientations = input_generator(T_measure_corr,N_E,N_I,H_min,H_max,input_type,500,'OU_drive',10.0,input_OU_sigma)

    for i in xrange(T_measure_corr):
        ## randomly chosen subset of neurons receieve high input
        #if i%500 == 0:
        #    #H = np.zeros(N_E+N_I)
        #    H = np.ones(N_E+N_I)*H_min
        #    #if i%1000 == 0:
        #    if i%1000 == 0:
        #    #H[np.random.randint(0,N_E,N_E/10)]=10.0
        #        #H[np.random.randint(0,N_E,np.random.randint(N_E/20,N_E/5))]=100.0
        #        H[np.random.randint(0,N_E,np.random.randint(N_E/4,N_E/2))]=H_max

        H_noisy = H[i]+ext_OU[i]#+OU_drive[i]
        #H_noisy = H+ext_OU[i]

        x = update_state_no_plasticity(x,W,H_noisy)

        pop_rate.append(np.mean(x))
        sample_rates[i]=x.reshape(x.size,)[:N_sample]
        mean_incoming_weight[i]=np.sum(W,axis=1)[:N_E]
        for j in xrange(N_sample):
            sample_weights[i][j]=W[j,0]


    corr_coeffs = []

    print len(pop_rate)
    print sample_rates.shape

    #plt.figure()
    #plt.plot(pop_rate,lw=5)
    #plt.plot(sample_rates)


    for i in xrange(N_E):
        corr_coeffs.append(np.corrcoef(pop_rate[settletime:],sample_rates.transpose()[i][settletime:])[0,1])

    if not return_rates:
        return corr_coeffs
    else:
        return (corr_coeffs, pop_rate, sample_rates, x)

def measure_stPr_analytic(x,W,T_measure_corr=1000,settletime=0,input_type='random_dynamic',input_OU_sigma=50.0):
    corr_coeffs, pop_rate, sample_rates, x_measured = measure_population_coupling(x,W,T_measure_corr,settletime,True,input_type,input_OU_sigma)

    sample_rates = sample_rates.transpose()

    stPr_analytic = np.zeros(N_sample)

    for i in xrange(N_sample):
        stPr_analytic[i] = np.sum(pop_rate*sample_rates[i])/np.sum(sample_rates[i])

    return (stPr_analytic,corr_coeffs,pop_rate,sample_rates, x_measured)


def generate_spiketrains(sample_rates):
    from NeuroTools import stgen

    stgen_Poiss = stgen.StGen()

    sample_spike_trains = []
    for i in xrange(N_sample):
        sample_spike_trains.append(stgen_Poiss.inh_poisson_generator(sample_rates[i],np.arange(0,len(sample_rates[i])*1000,1000.0),len(sample_rates[i])*1000,True))
    return sample_spike_trains

def measure_stPr_numerical(x,W,T_measure_corr=1000,settletime=0,input_type='random_dynamic',input_OU_sigma=50.0):
    corr_coeffs, pop_rate, sample_rates, x_measured = measure_population_coupling(x,W,T_measure_corr,settletime,True,input_type,input_OU_sigma)

    sample_rates = sample_rates.transpose()

    stPr_numerical = np.zeros(N_sample)

    spike_trains = generate_spiketrains(sample_rates)

    for i in xrange(N_sample):
        stPr_numerical[i] = np.mean([pop_rate[int(x/1000)] for x in spike_trains[i]])

    return (stPr_numerical,corr_coeffs,pop_rate,sample_rates, x_measured)


