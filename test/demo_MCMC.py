#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 12:12:31 2017

@author: prmiles
"""

# import required packages
import sys
path_to_prm_pymcmcstat = '/Users/prmiles/Google Drive/prm_python_modules/pymcmcstat/'
sys.path.insert(0, path_to_prm_pymcmcstat)

#import math
import numpy as np
from pymcmcstat.MCMC import MCMC


#import time

# for graphics
#from pymcmcstat import mcmcplot as mcpl

# for graphics
#from pymcmcstat import mcmcplot as mcpl
import matplotlib.pyplot as plt

# define test model function
def test_modelfun(xdata, theta):
    m = theta[0]
    b = theta[1]
    
    nrow, ncol = xdata.shape
    y = np.zeros([nrow,1])
    y[:,0] = m*xdata.reshape(nrow,) + b
#    y[:,1] = m*(xdata.reshape(nrow,))**2 + b
    return y

def test_ssfun(theta, data):
    
    xdata = data.xdata[0]
    ydata = data.ydata[0]
    
    # eval model
    ymodel = test_modelfun(xdata, theta)
    
    # calc sos
    ss1 = sum((ymodel[:,0] - ydata[:,0])**2)
#    ss2 = sum((ymodel[:,1] - ydata[:,0])**2)
    return ss1#np.array([ss1, ss2])

# Initialize MCMC object
mcstat = MCMC()

# Add data
nds = 100
x = np.linspace(2, 3, num=nds)
x = x.reshape(nds,1)
m = 2 # slope
b = -3 # offset
noise = 0.1*np.random.standard_normal(x.shape)
y1 = m*x + b + noise
y2 = m*(x**2) + b + noise
ymat = np.zeros([nds,2])
ymat[:,0] = y1.reshape(nds,)
ymat[:,1] = y2.reshape(nds,)
mcstat.data.add_data_set(x, y1)
#mcstat.data.add_data_set(x, y)

#x2 = np.array([[1,2],[2,1]])
#y2 = 2*x2
#mcstat.data.add_data_set(x2, y2)

# initialize parameter array
mcstat.parameters.add_model_parameter(name = 'm', theta0 = 1., minimum = -10, maximum = 10)
mcstat.parameters.add_model_parameter(name = 'b', theta0 = -5., minimum = -10, maximum = 100)

# update simulation options
mcstat.simulation_options.define_simulation_options(nsimu = int(5.0e3), updatesigma = 1, method = 'dram',
                     adaptint = 100, verbosity = 1, waitbar = 1, save_to_bin = True, savesize = 1000)

# update model settings
mcstat.model_settings.define_model_settings(sos_function = test_ssfun)

# Run mcmcrun
mcstat.run_simulation()

# Extract results
results = mcstat.simulation_results.results

# extend simulation
#mcstat.run_simulation(use_previous_results = True)
#mcstat.run_simulation(use_previous_results = True)

chain = results['chain']
s2chain = results['s2chain']
sschain = results['sschain']

names = results['names']

# define burnin
burnin = 2000
# display chain statistics
mcstat.chainstats(chain[burnin:,:], results)
# generate mcmc plots
mcpl = mcstat.mcmcplot # initialize plotting methods
#mcpl.plot_density_panel(chain[burnin:,:], names)
#mcpl.plot_chain_panel(chain[burnin:,:], names)
#mcpl.plot_pairwise_correlation_panel(chain[burnin:,:], names)

# plot data & model
plt.figure()
plt.plot(x,y1,'.k')
plt.plot(x, m*x + b, '-r')
model = test_modelfun(x, np.mean(results['chain'],0))
plt.plot(x, model[:,0], '--k') 

#plt.figure()
#plt.plot(x,y2,'.k')
#plt.plot(x, m*x**2 + b, '-r')
#plt.plot(x, model[:,1], '--k')  

# generate prediction intervals
# define prediction model function
def pred_modelfun(preddata, theta):
    return test_modelfun(preddata.xdata[0], theta)
    
mcstat.PI.setup_prediction_interval_calculation(results = results, data = mcstat.data, 
                                                modelfunction = pred_modelfun)

mcstat.PI.generate_prediction_intervals()

# plot prediction intervals
mcstat.PI.plot_prediction_intervals(adddata = True)