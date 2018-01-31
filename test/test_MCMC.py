#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 12:12:31 2017

@author: prmiles
"""

# import required packages
import sys
#path_to_prm_pymcmcstat = '/Users/prmiles/Google Drive/prm_python_modules/pymcmcstat/'
#sys.path.insert(0, path_to_prm_pymcmcstat)

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
    
    y = m*xdata + b
    return y

def test_ssfun(theta, data):
    nbatch = len(data.xdata)
    ss = np.zeros(nbatch)
    
    for ibatch in range(nbatch):
            xdata = data.xdata[ibatch]
            ydata = data.ydata[ibatch]
            weight = data.weight[ibatch]
        
            # eval model
            ymodel = test_modelfun(xdata, theta)
        
            # calc sos
            ss[ibatch] += weight*sum((ymodel - ydata)**2)
    return ss

# Initialize MCMC object
mcstat = MCMC()

# Add data
x = np.linspace(2, 3, num=50)
m = 2 # slope
b = -3 # offset
noise = 0.1*np.random.randn(len(x))
y = m*x + b + noise
mcstat.data.add_data_set(x, y)

#x2 = np.array([[1,2],[2,1]])
#y2 = 2*x2
#mcstat.data.add_data_set(x2, y2)

# initialize parameter array
mcstat.parameters.add_model_parameter(name = 'm', theta0 = 1., minimum = -10, maximum = 10)
mcstat.parameters.add_model_parameter(name = 'b', theta0 = -5., minimum = -10, maximum = 100)

# update simulation options
mcstat.simulation_options.define_simulation_options(nsimu = int(5.0e3), updatesigma = 1, method = 'dram',
                     adaptint = 100, verbosity = 1, waitbar = 1)

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
mcpl.plot_density_panel(chain[burnin:,:], names)
mcpl.plot_chain_panel(chain[burnin:,:], names)
mcpl.plot_pairwise_correlation_panel(chain[burnin:,:], names)

# plot data & model
plt.figure()
plt.plot(x,y,'.k')
plt.plot(x, m*x + b, '-r')
plt.plot(x, test_modelfun(x, np.mean(results['chain'],0)), '--k')  