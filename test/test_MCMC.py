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
import time

# for graphics
#from pymcmcstat import mcmcplot as mcpl
import matplotlib.pyplot as plt

# clear command window
#print(chr(27) + "[2J")

# load random number sequences
#mhrndseq = np.loadtxt('mhrndseq0.txt')
#mhrndseq2 = np.loadtxt('mhrndseq1.txt')
#drrndseq = np.loadtxt('drrndseq0.txt')
#drrndseq2 = np.loadtxt('drrndseq1.txt')

#rndseq = [mhrndseq, mhrndseq2, drrndseq, drrndseq2]
#rndnum_u_n = np.loadtxt('rndnum_u_n.txt');

# define test model function
def test_modelfun(xdata, theta):
    m = theta[0]
    b = theta[1]
    
    y = m*xdata + b
    return y

def test_ssfun(theta, data):
    nbatch = len(data)
    ss = np.zeros(nbatch)
    
    for ibatch in range(nbatch):
        for iset in range(len(data[ibatch].n)):
            xdata = data[ibatch].xdata[iset]
            ydata = data[ibatch].ydata[iset]
            weight = data[ibatch].weight[iset]
        
            # eval model
            ymodel = test_modelfun(xdata, theta)
        
            # calc sos
            ss[ibatch] += weight*sum((ymodel - ydata)**2)
    return ss

# Initialize MCMC object
mcstat = MCMC()

# Add data
x = np.linspace(2, 3, num=10)
y = 2*x - 3
mcstat.data.add_data_set(x, y)

#x2 = np.array([[1,2],[2,1]])
#y2 = 2*x2
#mcstat.data.add_data_set(x2, y2)

# initialize parameter array
mcstat.parameters.add_model_parameter(name = 'm', theta0 = 1.2, minimum = -10, maximum = 10, prior_sigma = -1)
mcstat.parameters.add_model_parameter(name = 'b', theta0 = 40.2, minimum = -10, maximum = 100)

# update simulation options
mcstat.options.update_simulation_options(nsimu = int(5.0e3), updatesigma = 1, method = 'dram',
                     adaptint = 100, verbosity = 0, waitbar = 1)

# update model settings
mcstat.model.update_model_settings(ssfun = test_ssfun)

# Run mcmcrun
mcstat.run_simulation()
