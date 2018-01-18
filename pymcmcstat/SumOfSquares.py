#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 16:21:48 2018

Description: Sum-of-squares (sos) class intended for used in MCMC simulator.  Each instance
will contain the sos function.  If the user did not specify a sos-function,
then the user supplied model function will be used in the default mcmc sos-function.

@author: prmiles
"""

# Import required packages
import numpy as np
import sys

class SumOfSquares:
    def __init__(self, model, data, parameters):
                
        # check if sos function and model function are defined
        if model.sos_function is None: #isempty(ssfun)
            if model.model_function is None: #isempty(modelfun)
                sys.exit('No ssfun or modelfun specified!')
            sos_style = 4
        else:
            sos_style = 1
        
        self.sos_function = model.sos_function
        self.sos_style = sos_style
        self.model_function = model.model_function
        self.parind = parameters.parind
        self.local = parameters.local
        self.data = data
        self.nbatch = model.nbatch
        
    def evaluate_sos_function(self, theta):
        # evaluate sum-of-squares function
        if self.sos_style == 1:
            ss = self.sos_function(theta, self.data)
        elif self.sos_style == 4:
            ss = self.mcmc_sos_function(theta, self.data, self.local, self.model_function)
        else:
            ss = self.sos_function(theta, self.data, self.local)
        
        # make sure sos is a numpy array
        if not isinstance(ss, np.ndarray):
            ss = np.array([ss])
            
        return ss
                     
    def mcmc_sos_function(self, theta):
        # initialize
        ss = np.zeros(self.nbatch)

        for ibatch in range(self.nbatch):
                xdata = self.data.xdata[ibatch]
                ydata = self.data.ydata[ibatch]
                weight = self.data.weight[ibatch]
            
                # evaluate model
                ymodel = self.model_function(xdata, theta)
    
                # calculate sum-of-squares error    
                ss[ibatch] += sum(weight*(ydata-ymodel)**2)
        
        return ss