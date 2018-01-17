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

class SumOfSquares:
    def __init__(self, sos_function = None, sos_style = None, model_function = None, 
                 parind = None, local = None, data = None, nbatch = None):
        self.sos_function = sos_function
        self.sos_style = sos_style
        self.model_function = model_function
        self.parind = parind
        self.local = local
        self.data = data
        self.nbatch = nbatch
        
    def evaluate_sos_function(self, theta):
        # evaluate sum-of-squares function
        if self.sos_style == 1:
            ss = self.ssfun(theta, self.data)
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
