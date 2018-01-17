#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:06:51 2018

@author: prmiles
"""

# import required packages
import numpy as np
import sys

class ModelSettings:
    def __init__(self):
        # Initialize all variables to default values
        self.ssfun = None
        self.priorfun = None
        self.priortype = 1
        self.priorupdatefun = None
        self.priorpars = None
        self.modelfun = None
        
        # check value of sigma2 - initial error variance
        self.sigma2 = None
        
        # check value of N - total number of observations
        self.N = None
        
        # check value of N0 - prior accuracy for S20
        self.N0 = None       
        
        # check nbatch - number of data sets
        self.nbatch = None
            
        # S20 - prior for sigma2
        self.S20 = np.nan
        
    def update_model_settings(self, ssfun = None, priorfun = None, priortype = 1, 
                 priorupdatefun = None, priorpars = None, modelfun = None, 
                 sigma2 = None, N = None, 
                 S20 = np.nan, N0 = None, nbatch = None):
    
        self.ssfun = ssfun
        self.priorfun = priorfun
        self.priortype = priortype
        self.priorupdatefun = priorupdatefun
        self.priorpars = priorpars
        self.modelfun = modelfun
        
        # check value of sigma2 - initial error variance
        self.sigma2 = self.array_type(sigma2)
        
        # check value of N - total number of observations
        self.N = self.array_type(N)
        
        # check value of N0 - prior accuracy for S20
        self.N0 = self.array_type(N0)       
        
        # check nbatch - number of data sets
        self.nbatch = self.array_type(nbatch)
            
        # S20 - prior for sigma2
        self.S20 = self.array_type(S20)
    
    def array_type(self, x):
        # All settings in this class should be converted to numpy ndarray
        if x is None:
            x = x
        else:
            if isinstance(x, int): # scalar -> ndarray[scalar]
                x = np.array([np.array(x)])
            elif isinstance(x, float): # scalar -> ndarray[scalar]
                x = np.array([np.array(x)])
            elif isinstance(x, list): # [...] -> ndarray[...]
                x = np.array(x)
            elif isinstance(x, np.ndarray): 
                x = x
            else:
                sys.exit('Unknown data type - Please use int, ndarray, or list')
        
        return x