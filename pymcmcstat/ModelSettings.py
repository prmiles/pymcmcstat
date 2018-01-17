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
    
    def check_dependent_model_settings(self, data, options):
        # check dependent parameters
        if self.nbatch is None:
            self.nbatch = data.get_number_of_batches()
            
        if self.N is not None:
            N = data.get_number_of_observations()
            if self.N == N:
                self.N = N
            else:
                print('User defined N = {}.  Estimate based on data structure is N = {}.  Possible error?'.format(self.N, N))
        else:
            self.N = data.get_number_of_observations()
            
        # This is for backward compatibility
        # if sigma2 given then default N0=1, else default N0=0
        if self.N0 is None:
            if self.sigma2 is None:
                self.sigma2 = np.ones([1])
                self.N0 = np.zeros([1])
            else:
                self.N0 = np.ones([1])
            
        # set default value for sigma2    
        # default for sigma2 is S20 or 1
        if self.sigma2 is None:
            if not(np.isnan(self.S20)).any:
                self.sigma2 = self.S20
            else:
                self.sigma2 = np.ones(self.nbatch)
        
        if np.isnan(self.S20).any:
            self.S20 = self.sigma2  # prior parameters for the error variance
        
        # in matlab version, ny = length(ss) where ss is the output from the sos evaluation
        ny = int(self.nbatch)
        if len(self.S20)==1:
            self.S20 = np.ones(ny)*self.S20
            
        if len(self.N) == 1:
            self.N = np.ones(ny)*self.N
            
        if len(self.N) == ny + 1:
            self.N = self.N[1:] # remove first column
            
        if len(self.N0) == 1:
            self.N0 = np.ones(ny)*self.N0
            
        if len(self.sigma2) == 1:
            self.sigma2 = np.ones(ny)*self.sigma2