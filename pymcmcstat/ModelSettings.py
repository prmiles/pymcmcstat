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
        self.model = BaseModelSettings()
        
    def update_model_settings(self, sos_function = None, prior_function = None, prior_type = 1, 
                 prior_update_function = None, prior_pars = None, model_function = None, 
                 sigma2 = None, N = None, 
                 S20 = np.nan, N0 = None, nbatch = None):
    
        self.model.sos_function = sos_function
        self.model.prior_function = prior_function
        self.model.prior_type = prior_type
        self.model.prior_update_function = prior_update_function
        self.model.prior_pars = prior_pars
        self.model.model_function = model_function
        
        # check value of sigma2 - initial error variance
        self.model.sigma2 = self.array_type(sigma2)
        
        # check value of N - total number of observations
        self.model.N = self.array_type(N)
        
        # check value of N0 - prior accuracy for S20
        self.model.N0 = self.array_type(N0)       
        
        # check nbatch - number of data sets
        self.model.nbatch = self.array_type(nbatch)
            
        # S20 - prior for sigma2
        self.model.S20 = self.array_type(S20)
    
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
        if self.model.nbatch is None:
            self.model.nbatch = data.get_number_of_batches()
            
        if self.model.N is not None:
            N = data.get_number_of_observations()
            if self.model.N == N:
                self.model.N = N
            else:
                print('User defined N = {}.  Estimate based on data structure is N = {}.  Possible error?'.format(self.model.N, N))
        else:
            self.model.N = data.get_number_of_observations()
            
        # This is for backward compatibility
        # if sigma2 given then default N0=1, else default N0=0
        if self.model.N0 is None:
            if self.model.sigma2 is None:
                self.model.sigma2 = np.ones([1])
                self.model.N0 = np.zeros([1])
            else:
                self.model.N0 = np.ones([1])
            
        # set default value for sigma2    
        # default for sigma2 is S20 or 1
        if self.model.sigma2 is None:
            if not(np.isnan(self.model.S20)).any:
                self.model.sigma2 = self.model.S20
            else:
                self.model.sigma2 = np.ones(self.model.nbatch)
        
        if np.isnan(self.model.S20).any:
            self.model.S20 = self.model.sigma2  # prior parameters for the error variance
        
    def check_dependent_model_settings_wrt_nsos(self, nsos):
        # in matlab version, ny = length(sos) where ss is the output from the sos evaluation
        if len(self.model.S20)==1:
            self.model.S20 = np.ones(nsos)*self.model.S20
            
        if len(self.model.N) == 1:
            self.model.N = np.ones(nsos)*self.model.N
            
        if len(self.model.N) == nsos + 1:
            self.model.N = self.model.N[1:] # remove first column
            
        if len(self.model.N0) == 1:
            self.model.N0 = np.ones(nsos)*self.model.N0
            
        if len(self.model.sigma2) == 1:
            self.model.sigma2 = np.ones(nsos)*self.model.sigma2
            
        self.model.nsos = nsos
        
    def display_model_settings(self):
        print_these = ['sos_function', 'model_function', 'sigma2', 'N', 'N0', 'S20', 'nsos', 'nbatch']
        print('model_settings:')
        for ii in xrange(len(print_these)):
            print('\t{} = {}'.format(print_these[ii], getattr(self.model, print_these[ii])))
            
class BaseModelSettings:
    def __init__(self):
        # Initialize all variables to default values
        self.sos_function = None
        self.prior_function = None
        self.prior_type = 1
        self.prior_update_function = None
        self.prior_pars = None
        self.model_function = None
        
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