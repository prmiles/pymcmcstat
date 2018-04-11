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
#        self.model = BaseModelSettings()
        self.description = 'Model Settings'
        
    def define_model_settings(self, sos_function = None, prior_function = None, prior_type = 1, 
                 prior_update_function = None, prior_pars = None, model_function = None, 
                 sigma2 = None, N = None, 
                 S20 = np.nan, N0 = None, nbatch = None):
    
        self.sos_function = sos_function
        self.prior_function = prior_function
        self.prior_type = prior_type
        self.prior_update_function = prior_update_function
        self.prior_pars = prior_pars
        self.model_function = model_function
        
        # check value of sigma2 - initial error variance
        self.sigma2 = self.__array_type(sigma2)
        
        # check value of N - total number of observations
        self.N = self.__array_type(N)
        
        # check value of N0 - prior accuracy for S20
        self.N0 = self.__array_type(N0)       
        
        # check nbatch - number of data sets
        self.nbatch = self.__array_type(nbatch)
            
        # S20 - prior for sigma2
        self.S20 = self.__array_type(S20)
    
    def __array_type(self, x):
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
    
    def _check_dependent_model_settings(self, data, options):
        # check dependent parameters
        if self.nbatch is None:
            self.nbatch = data.get_number_of_batches()
            
        if self.N is not None:
            N = data.get_number_of_observations()
            if self.N == N:
                self.N = N
            else:
                print('User defined N = {}.  Estimate based on data structure is N = {}.  Possible error?'.format(self.model.N, N))
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
        
    def _check_dependent_model_settings_wrt_nsos(self, nsos):
        # in matlab version, ny = length(sos) where sos is the output from the sos evaluation
        if len(self.S20)==1:
            self.S20 = np.ones(nsos)*self.S20
            
        if len(self.N) == 1:
            self.N = np.ones(nsos)*self.N
            
        if len(self.N) == nsos + 1:
            self.N = self.N[1:] # remove first column
            
        if len(self.N0) == 1:
            self.N0 = np.ones(nsos)*self.N0
            
        if len(self.sigma2) == 1:
            self.sigma2 = np.ones(nsos)*self.sigma2
            
        self.nsos = nsos
        
    def display_model_settings(self):
        print_these = ['sos_function', 'model_function', 'sigma2', 'N', 'N0', 'S20', 'nsos', 'nbatch']
        print('model settings:')
        for ii in range(len(print_these)):
            print('\t{} = {}'.format(print_these[ii], getattr(self, print_these[ii])))
            
#class BaseModelSettings:
#    def __init__(self):
#        # Initialize all variables to default values
#        self.sos_function = None
#        self.prior_function = None
#        self.prior_type = 1
#        self.prior_update_function = None
#        self.prior_pars = None
#        self.model_function = None
#        
#        # check value of sigma2 - initial error variance
#        self.sigma2 = None
#        
#        # check value of N - total number of observations
#        self.N = None
#        
#        # check value of N0 - prior accuracy for S20
#        self.N0 = None       
#        
#        # check nbatch - number of data sets
#        self.nbatch = None
#            
#        # S20 - prior for sigma2
#        self.S20 = np.nan