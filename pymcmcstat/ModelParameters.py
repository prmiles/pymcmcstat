#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:13:03 2018

@author: prmiles
"""
# import required packages
import numpy as np
import math
import sys

class ModelParameters:
    def __init__(self):
        self.parameters = [] # initialize list
        self.label = 'MCMC model parameters'
        
    def add_model_parameter(self, name, theta0, minimum = -np.inf,
                      maximum = np.inf, prior_mu = np.zeros([1]), prior_sigma = np.inf,
                      sample = None, local = 0):
        
        # append dictionary element
        self.parameters.append({'name': name, 'theta0': theta0, 'minimum': minimum,
                                'maximum': maximum, 'prior_mu': prior_mu, 'prior_sigma': prior_sigma,
                                'sample': sample, 'local': local})
    
    def openparameterstructure(self, nbatch):
        
        # unpack input object
        parameters = self.parameters
        npar = len(parameters)
        
        # initialize arrays - as lists and numpy arrays (improved functionality)
        self.names = []
        self.initial_value = np.zeros(npar)
        self.parind = np.ones(npar, dtype = int)
        self.local = np.zeros(npar)
        self.upper_limits = np.ones(npar)*np.inf
        self.lower_limits = -np.ones(npar)*np.inf
        self.thetamu = np.zeros(npar)
        self.thetasigma = np.ones(npar)*np.inf
                
        # scan for local variables
        ii = 0
        for kk in range(npar):
            if parameters[kk]['sample'] is not None:
                if parameters[kk]['local'] != 0:
                    self.local[ii:(ii+nbatch-1)] = range(0,nbatch)
                    npar = npar + nbatch - 1
                    ii = ii + nbatch - 1

            ii += 1 # update counter
            
         
        ii = 0
        for kk in range(npar):
            if self.local[ii] == 0:
                self.names.append(parameters[kk]['name'])
                self.initial_value[ii] = parameters[kk]['theta0']
                
                # default values defined in "Parameters" class in classes.py
                # define lower limits
                self.lower_limits[ii] = parameters[kk]['minimum']
                # define upper limits
                self.upper_limits[ii] = parameters[kk]['maximum']
                # define prior mean
                self.thetamu[ii] = parameters[kk]['prior_mu']
                if np.isnan(self.thetamu[ii]):
                    self.thetamu[ii] = self.value[ii]
                # define prior standard deviation
                self.thetasigma[ii] = parameters[kk]['prior_sigma']
                if self.thetasigma[ii] == 0:
                    self.thetasigma[ii] = np.inf
                    
            ii += 1 # update counter
            
        # make parind list of nonzero elements
        self.parind = np.flatnonzero(self.parind)

    def display_parameter_settings(self, parind, names, value, low, upp, thetamu, 
                               thetasig, noadaptind):
        
        print('Sampling these parameters:')
        print('{:10s} {:>7s} [{:>6s}, {:>6s}] N({:>4s}, {:>4s})'.format('name',
              'start', 'min', 'max', 'mu', 'sigma^2'))
        nprint = len(parind)
        for ii in range(nprint):
            if ii in noadaptind: # THIS PARAMETER IS FIXED
                st = ' (*)'
            else:
                st = ''
            if math.isinf(thetasig[parind[ii]]):
                h2 = ''
            else:
                h2 = '^2'
                
            if value[parind[ii]] > 1e4:
                print('{:10}: {:6.2g} [{:6.2g}, {:6.2g}] N({:4.2g},{:4.2f}{:s}){:s}'.format(names[parind[ii]], 
                  value[parind[ii]], low[parind[ii]], upp[parind[ii]],
                  thetamu[parind[ii]], thetasig[parind[ii]], h2, st))
            else:
                print('{:10}: {:6.2f} [{:6.2f}, {:6.2f}] N({:4.2f},{:4.2f}{:s}){:s}'.format(names[parind[ii]], 
                  value[parind[ii]], low[parind[ii]], upp[parind[ii]],
                  thetamu[parind[ii]], thetasig[parind[ii]], h2, st))
                
    def results_to_params(self, results, use_local = 1):
    
        # unpack results dictionary
        parameters = self.parameters
        parind = results['parind']
        names = results['names']
        local = results['local']
        theta = results['theta']
        
        for ii in range(len(parind)):
            if use_local == 1 and local[parind[ii]] == 1:
                name = names[ii] # unclear usage
            else:
                name = names[ii]
    
            for kk in range(len(parameters)):
                if name == parameters[kk]['name']:
                    # change NaN prior mu (=use initial) to the original initial value
                    if np.isnan(parameters[kk]['mu']):
                        parameters[kk]['mu'] = parameters[kk]['theta0']
                        
                    # only change if parind = 1 in params (1 is the default)
                    if parameters[kk]['sample'] == 1 or parameters[kk]['sample'] is None:
                        parameters[kk]['theta0'] = theta[parind[ii]]
    
    def check_initial_values_wrt_parameter_limits(self):
        # check initial parameter values are inside range
        if (self.initial_value < self.lower_limits[self.parind[:]]).any() or (self.initial_value > self.upper_limits[self.parind[:]]).any():
            # proposed value outside parameter limits
            sys.exit('Proposed value outside parameter limits - select new initial parameter values')
            
    def check_prior_sigma(self, verbosity):
        self.message(verbosity, 2, 'If prior variance <= 0, setting to Inf\n')
        self.thetasigma = self.replace_list_elements(self.thetasigma, self.less_than_or_equal_to_zero, float('Inf'))
        
    def less_than_or_equal_to_zero(self, x):
        return (x<=0)

    def replace_list_elements(self, x, testfunction, value):
        for ii in range(len(x)):
            if testfunction(x[ii]):
                x[ii] = value
        return x
    
    def message(self, verbosity, level, printthis):
        printed = False
        if verbosity >= level:
            print(printthis)
            printed = True
        return printed