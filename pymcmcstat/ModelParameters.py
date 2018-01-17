#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:13:03 2018

@author: prmiles
"""
# import required packages
import numpy as np
import math

class Parameters:
    def __init__(self):
#        self.parameters = [] # initialize list
        self.parameters = [] # initialize list
        self.label = 'MCMC model parameters'
        
    def add_parameter(self, name, theta0, minimum = -np.inf,
                      maximum = np.inf, mu = np.zeros([1]), sig = np.inf,
                      sample = None, local = 0):
        
        # append dictionary element
        self.parameters.append({'name': name, 'theta0': theta0, 'minimum': minimum,
                                'maximum': maximum, 'mu': mu, 'sig': sig,
                                'sample': sample, 'local': local})
#        self.parameters.append([name, theta0, minimum, maximum, mu, sig, 
#                          sample, local])
        
#        parameter = [name, theta0, minimum, maximum, mu, sig, 
#                          sample, local]
        
#        self.parameters.append(parameter)
    
    
    def openparameterstructure(self, params, nbatch):
        # NEED TO ADD COMPATIBILITY WITH HYPERPARAMETERS
        
        # unpack input object
        parameters = params.parameters
        npar = len(parameters)
        
        # initialize arrays - as lists and numpy arrays (improved functionality)
        names = []
        value = np.zeros(npar)
        parind = np.ones(npar, dtype = int)
        local = np.zeros(npar)
        upp = np.ones(npar)*np.inf
        low = -np.ones(npar)*np.inf
        thetamu = np.zeros(npar)
        thetasig = np.ones(npar)*np.inf
        
        nhpar = 0 # number of hyper parameters
        
        # scan for local variables
        ii = 0
        for kk in range(npar):
            if parameters[kk]['sample'] is not None:
                if parameters[kk]['local'] != 0:
                    if parameters[kk]['local'] == 2:
                        nhpar += 1 # add hyper parameter
                        
                    local[ii:(ii+nbatch-1)] = range(0,nbatch)
                    npar = npar + nbatch - 1
                    ii = ii + nbatch - 1
                    
                    # add functionality for hyper parameters
    #                            for k=2:7
    #                if parstruct{i}{8}==2 & (k==5|k==6)
    #                    if not(length(parstruct{i}{k})==1|length(parstruct{i}{k})==2)
    #                        error(sprintf('Error in hyper parameters for %s',parstruct{i}{1}))
    #                    end
    #                else
    #                    if length(parstruct{i}{k})~=nbatch
    #                        if length(parstruct{i}{k})==1
    #                            parstruct{i}{k} = parstruct{i}{k}*ones(1,nbatch);
    #                        else
    #                            error(sprintf('Not enough values for %s',parstruct{i}{1}))
            ii += 1 # update counter
            
         
        ii = 0
        for kk in range(npar):
            if local[ii] == 0:
                names.append(parameters[kk]['name'])
                value[ii] = parameters[kk]['theta0']
                
                # default values defined in "Parameters" class in classes.py
                # define lower limits
                low[ii] = parameters[kk]['minimum']
                # define upper limits
                upp[ii] = parameters[kk]['maximum']
                # define prior mean
                thetamu[ii] = parameters[kk]['mu']
                if np.isnan(thetamu[ii]):
                    thetamu[ii] = value[ii]
                # define prior standard deviation
                thetasig[ii] = parameters[kk]['sig']
                if thetasig[ii] == 0:
                    thetasig[ii] = np.inf
                    
            ii += 1 # update counter
            
        # make parind list of nonzero elements
        parind = np.flatnonzero(parind)
                
        return names, value, parind, local, upp, low, thetamu, thetasig, npar

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
                
    def results2params(self, results, params, use_local = 1):
    
        # unpack results dictionary
        parind = results['parind']
        names = results['names']
        local = results['local']
        theta = results['theta']
        
        for ii in range(len(parind)):
            if use_local == 1 and local[parind[ii]] == 1:
                name = names[ii] # unclear usage
            else:
                name = names[ii]
    
            for kk in range(len(params.parameters)):
                if name == params.parameters[kk]['name']:
                    # change NaN prior mu (=use initial) to the original initial value
                    if np.isnan(params.parameters[kk]['mu']):
                        params.parameters[kk]['mu'] = params[kk]['theta0']
                        
                    # only change if parind = 1 in params (1 is the default)
                    if params.parameters[kk]['sample'] == 1 or params.parameters[kk]['sample'] is None:
                        params.parameters[kk]['theta0'] = theta[parind[ii]]
            
        return params