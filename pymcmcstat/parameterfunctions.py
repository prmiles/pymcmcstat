#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 14:48:24 2017

@author: prmiles
"""

import math
import numpy as np

def openparameterstructure(params, nbatch):
    # NEED TO ADD COMPATIBILITY WITH HYPERPARAMETERS
    
    # unpack input object
    parameters = params.parameters
    npar = len(parameters)
    
    # initialize arrays - as lists and numpy arrays (improved functionality)
    names = []
    values = np.zeros(npar)
    parind = []
    local = np.zeros(npar)
    upp = np.zeros(npar)
    low = np.zeros(npar)
    thetamu = np.zeros(npar)
    thetasig = np.zeros(npar)
    
    for ii in range(npar):
        names.append(parameters[ii][0])
        values[ii] = parameters[ii][1]
        parind.append(ii)
        local[ii] = parameters[ii][7]
        upp[ii] = parameters[ii][3]
        low[ii] = parameters[ii][2]
        thetamu[ii] = parameters[ii][4]
        thetasig[ii] = parameters[ii][5]
        
    return names, values, parind, local, upp, low, thetamu, thetasig, npar

def display_parameter_settings(parind, names, value, low, upp, thetamu, 
                               thetasig, noadaptind):
        
        print('Sampling these parameters:')
        print('{:10s} {:>7s} [{:>6s}, {:>6s}] N({:>4s}, {:>4s})'.format('name',
              'start', 'min', 'max', 'mu', 'sigma^2'))
        nprint = len(parind)
        for ii in range(nprint):
            if ii in noadaptind: # THIS NEEDS TO BE FIXED!
                st = ' (*)'
            else:
                st = ''
            if math.isinf(thetasig[parind[ii]]):
                h2 = ''
            else:
                h2 = '^2'
                
            if parind[ii] > 1e4:
                print('{:10}: {:6.2e} [{:6.2e}, {:6.2e}] N({:4.2f},{:4.2f}{:s}){:s}'.format(names[parind[ii]], 
                  value[parind[ii]], low[parind[ii]], upp[parind[ii]],
                  thetamu[parind[ii]], thetasig[parind[ii]], h2, st))
            else:
                print('{:10}: {:6.2f} [{:6.2f}, {:6.2f}] N({:4.2f},{:4.2f}{:s}){:s}'.format(names[parind[ii]], 
                  value[parind[ii]], low[parind[ii]], upp[parind[ii]],
                  thetamu[parind[ii]], thetasig[parind[ii]], h2, st))
                
def results2params(results, params, use_local = 1):
    
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
            if name == params.parameters[kk][0]:
                # change NaN prior mu (=use initial) to the original initial value
                if np.isnan(params.parameters[kk][4]):
                    params.parameters[kk][4] = params[kk][1]
                    
                # only change if parind = 1 in params (1 is the default)
                if params.parameters[kk][6] == 1 or params.parameters[kk][6] is None:
                    params.parameters[kk][1] = theta[parind[ii]]
        
    return params