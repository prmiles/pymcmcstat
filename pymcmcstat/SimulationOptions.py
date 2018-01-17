#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:08:13 2018

@author: prmiles
"""

# import required packages
import numpy as np
from datetime import datetime

class SimulationOptions:
    """Simulator Options"""
    def __init__(self, nsimu=10000, adaptint = None, ntry = None, method='dram',
                 printint=np.nan, adaptend = 0, lastadapt = 0, burnintime = 0,
                 waitbar = 1, debug = 0, qcov = None, updatesigma = 0, 
                 noadaptind = [], stats = 0, drscale = np.array([5, 4, 3], dtype = float),
                 adascale = None, savesize = 0, maxmem = 0, chainfile = None,
                 s2chainfile = None, sschainfile = None, savedir = None, skip = 1,
                 label = None, RDR = None, verbosity = 1, maxiter = None, 
                 priorupdatestart = 0, qcov_adjust = 1e-8, burnin_scale = 10, 
                 alphatarget = 0.234, etaparam = 0.7, initqcovn = None,
                 doram = None, rndseq = None):
        
        method_dictionary = {
            'mh': {'adaptint': 0, 'ntry': 1, 'doram': 0, 'adascale': adascale}, 
            'am': {'adaptint': 100, 'ntry': 1, 'doram': 0, 'adascale': adascale},
            'dr': {'adaptint': 0, 'ntry': 2, 'doram': 0, 'adascale': adascale},
            'dram': {'adaptint': 100, 'ntry': 2, 'doram': 0, 'adascale': adascale},
            'ram': {'adaptint': 1, 'ntry': 1, 'doram': 1, 'adascale': 1},
            }
        
        # define items from dictionary
        if adaptint is None:
            self.adaptint = method_dictionary[method]['adaptint']  # update interval for adaptation
        elif method == 'mh' or method == 'dr':
            self.adaptint = method_dictionary[method]['adaptint']  # no adaptation - enforce!
        else:
            self.adaptint = adaptint
        
        if ntry is None:
            self.ntry = method_dictionary[method]['ntry']
        else:
            self.ntry = ntry
            
        if adascale is None:
            self.adascale = method_dictionary[method]['adascale']  # qcov_scale
        else:
            self.adascale = adascale
            
        if doram is None:
            self.doram = method_dictionary[method]['doram']
        else:
            self.doram = doram
        
        
        self.nsimu = nsimu  # length of chain to simulate
        self.method = method
        
        self.printint = printint  # print interval
        self.adaptend = adaptend  # last adapt
        self.lastadapt = lastadapt # last adapt
        self.burnintime = burnintime
        self.waitbar = waitbar # use waitbar
        self.debug = debug  # show some debug information
        self.qcov = qcov  # proposal covariance
        self.initqcovn = initqcovn  # proposal covariance weight in update
        self.updatesigma = updatesigma  # 
        self.noadaptind = noadaptind  # do not adapt these indices
        self.priorupdatestart = priorupdatestart
        self.qcov_adjust = qcov_adjust  # eps adjustment
        self.burnin_scale = burnin_scale
        self.alphatarget = alphatarget  # acceptance ratio target
        self.etaparam = etaparam  #
        self.stats = stats  # convergence statistics
        self.drscale = drscale

        self.skip = skip
        
        if label is None:
            self.label = str('MCMC run at {}'.format(datetime.now().strftime("%Y%m%d_%H%M%S")))
        else:
            self.label = label
            
        self.RDR = RDR
        self.verbosity = verbosity # amount of information to print
        self.maxiter = maxiter
        self.savesize = savesize
        self.maxmem = maxmem
        self.chainfile = chainfile
        self.s2chainfile = s2chainfile
        self.sschainfile = sschainfile
        self.savedir = savedir
        
        self.rndseq = rndseq # define random number sequence for testing
        