#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:08:13 2018
    # general settings
    nsimu - number of chain interates
    method - sampling method ('mh', 'am', 'dr', 'dram')
    waitbar - flag to display progress bar
    debug - display certain features to assist in code debugging
    noadaptind - do not adapt these indices
    stats - convergence statistics
    verbosity - verbosity of display output
    printint - printing interval
    nbatch - number of batches
    rndseq - random numbers for testing
        
    # settings for adaptation
    adaptint - number of iterates between adaptation
    qcov - proposal covariance
    qcov_adjust -
    initqcovn - proposal covariance weight in update
    adascale - user defined covariance scale
    lastadapt - last adapt (i.e., no more adaptation beyond this iteration)
    burnintime -
    burnin_scale - scale in burn-in down/up
    
    # settings for updating error variance estimator
    updatesigma - flag saying whether or not to update the measurement variance estimate
    
    # settings associated with saving to bin files
    savesize - rows of the chain in memory
    maxmem - memory available in mega bytes
    chainfile - chain file name
    s2chainfile - s2chain file name
    sschainfile - sschain file name
    savedir - directory files saved to
    skip -
    label -
    
    # settings for delayed rejection
    ntry - number of stages in delayed rejection algorithm
    RDR - R matrix for each stage of delayed rejection
    drscale - scale sampling distribution for delayed rejection
    alphatarget - acceptance ratio target
   
@author: prmiles
"""

# import required packages
import numpy as np
from datetime import datetime

class SimulationOptions:
    
    def __init__(self):
        # initialize simulation option variables
        self.options = BaseSimulationOptions()
        
    def update_simulation_options(self, nsimu=10000, adaptint = None, ntry = None, method='dram',
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
            self.options.adaptint = method_dictionary[method]['adaptint']  # update interval for adaptation
        elif method == 'mh' or method == 'dr':
            self.options.adaptint = method_dictionary[method]['adaptint']  # no adaptation - enforce!
        else:
            self.options.adaptint = adaptint
        
        if ntry is None:
            self.options.ntry = method_dictionary[method]['ntry']
        else:
            self.options.ntry = ntry
            
        if adascale is None:
            self.options.adascale = method_dictionary[method]['adascale']  # qcov_scale
        else:
            self.options.adascale = adascale
            
        if doram is None:
            self.options.doram = method_dictionary[method]['doram']
        else:
            self.options.doram = doram
        
        
        self.options.nsimu = nsimu  # length of chain to simulate
        self.options.method = method
        
        self.options.printint = printint  # print interval
        self.options.adaptend = adaptend  # last adapt
        self.options.lastadapt = lastadapt # last adapt
        self.options.burnintime = burnintime
        self.options.waitbar = waitbar # use waitbar
        self.options.debug = debug  # show some debug information
        self.options.qcov = qcov  # proposal covariance
        self.options.initqcovn = initqcovn  # proposal covariance weight in update
        self.options.updatesigma = updatesigma  # 
        self.options.noadaptind = noadaptind  # do not adapt these indices
        self.options.priorupdatestart = priorupdatestart
        self.options.qcov_adjust = qcov_adjust  # eps adjustment
        self.options.burnin_scale = burnin_scale
        self.options.alphatarget = alphatarget  # acceptance ratio target
        self.options.etaparam = etaparam  #
        self.options.stats = stats  # convergence statistics
        self.options.drscale = drscale

        self.options.skip = skip
        
        if label is None:
            self.options.label = str('MCMC run at {}'.format(datetime.now().strftime("%Y%m%d_%H%M%S")))
        else:
            self.options.label = label
            
        self.options.RDR = RDR
        self.options.verbosity = verbosity # amount of information to print
        self.options.maxiter = maxiter
        self.options.savesize = savesize
        self.options.maxmem = maxmem
        self.options.chainfile = chainfile
        self.options.s2chainfile = s2chainfile
        self.options.sschainfile = sschainfile
        self.options.savedir = savedir
        
        self.options.rndseq = rndseq # define random number sequence for testing
        
    def check_dependent_simulation_options(self, data, model):
        # check dependent parameters
                
        # save options
        if self.options.savesize <= 0 or self.options.savesize > self.options.nsimu:
            self.options.savesize = self.options.nsimu
        
        # turn on DR if ntry > 1
        if self.options.ntry > 1:
            self.options.dodram = 1
        else:
            self.options.dodram = 0
            
        if self.options.lastadapt < 1:
            self.options.lastadapt = self.options.nsimu
            
        if np.isnan(self.options.printint):
            self.options.printint = max(100,min(1000,self.options.adaptint))
            
        # if N0 given, then also turn on updatesigma
        if model.N0 is not None:
            self.options.updatesigma = 1  
            
    def display_simulation_options(self):
        print_these = ['nsimu', 'adaptint', 'ntry', 'method', 'printint', 'lastadapt', 'drscale']
        print('simulation_options:')
        for ii in xrange(len(print_these)):
            print('\t{} = {}'.format(print_these[ii], getattr(self.options, print_these[ii])))
            
class BaseSimulationOptions:
    def __init__(self):
        self.nsimu = 10000
        self.adaptint = None
        self.ntry = None
        self.method = 'dram'
        self.printint = np.nan
        self.adaptend = 0
        self.lastadapt = 0
        self.burnintime = 0
        self.noadaptind = []
        self.stats = 0
        self.drscale = np.array([5,4,3], dtype = float)
        self.adascale = None
        self.savesize = 0
        self.maxmem = 0
        self.chainfile = None
        self.s2chainfile = None
        self.sschainfile = None
        self.savedir = None
        self.skip = 1
        self.priorupdatestart = 0
        self.qcov_adjust = 1e-8
        self.burnin_scale = 10
        self.alphatarget = 0.234
        self.etaparam = 0.7
        self.initqcovn = None
        self.doram = None
        self.rndseq = None