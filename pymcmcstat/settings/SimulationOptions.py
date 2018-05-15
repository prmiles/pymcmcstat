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
#import os

class SimulationOptions:
    
    def __init__(self):
        # initialize simulation option variables
#        self.options = BaseSimulationOptions()
        self.description = 'simulation_options'
        self.__options_set = False
        
    def define_simulation_options(self, nsimu=10000, adaptint = None, ntry = None, method='dram',
                 printint=np.nan, adaptend = 0, lastadapt = 0, burnintime = 0,
                 waitbar = 1, debug = 0, qcov = None, updatesigma = 0, 
                 noadaptind = [], stats = 0, drscale = np.array([5, 4, 3], dtype = float),
                 adascale = None, savesize = 0, maxmem = 0, chainfile = 'chainfile',
                 s2chainfile = 's2chainfile', sschainfile = 'sschainfile', covchainfile = 'covchainfile', savedir = None, 
                 save_to_bin = False, skip = 1, label = None, RDR = None, verbosity = 1, maxiter = None, 
                 priorupdatestart = 0, qcov_adjust = 1e-8, burnin_scale = 10, 
                 alphatarget = 0.234, etaparam = 0.7, initqcovn = None,
                 doram = None, rndseq = None, results_filename = None, save_to_json = False, save_to_txt = False,
                 json_restart_file = None):
        
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
        
        datestr = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.datestr = datestr
        
        if label is None:
            self.label = str('MCMC run at {}'.format(datestr))
        else:
            self.label = label
            
        self.RDR = RDR
        self.verbosity = verbosity # amount of information to print
        self.maxiter = maxiter
        
        # log settings
        self.savesize = savesize
        self.maxmem = maxmem
            
        self.chainfile = chainfile
        self.s2chainfile = s2chainfile
        self.sschainfile = sschainfile
        self.covchainfile = covchainfile
        
        if savedir is None:
            self.savedir = str('{}_{}'.format(datestr,'chain_log'))
        else:
            self.savedir = savedir
            
        self.save_to_bin = save_to_bin
        self.save_to_txt = save_to_txt
        
        self.results_filename = results_filename
        self.save_to_json = save_to_json
        self.json_restart_file = json_restart_file
        
        self.__options_set = True # options have been defined
        
    def _check_dependent_simulation_options(self, data, model):
        # check dependent parameters
                
        # save options
        if self.savesize <= 0 or self.savesize > self.nsimu:
            self.savesize = self.nsimu
        
        # turn on DR if ntry > 1
        if self.ntry > 1:
            self.dodram = 1
        else:
            self.dodram = 0
            
        if self.lastadapt < 1:
            self.lastadapt = self.nsimu
            
        if np.isnan(self.printint):
            self.printint = max(100,min(1000,self.adaptint))
            
        # if N0 given, then also turn on updatesigma
        if model.N0 is not None:
            self.updatesigma = 1  
            
    def display_simulation_options(self):
        print_these = ['nsimu', 'adaptint', 'ntry', 'method', 'printint', 'lastadapt', 'drscale', 'qcov']
        print('simulation options:')
        for ii in range(len(print_these)):
            print('\t{} = {}'.format(print_these[ii], getattr(self, print_these[ii])))
            
#class BaseSimulationOptions:
#    def __init__(self):
#        self.nsimu = 10000
#        self.adaptint = None
#        self.ntry = None
#        self.method = 'dram'
#        self.printint = np.nan
#        self.adaptend = 0
#        self.lastadapt = 0
#        self.burnintime = 0
#        self.noadaptind = []
#        self.stats = 0
#        self.drscale = np.array([5,4,3], dtype = float)
#        self.adascale = None
#        self.savesize = 0
#        self.maxmem = 0
#        self.chainfile = None
#        self.s2chainfile = None
#        self.sschainfile = None
#        self.savedir = None
#        self.skip = 1
#        self.priorupdatestart = 0
#        self.qcov_adjust = 1e-8
#        self.burnin_scale = 10
#        self.alphatarget = 0.234
#        self.etaparam = 0.7
#        self.initqcovn = None
#        self.doram = None
#        self.rndseq = None