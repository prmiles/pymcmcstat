#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:18:19 2018

Description: Class used to organize results of MCMC simulation.

@author: prmiles
"""

# import required packages
import numpy as np

class ResultsStructure:
    def __init__(self):
        self.results = {} # initialize empty dictionary
        
    def add_basic(self, options, model, covariance, parameters, rejected, simutime, theta):
#                  nsimu, rejected, R, covchain, meanchain, names, 
#                  lowerlims, upperlims, theta, parind, local, simutime, qcovorig):
    
        self.results['theta'] = theta
        
        self.results['parind'] = parameters._parind
        self.results['local'] = parameters._local
        
        self.results['total_rejected'] = rejected['total']*(options.nsimu**(-1)) # total rejected
        self.results['rejected_outside_bounds'] = rejected['outside_bounds']*(options.nsimu**(-1)) # rejected due to sampling outside limits
        self.results['R'] = covariance._R
        self.results['qcov'] = np.dot(covariance._R.transpose(),covariance._R)
        self.results['cov'] = covariance._covchain
        self.results['mean'] = covariance._meanchain
        self.results['names'] = [parameters._names[ii] for ii in parameters._parind]
        self.results['limits'] = [parameters._lower_limits[parameters._parind[:]], 
                     parameters._upper_limits[parameters._parind[:]]]
        
        self.results['nsimu'] = options.nsimu
        self.results['simutime'] = simutime
        covariance._qcovorig[np.ix_(parameters._parind,parameters._parind)] = self.results['qcov']
        self.results['qcovorig'] = covariance._qcovorig
    
    def add_updatesigma(self, updatesigma, sigma2, S20, N0):
        self.results['updatesigma'] = updatesigma
        if updatesigma:
            self.results['sigma2'] = np.nan
            self.results['S20'] = S20
            self.results['N0'] = N0
        else:
            self.results['sigma2'] = sigma2
            self.results['S20'] = np.nan
            self.results['N0'] = np.nan
    
    def add_dram(self, dodram, drscale, iacce, alpha_count, RDR, nsimu, rej):
        if dodram == 1:
            self.results['drscale'] = drscale
            iacce[0] = nsimu - rej - sum(iacce[1:])
            # 1 - number accepted without DR, 2 - number accepted via DR try 1, 
            # 3 - number accepted via DR try 2, etc.
            self.results['iacce'] = iacce 
            self.results['alpha_count'] = alpha_count
            self.results['RDR'] = RDR
    
    def add_prior(self, mu, sig, priorfun, priortype, priorpars):
        self.results['prior'] = [mu, sig]
        self.results['priorfun'] = priorfun
        self.results['priortype'] = priortype
        self.results['priorpars'] = priorpars
        
    def add_options(self, options = None):
        # must convert 'options' object to a dictionary
        self.results['options'] = options.__dict__

    def add_model(self, model = None):
        # must convert 'model' object to a dictionary
        self.results['model'] = model.__dict__
        
    def add_chain(self, chain = None):
        self.results['chain'] = chain
        
    def add_s2chain(self, s2chain = None):
        self.results['s2chain'] = s2chain
        
    def add_sschain(self, sschain = None):
        self.results['sschain'] = sschain
        
    def add_time_stats(self, mtime, drtime, adtime):
        self.results['time [mh, dr, am]'] = [mtime, drtime, adtime]
        
    def add_random_number_sequence(self, rndseq):
        self.results['rndseq'] = rndseq
    