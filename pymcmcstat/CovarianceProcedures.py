#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 07:55:46 2018

Description: Support methods for initializing and updating the covariance matrix.
Additional routines associated with Cholesky Decomposition.

@author: prmiles
"""

# import required packages
import numpy as np
import math

class CovarianceProcedures:
    def __init__(self):
        self.label = 'Covariance Variables and Methods'
        self.qcov = None
        self.qcov_scale = None
        self.R = None
        self.qcov_original = None
        self.invR = None
        self.iacce = None
        
        self.covchain = None
        self.meanchain = None
        
        self.last_index_since_adaptation = 0
        
#        R, covchain, meanchain, wsum, lasti, RDR, invR, iiadapt, rejection
        
    def initialize_covariance_settings(self, parameters, options):
        
        self.wsum = options.initqcovn
        
        # define noadaptind as a boolean - inputted as list of index values not updated
        self.setup_no_adapt_index(noadaptind = options.noadaptind, parind = parameters.parind)
        
        # ----------------
        # setup covariance matrix
        self.setup_covariance_matrix(options.qcov, parameters.thetasigma, parameters.initial_value)
            
        # ----------------
        # check adascale
        self.check_adascale(options.adascale, parameters.npar)
        
        # ----------------
        # setup R matrix (R used to update in metropolis)
        self.setup_R_matrix(parameters.parind)
            
        # ----------------
        # setup RDR matrix (RDR used in DR)
        if options.method == 'dram' or options.method == 'dr':
            self.invR = []
            self.setup_RDR_matrix(npar = parameters.npar, 
                    drscale = options.drscale, ntry = options.ntry,
                    RDR = options.RDR)
            
    def update_covariance_from_adaptation(self, R, covchain, meanchain, wsum, 
                                          last_index_since_adaptation, iiadapt):
        self.R = R
        self.covchain = covchain
        self.meanchain = meanchain
        self.wsum = wsum
        self.last_index_since_adaptation = last_index_since_adaptation
        self.iiadapt = iiadapt
        
    def update_covariance_for_delayed_rejection_from_adaptation(self, RDR = None, invR = None):
        self.RDR = RDR
        self.invR = invR
        
    def update_covariance_settings(self, parameter_set):
        if self.wsum is not None:
            self.covchain = self.qcov
            self.meanchain = parameter_set
        
    def setup_covariance_matrix(self, qcov, thetasig, value):
        # check qcov
        if qcov is None: # i.e., qcov is None (not defined)
            qcov = thetasig**2 # variance
            ii1 = np.isinf(qcov)
            ii2 = np.isnan(qcov)
            ii = ii1 + ii2
            qcov[ii] = (np.abs(value[ii])*0.05)**2 # default is 5% stdev
            qcov[qcov==0] = 1 # if initial value was zero, use 1 as stdev
            qcov = np.diagflat(qcov) # create covariance matrix
    
        self.qcov = qcov        
        
    def check_adascale(self, adascale, npar):
        # check adascale
        if adascale is None or adascale <= 0:
            qcov_scale = 2.4*(math.sqrt(npar)**(-1)) # scale factor in R
        else:
            qcov_scale = adascale
        
        self.qcov_scale = qcov_scale
    
    def setup_R_matrix(self, parind):
        cm, cn = self.qcov.shape # number of rows, number of columns
        if min([cm, cn]) == 1: # qcov contains variances!
            s = np.sqrt(self.qcov[parind[:]])
            self.R = np.diagflat(s)
            self.qcovorig = np.diagflat(self.qcov[:]) # save original qcov
            self.qcov = np.diag(self.qcov[parind[:]])
        else: # qcov has covariance matrix in it
            self.qcovorig = self.qcov # save qcov
    #        qcov = qcov[parind[:],parind[:]] # this operation in matlab maintains matrix (debug)
            self.R = np.linalg.cholesky(self.qcov) # cholesky decomposition
            self.R = self.R.transpose() # matches output of matlab function
    
    def setup_RDR_matrix(self, npar, drscale, ntry, RDR):
        # if not empty
        if RDR is None: # check implementation
            RDR = [] # initialize list
            RDR.append(self.R)
            self.invR.append(np.linalg.solve(self.R, np.eye(npar)))
            for ii in range(1,ntry):
                RDR.append(RDR[ii-1]*(drscale[min(ii,len(drscale))-1]**(-1)))
                self.invR.append(np.linalg.solve(RDR[ii],np.eye(npar)))
        else: # DR strategy: just scale R's down by DR_scale
            for ii in range(ntry):
                self.invR.append(np.linalg.solve(RDR[ii], np.eye(npar)))
                    
            self.R = RDR[0]
        
        self.RDR = RDR
        
    def setup_no_adapt_index(self, noadaptind, parind):
        # define noadaptind as a boolean - inputted as list of index values not updated
        self.no_adapt_index = []
        if len(noadaptind) == 0:
            self.no_adapt_index = np.zeros([len(parind)],dtype=bool)
        else:
            for jj in range(len(noadaptind)):
                for ii in range(len(parind)):
                    if noadaptind[jj] == parind[ii]:
                        self.no_adapt_index[jj] = np.ones([1],dtype=bool)
                    else:
                        self.no_adapt_index[jj] = np.zeros([1],dtype=bool)
    
    def display_covariance_settings(self):
        print_these = ['qcov', 'R', 'RDR', 'invR', 'last_index_since_adaptation']
        print('covariance:')
        for ii in xrange(len(print_these)):
            print('\t{} = {}'.format(print_these[ii], getattr(self, print_these[ii])))