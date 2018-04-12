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
        self.description = 'Covariance Variables and Methods'
        
    def _initialize_covariance_settings(self, parameters, options):
        
        self._qcov = None
        self._qcov_scale = None
        self._R = None
        self._qcov_original = None
        self._invR = None
        self._iacce = None
        self._covchain = None
        self._meanchain = None
        self._last_index_since_adaptation = 0
        
        self._wsum = options.initqcovn
        
        # define noadaptind as a boolean - inputted as list of index values not updated
        self.__setup_no_adapt_index(noadaptind = options.noadaptind, parind = parameters._parind)
        
        # ----------------
        # setup covariance matrix
        self.__setup_covariance_matrix(options.qcov, parameters._thetasigma, parameters._initial_value)
            
        # ----------------
        # check adascale
        self.__check_adascale(options.adascale, parameters.npar)
        
        # ----------------
        # setup R matrix (R used to update in metropolis)
        self.__setup_R_matrix(parameters._parind)
            
        # ----------------
        # setup RDR matrix (RDR used in DR)
        if options.method == 'dram' or options.method == 'dr':
            self._invR = []
            self.__setup_RDR_matrix(npar = parameters.npar, 
                    drscale = options.drscale, ntry = options.ntry,
                    RDR = options.RDR)
            
    def _update_covariance_from_adaptation(self, R, covchain, meanchain, wsum, 
                                          last_index_since_adaptation, iiadapt):
        self._R = R
        self._covchain = covchain
        self._meanchain = meanchain
        self._wsum = wsum
        self._last_index_since_adaptation = last_index_since_adaptation
        self._iiadapt = iiadapt
        
    def _update_covariance_for_delayed_rejection_from_adaptation(self, RDR = None, invR = None):
        self._RDR = RDR
        self._invR = invR
        
    def _update_covariance_settings(self, parameter_set):
        if self._wsum is not None:
            self._covchain = self._qcov
            self._meanchain = parameter_set
        
    def __setup_covariance_matrix(self, qcov, thetasig, value):
        # check qcov
        if qcov is None: # i.e., qcov is None (not defined)
            qcov = thetasig**2 # variance
            ii1 = np.isinf(qcov)
            ii2 = np.isnan(qcov)
            ii = ii1 + ii2
            qcov[ii] = (np.abs(value[ii])*0.05)**2 # default is 5% stdev
            qcov[qcov==0] = 1 # if initial value was zero, use 1 as stdev
            qcov = np.diagflat(qcov) # create covariance matrix
    
        self._qcov = qcov        
        
    def __check_adascale(self, adascale, npar):
        # check adascale
        if adascale is None or adascale <= 0:
            qcov_scale = 2.4*(math.sqrt(npar)**(-1)) # scale factor in R
        else:
            qcov_scale = adascale
        
        self._qcov_scale = qcov_scale
    
    def __setup_R_matrix(self, parind):
        cm, cn = self._qcov.shape # number of rows, number of columns
        if min([cm, cn]) == 1: # qcov contains variances!
            s = np.sqrt(self._qcov[parind[:]])
            self._R = np.diagflat(s)
            self._qcovorig = np.diagflat(self._qcov[:]) # save original qcov
            self._qcov = np.diag(self._qcov[parind[:]])
        else: # qcov has covariance matrix in it
            self._qcovorig = self._qcov # save qcov
    #        qcov = qcov[parind[:],parind[:]] # this operation in matlab maintains matrix (debug)
            self._R = np.linalg.cholesky(self._qcov) # cholesky decomposition
            self._R = self._R.transpose() # matches output of matlab function
    
    def __setup_RDR_matrix(self, npar, drscale, ntry, RDR):
        # if not empty
        if RDR is None: # check implementation
            RDR = [] # initialize list
            RDR.append(self._R)
            self._invR.append(np.linalg.solve(self._R, np.eye(npar)))
            for ii in range(1,ntry):
                RDR.append(RDR[ii-1]*(drscale[min(ii,len(drscale))-1]**(-1)))
                self._invR.append(np.linalg.solve(RDR[ii],np.eye(npar)))
        else: # DR strategy: just scale R's down by DR_scale
            for ii in range(ntry):
                self._invR.append(np.linalg.solve(RDR[ii], np.eye(npar)))
                    
            self._R = RDR[0]
        
        self._RDR = RDR
        
    def __setup_no_adapt_index(self, noadaptind, parind):
        # define noadaptind as a boolean - inputted as list of index values not updated
        self._no_adapt_index = []
        if len(noadaptind) == 0:
            self._no_adapt_index = np.zeros([len(parind)],dtype=bool)
        else:
            for jj in range(len(noadaptind)):
                for ii in range(len(parind)):
                    if noadaptind[jj] == parind[ii]:
                        self._no_adapt_index[jj] = np.ones([1],dtype=bool)
                    else:
                        self._no_adapt_index[jj] = np.zeros([1],dtype=bool)
    
    def display_covariance_settings(self):
        print_these = ['qcov', 'R', 'RDR', 'invR', 'last_index_since_adaptation', 'covchain']
        print('covariance:')
        for ii in range(len(print_these)):
            print('\t{} = {}'.format(print_these[ii], getattr(self, str('_{}'.format(print_these[ii])))))