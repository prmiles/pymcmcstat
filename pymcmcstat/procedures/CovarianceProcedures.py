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
    '''
    Covariance matrix variables and methods.

    Attributes:
        * :meth:`~display_covariance_settings`
        * :meth:`~setup_covariance_matrix`
    '''
    def __init__(self):
        self.description = 'Covariance Variables and Methods'

    def _initialize_covariance_settings(self, parameters, options):
        '''
        Initialize covariance settings.

        Args:
            * **parameters** (:class:`~.ModelParameters`): MCMC model parameters
            * **options** (:class:`~.SimulationOptions`): MCMC simulation options
        '''
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
        self.setup_covariance_matrix(options.qcov, parameters._thetasigma, parameters._initial_value)
            
        # ----------------
        # check adascale
        self.__check_adascale(options.adascale, parameters.npar)
        
        # ----------------
        # setup R matrix (R used to update in metropolis)
#        print('qcov.shape = {}'.format(self._qcov.shape))
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
        '''
        Update covariance from adaptation algorithm.

        Args:
            * **R** (:class:`~numpy.ndarray`): Cholesky decomposition of covariance matrix.
            * **covchain** (:class:`~numpy.ndarray`): Covariance matrix history.
            * **meanchain** (:class:`~numpy.ndarray`): Current mean chain values.
            * **wsum** (:class:`~numpy.ndarray`): Weights
            * **last_index_since_adaptation** (:py:class:`int`): Last index since adaptation occured.
            * **iiadapt** (:py:class:`int`): Adaptation counter.
        '''
        self._R = R
        self._covchain = covchain
        self._meanchain = meanchain
        self._wsum = wsum
        self._last_index_since_adaptation = last_index_since_adaptation
        self._iiadapt = iiadapt
        
    def _update_covariance_for_delayed_rejection_from_adaptation(self, RDR = None, invR = None):
        '''
        Update covariance variables for delayed rejection based on adaptation.

        Args:
            * **RDR** (:class:`~numpy.ndarray`): Cholesky decomposition of covariance matrix based on DR.
            * **invR** (:class:`~numpy.ndarray`): Inverse of Cholesky decomposition matrix.
        '''
        self._RDR = RDR
        self._invR = invR
        
    def _update_covariance_settings(self, parameter_set):
        '''
        Update covariance settings based on parameter set

        Args:
            * **parameter_set** (:class:`~numpy.ndarray`): Mean parameter values
        '''
        if self._wsum is not None:
            self._covchain = self._qcov
            self._meanchain = parameter_set
        
    def setup_covariance_matrix(self, qcov, thetasig, value):
        '''
        Initialize covariance matrix.

        If no proposal covariance matrix is provided, then the default is generated
        by squaring 5% of the initial value.  This yields a diagonal covariance matrix.

        .. math::

            V = diag([(0.05\\theta_i)^2])

        If the initial value was one, this would lead to zero variance.  In those
        instances the variance is set equal to :code:`qcov[qcov==0] = 1.0`.

        Args:
            * **qcov** (:class:`~numpy.ndarray`): Parameter covariance matrix.
            * **thetasig** (:class:`~numpy.ndarray`): Prior variance.
            * **value** (:class:`~numpy.ndarray`): Current parameter value.
        '''
        # check qcov
        if qcov is None: # i.e., qcov is None (not defined)
            qcov = thetasig**2 # variance
            ii1 = np.isinf(qcov)
            ii2 = np.isnan(qcov)
            ii = ii1 + ii2
            qcov[ii] = (np.abs(value[ii])*0.05)**2 # default is 5% stdev
            qcov[qcov==0] = 1.0 # if initial value was zero, use 1 as stdev
            qcov = np.diagflat(qcov) # create covariance matrix
    
        self._qcov = np.atleast_2d(qcov[:])
        
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
            self.__setup_R_based_on_variances(parind)
        else: # qcov has covariance matrix in it
            self.__setup_R_based_on_covariance_matrix(parind)
    
    def __setup_R_based_on_variances(self, parind):
        qcov = np.copy(self._qcov)
        qcov = np.diagflat(qcov)
        self._R = np.sqrt(qcov[np.ix_(parind,parind)])
        self._qcovorig = np.diagflat(self._qcov[:]) # save original qcov
        self._qcov = qcov[np.ix_(parind,parind)]
        
    def __setup_R_based_on_covariance_matrix(self, parind):
        self._qcovorig = np.copy(self._qcov) # save qcov
        self._qcov = self._qcov[np.ix_(parind,parind)] # this operation in matlab maintains matrix (debug)
        if self._qcov.size == 1:
            self._R = np.sqrt(self._qcov)
        else:
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
        no_adapt_index = np.zeros([len(parind)],dtype=bool)
        if len(noadaptind) == 0:
            no_adapt_index = np.zeros([len(parind)],dtype=bool)
        else:
            c = list(set.intersection(set(noadaptind), set(parind)))
            no_adapt_index[c] = np.ones([1], dtype = bool)
    
        self._no_adapt_index = no_adapt_index
        
    def display_covariance_settings(self, print_these = None):
        '''
        Display subset of the covariance settings.

        Args:
            * **print_these** (:py:class:`list`): List of strings corresponding to keywords.  Default below.

        ::

            print_these = ['qcov', 'R', 'RDR', 'invR', 'last_index_since_adaptation', 'covchain']
        '''
        if print_these is None:
            print_these = ['qcov', 'R', 'RDR', 'invR', 'last_index_since_adaptation', 'covchain']
        print('covariance:')
        for ptii in print_these:
            print('\t{} = {}'.format(ptii, getattr(self, str('_{}'.format(ptii)))))
        return print_these