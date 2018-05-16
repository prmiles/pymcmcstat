#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:42:07 2018

@author: prmiles
"""
# import required packages
import numpy as np
from scipy.special import expit
from ..structures.ParameterSet import ParameterSet

class DelayedRejection:
    """
    Delayed Rejection (DR) algorithm based on [haario2006dram]_
        
    .. [haario2006dram] `Haario, Heikki, Marko Laine, Antonietta Mira, and Eero Saksman. "DRAM: efficient adaptive MCMC." Statistics and Computing 16, no. 4 (2006): 339-354. <https://link.springer.com/article/10.1007/s11222-006-9438-0>`_
        
    """
        # -------------------------------------------
    def run_delayed_rejection(self, old_set, new_set, RDR, ntry, parameters, invR, sosobj, priorobj):
        """
        Perform delayed rejection step - occurs in standard metropolis is not accepted.
        
        **Args:**
            * **old_set** (:class:`~.ParameterSet`): Features of :math:`q^{k-1}`
            * **new_set** (:class:`~.ParameterSet`): Features of :math:`q^*`
            * **RDR** (:class:`~numpy.ndarray`): Cholesky decomposition of parameter covariance matrix for DR steps
            * **ntry** (:py:class:`int`): Number of DR steps to perform until rejection
            * **parameters** (:class:`~.ModelParameters`): Model parameters
            * **invR** (:class:`~numpy.ndarray`): Inverse Cholesky decomposition matrix
            * **sosobj** (:class:`~.SumOfSquares`): Sum-of-Squares function
            * **priorobj** (:class:`~.PriorFunction`): Prior function

        **Returns:**
            * **accept** (:py:class:`int`): 0 - reject, 1 - accept
            * **out_set** (:class:`~.ParameterSet`): If accept == 1, then latest DR set; Else, :math:`q^k=q^{k-1}`
            * **outbound** (:py:class:`int`): 1 - rejected due to sampling outside of parameter bounds
            
        """
        # create trypath
        trypath = [old_set, new_set]
        itry = 1; # dr step index
        accept = 0 # initialize acceptance criteria
        while accept == 0 and itry < ntry:
            itry = itry + 1 # update dr step index
            # initialize next step parameter set
            next_set, u = self.initialize_next_metropolis_step(parameters.npar, old_set, new_set, RDR, itry)
                    
            # Reject points outside boundaries
            outsidebounds = self._is_sample_outside_bounds(next_set.theta, parameters._lower_limits[parameters._parind[:]], parameters._upper_limits[parameters._parind[:]])
            if outsidebounds is True:
                out_set, next_set, trypath, outbound = self._outside_bounds(old_set = old_set, next_set = next_set, trypath = trypath)
                continue
                    
            # Evaluate new proposals
            outbound = 0
            next_set.ss = sosobj.evaluate_sos_function(next_set.theta)
            next_set.prior = priorobj.evaluate_prior(next_set.theta)
            trypath.append(next_set) # add set to trypath
            alpha = self.__alphafun(trypath, invR)
            trypath[-1].alpha = alpha # add propability ratio
                           
            # check results of delayed rejection
            accept, out_set = self.acceptance_test(alpha = alpha, old_set = old_set, next_set = next_set, itry = itry)
                
        return accept, out_set, outbound
    
    def initialize_next_metropolis_step(self, npar, old_set, new_set, RDR, itry):
        '''
        Take metropolis step according to DR
        
        **Args:**
            * **npar** (:py:class:`int`): Number of parameters
            * **old_set** (:class:`~.ParameterSet`): Features of :math:`q^{k-1}`
            * **new_set** (:class:`~.ParameterSet`): Features of :math:`q^*`
            * **RDR** (:class:`~numpy.ndarray`): Cholesky decomposition of parameter covariance matrix for DR steps
            * **itry** (:py:class:`int`): DR step counter
            
        **Returns:**
            * **next_set** (:class:`~.ParameterSet`): New proposal set
            * **u** (:class:`numpy.ndarray`): Numbers sampled from standard normal distributions (:code:`u.shape = (1,npar)`)
        
        '''
        next_set = ParameterSet()
        u = np.random.randn(1,npar) # u
        next_set.theta = old_set.theta + np.dot(u,RDR[itry-1])
        next_set.theta = next_set.theta.reshape(npar)
        next_set.sigma2 = new_set.sigma2
        return next_set, u
    
    def acceptance_test(self, alpha, old_set, next_set, itry):
        '''
        Run acceptance test
        
        **Args:**
            * **alpha** (:py:class:`float`): Result of likelihood function according to delayed rejection
            * **old_set** (:class:`~.ParameterSet`): Features of :math:`q^{k-1}`
            * **next_set** (:class:`~.ParameterSet`): New proposal set
            * **itry** (:py:class:`int`): DR step counter
            
        **Returns:**
            * **accept** (:py:class:`int`): 0 - reject, 1 - accept
            * **out_set** (:class:`~.ParameterSet`): If accept == 1, then latest DR set; Else, :math:`q^k=q^{k-1}`
        '''
        if alpha >= 1 or np.random.rand(1) < alpha: # accept
            accept = 1
            out_set = next_set
            self.iacce[itry-1] += 1 # number accepted from DR
        else:
            accept = 0
            out_set = old_set
        return accept, out_set
    
    def _initialize_dr_metrics(self, options):
        self.iacce = np.zeros(options.ntry, dtype = int)
        self.dr_step_counter = 0
    
    def _is_sample_outside_bounds(self, theta, lower_limits, upper_limits):
        if (theta < lower_limits).any() or (theta > upper_limits).any():
            outsidebounds = True
        else:
            outsidebounds = False
        return outsidebounds
    
    def _outside_bounds(self, old_set, next_set, trypath):
        next_set.alpha = 0
        next_set.prior = 0
        next_set.ss = np.inf
        trypath.append(next_set)
        outbound = 1
        out_set = old_set
        
        return out_set, next_set, trypath, outbound
        
    def __alphafun(self, trypath, invR):
        '''
        Calculate likelihood according to DR
        
        **Args:**
            * **trypath** (:py:class:`list`): Sequence of DR steps
            * **invR** (:class:`~numpy.ndarray`): Inverse Cholesky decomposition matrix
            
        **Returns:**
            * **alpha** (:py:class:`float`): Result of likelihood function according to delayed rejection
        '''
        self.dr_step_counter = self.dr_step_counter + 1
        stage = len(trypath) - 1 # The stage we're in, elements in trypath - 1
        # recursively compute past alphas
        a1 = 1 # initialize
        a2 = 1 # initialize
        for k in range(0,stage-1):
            tmp1 = self.__alphafun(trypath[0:(k+2)], invR)
            a1 = a1*(1 - tmp1)
            tmp2 = self.__alphafun(trypath[stage:stage-k-2:-1], invR)
            a2 = a2*(1 - tmp2)
            if a2 == 0: # we will come back with prob 1
                alpha = np.zeros(1)
                return alpha
            
        y = self.__logposteriorratio(trypath[0], trypath[-1])
        
        for k in range(stage):
            y = y + self.__qfun(k, trypath, invR)
            
        alpha = min(np.ones(1), expit(y)*a2*(a1**(-1)))
        
        return alpha
        
    def __qfun(self, iq, trypath, invR):
        # Gaussian nth stage log proposal ratio
        # log of q_i(y_n,...,y_{n-j})/q_i(x,y_1,...,y_j)
            
        stage = len(trypath) - 1 - 1 # - 1, iq; 
        if stage == iq: # shift index due to 0-indexing
            zq = np.zeros(1) # we are symmetric
        else:
            iR = invR[iq] # proposal^(-1/2)
            y1 = trypath[0].theta
            y2 = trypath[iq + 1].theta # check index
            y3 = trypath[stage + 1].theta
            y4 = trypath[stage - iq].theta
            zq = -0.5*((np.linalg.norm(np.dot(y4-y3, iR)))**2 - (np.linalg.norm(np.dot(y2-y1, iR)))**2)
            
        return zq 
        
    def __logposteriorratio(self, x1, x2):
        zq = -0.5*(sum((x2.ss*(x2.sigma2**(-1.0)) - x1.ss*(x1.sigma2**(-1.0)))) + x2.prior - x1.prior)
        return sum(zq)