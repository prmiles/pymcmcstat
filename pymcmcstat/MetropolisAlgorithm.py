#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:30:29 2018

@author: prmiles
"""
# import required packages
import numpy as np
from scipy.special import expit
from .ParameterSet import ParameterSet

class MetropolisAlgorithm:
    # -------------------------------------------
    def run_metropolis_step(self, old_set, parameters, R, prior_object, sos_object):
           
        # unpack oldset
        oldpar = old_set.theta
        ss = old_set.ss
        oldprior = old_set.prior
        sigma2 = old_set.sigma2
        
        # Sample new candidate from Gaussian proposal
        npar_sample_from_normal = np.random.randn(1, parameters.npar)
        newpar = oldpar + np.dot(npar_sample_from_normal,R)   
        newpar = newpar.reshape(parameters.npar)
           
        # Reject points outside boundaries
        if (newpar < parameters._lower_limits[parameters._parind[:]]).any() or (newpar > parameters._upper_limits[parameters._parind[:]]).any():
            # proposed value outside parameter limits
            accept = 0
            newprior = 0
            alpha = 0
            ss1 = np.inf
            ss2 = ss
            outbound = 1
        else:
            outbound = 0
            # prior SS for the new theta
            newprior = prior_object.evaluate_prior(newpar) 
            
            # calculate sum-of-squares
            ss2 = ss # old ss
            ss1 = sos_object.evaluate_sos_function(newpar)
            
            # evaluate test
#            alpha = np.exp(-0.5*(sum((ss1 - ss2)*(sigma2**(-1))) + newprior - oldprior))
#            alpha = sum(alpha)
            alpha = self.evaluate_likelihood_function(ss1, ss2, sigma2, newprior, oldprior)
            
            if alpha <= 0:
                accept = 0
            elif alpha >= 1 or alpha > np.random.rand(1,1):
                accept = 1
            else:
                accept = 0
                    
        # store parameter sets in objects    
        newset = ParameterSet(theta = newpar, ss = ss1, prior = newprior,
                                sigma2 = sigma2, alpha = alpha)
        
        return accept, newset, outbound, npar_sample_from_normal
    
    def evaluate_likelihood_function(self, ss1, ss2, sigma2, newprior, oldprior):
        alpha = expit(-0.5*(sum((ss1 - ss2)*(sigma2**(-1))) + newprior - oldprior))
        return sum(alpha)