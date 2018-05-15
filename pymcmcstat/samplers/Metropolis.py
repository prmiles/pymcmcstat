#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:30:29 2018

@author: prmiles
"""
# import required packages
import numpy as np
from scipy.special import expit
from ..structures.ParameterSet import ParameterSet

class Metropolis:
    # -------------------------------------------
    def run_metropolis_step(self, old_set, parameters, R, prior_object, sos_object):
           
        # unpack oldset
        oldpar, ss, oldprior, sigma2 = self.unpack_set(old_set)
        
        # Sample new candidate from Gaussian proposal
        npar_sample_from_normal = np.random.randn(1, parameters.npar)
        newpar = oldpar + np.dot(npar_sample_from_normal,R)   
        newpar = newpar.reshape(parameters.npar)
           
        # Reject points outside boundaries
        outsidebounds = self.is_sample_outside_bounds(newpar, parameters._lower_limits[parameters._parind[:]], parameters._upper_limits[parameters._parind[:]])
        if outsidebounds is True:
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
            # evaluate likelihood
            alpha = self.evaluate_likelihood_function(ss1, ss2, sigma2, newprior, oldprior)
            # make acceptance decision
            accept = self.acceptance_test(alpha)
                    
        # store parameter sets in objects    
        newset = ParameterSet(theta = newpar, ss = ss1, prior = newprior, sigma2 = sigma2, alpha = alpha)
        
        return accept, newset, outbound, npar_sample_from_normal
    
    def unpack_set(self, parset):
        theta = parset.theta
        ss = parset.ss
        prior = parset.prior
        sigma2 = parset.sigma2
        return theta, ss, prior, sigma2
    
    def is_sample_outside_bounds(self, theta, lower_limits, upper_limits):
        if (theta < lower_limits).any() or (theta > upper_limits).any():
            outsidebounds = True
        else:
            outsidebounds = False
        return outsidebounds
    
    def evaluate_likelihood_function(self, ss1, ss2, sigma2, newprior, oldprior):
        alpha = expit(-0.5*(sum((ss1 - ss2)*(sigma2**(-1))) + newprior - oldprior))
        return sum(alpha)
    
    def acceptance_test(self, alpha):
        if alpha <= 0:
                accept = 0
        elif alpha >= 1 or alpha > np.random.rand(1,1):
            accept = 1
        else:
            accept = 0
            
        return accept
    
    