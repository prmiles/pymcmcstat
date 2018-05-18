#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 08:48:00 2018

@author: prmiles
"""

#from pymcmcstat.CovarianceProcedures import CovarianceProcedures
#from pymcmcstat.SumOfSquares import SumOfSquares
#from pymcmcstat.PriorFunction import PriorFunction
from pymcmcstat.structures.ParameterSet import ParameterSet
from pymcmcstat.samplers.Metropolis import Metropolis
#from pymcmcstat.MCMC import MCMC

import unittest
import numpy as np
import unittest.mock as mock

# --------------------------
# Evaluation
# --------------------------
class UnpackSet(unittest.TestCase):

    def test_unpack_set(self):
        CL = {'theta':1.0, 'ss': 1.0, 'prior':0.0, 'sigma2': 0.0}
        parset = ParameterSet(theta = CL['theta'], ss = CL['ss'], prior = CL['prior'], sigma2 = CL['sigma2'])
        
        MA = Metropolis()
        oldpar, ss, oldprior, sigma2 = MA.unpack_set(parset)
        NL = {'theta':oldpar, 'ss': ss, 'prior':oldprior, 'sigma2': sigma2}
        self.assertDictEqual(CL,NL)
      
class OutsideBounds(unittest.TestCase):
    
    def test_outsidebounds_p1_below(self):
        MA = Metropolis()
        self.assertTrue(MA.is_sample_outside_bounds(theta = np.array([-1.0, 1.5]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])))
        
    def test_outsidebounds_p2_below(self):
        MA = Metropolis()
        self.assertTrue(MA.is_sample_outside_bounds(theta = np.array([1.0, 0.5]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])))
        
    def test_outsidebounds_p1_above(self):
        MA = Metropolis()
        self.assertTrue(MA.is_sample_outside_bounds(theta = np.array([1.5, 1.5]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])))
        
    def test_outsidebounds_p2_above(self):
        MA = Metropolis()
        self.assertTrue(MA.is_sample_outside_bounds(theta = np.array([1.0, 2.5]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])))
        
    def test_not_outsidebounds_p1_on_lowlim(self):
        MA = Metropolis()
        self.assertFalse(MA.is_sample_outside_bounds(theta = np.array([0, 1.5]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])))
        
    def test_not_outsidebounds_p2_on_lowlim(self):
        MA = Metropolis()
        self.assertFalse(MA.is_sample_outside_bounds(theta = np.array([0.5, 1]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])))
        
    def test_not_outsidebounds_p1_on_upplim(self):
        MA = Metropolis()
        self.assertFalse(MA.is_sample_outside_bounds(theta = np.array([1, 1.5]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])))
        
    def test_not_outsidebounds_p2_on_upplim(self):
        MA = Metropolis()
        self.assertFalse(MA.is_sample_outside_bounds(theta = np.array([0.5, 2]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])))
        
    def test_not_outsidebounds_all(self):
        MA = Metropolis()
        self.assertFalse(MA.is_sample_outside_bounds(theta = np.array([0.5, 1.5]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])))
        
class EvaluateLikelihood(unittest.TestCase):
    def test_size_of_alpha_for_1d_nsos(self):
        MA = Metropolis()
        ss1 = np.array([1.])
        ss2 = np.array([1.1])
        newprior = np.array([0.])
        oldprior = np.array([0.])
        sigma2 = np.array([1.])
        alpha = MA.evaluate_likelihood_function(ss1, ss2, sigma2, newprior, oldprior)
        self.assertEqual(alpha.size, 1)
        
    def test_size_of_alpha_for_2d_nsos(self):
        MA = Metropolis()
        ss1 = np.array([1., 2.])
        ss2 = np.array([1.1, 2.4])
        newprior = np.array([0.,0.])
        oldprior = np.array([0.,0.])
        sigma2 = np.array([1.,1.])
        alpha = MA.evaluate_likelihood_function(ss1, ss2, sigma2, newprior, oldprior)
        self.assertEqual(alpha.size, 1)
        
    def test_likelihood_goes_up(self):
        MA = Metropolis()
        ss1 = np.array([1., 2.])
        ss2 = np.array([1.1, 2.4])
        newprior = np.array([0.,0.])
        oldprior = np.array([0.,0.])
        sigma2 = np.array([1.,1.])
        alpha1 = MA.evaluate_likelihood_function(ss1, ss2, sigma2, newprior, oldprior)
        
        ss3 = np.array([0.4, 0.5])
        alpha2 = MA.evaluate_likelihood_function(ss3, ss2, sigma2, newprior, oldprior)
        self.assertTrue(alpha1 < alpha2)
        
    def test_likelihood_goes_down(self):
        MA = Metropolis()
        ss1 = np.array([1., 2.])
        ss2 = np.array([1.1, 2.4])
        newprior = np.array([0.,0.])
        oldprior = np.array([0.,0.])
        sigma2 = np.array([1.,1.])
        alpha1 = MA.evaluate_likelihood_function(ss1, ss2, sigma2, newprior, oldprior)
        
        ss3 = np.array([1.4, 2.5])
        alpha2 = MA.evaluate_likelihood_function(ss3, ss2, sigma2, newprior, oldprior)
        self.assertTrue(alpha2 < alpha1)
        
class Acceptance(unittest.TestCase):
    
    def test_accept_alpha_gt_1(self):
        MA = Metropolis()
        self.assertEqual(MA.acceptance_test(alpha = 1.1), 1, msg = 'accept alpha >= 1')
        
    def test_accept_alpha_eq_1(self):
        MA = Metropolis()
        self.assertEqual(MA.acceptance_test(alpha = 1), 1, msg = 'accept alpha >= 1')
        
    def test_not_accept_alpha_lt_0(self):
        MA = Metropolis()
        self.assertEqual(MA.acceptance_test(alpha = -0.1), 0, msg = 'Reject alpha <= 0')
        
    def test_not_accept_alpha_eq_0(self):
        MA = Metropolis()
        self.assertEqual(MA.acceptance_test(alpha = 0), 0, msg = 'Reject alpha <= 0')
        
    def test_not_accept_alpha_based_on_rand(self):
        MA = Metropolis()
        np.random.seed(0)
        self.assertEqual(MA.acceptance_test(alpha = 0.4), 0, msg = 'Reject alpha < u (0.4 < 0.5488135)')
        
    def test_accept_alpha_based_on_rand(self):
        MA = Metropolis()
        np.random.seed(0)
        self.assertEqual(MA.acceptance_test(alpha = 0.6), 1, msg = 'Accept alpha > u (0.6 > 0.5488135)')