#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 08:48:00 2018

@author: prmiles
"""

from pymcmcstat.structures.ParameterSet import ParameterSet
from pymcmcstat.samplers.Metropolis import Metropolis
import test.general_functions as gf

import unittest
from mock import patch
import numpy as np

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

# --------------------------
class EvaluateLikelihood(unittest.TestCase):
    @classmethod
    def setup_size_2(cls):
        MA = Metropolis()
        ss1 = np.array([1., 2.])
        ss2 = np.array([1.1, 2.4])
        newprior = np.array([0.,0.])
        oldprior = np.array([0.,0.])
        sigma2 = np.array([1.,1.])
        return MA, ss1, ss2, newprior, oldprior, sigma2

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
        MA, ss1, ss2, newprior, oldprior, sigma2 = self.setup_size_2()
        alpha = MA.evaluate_likelihood_function(ss1, ss2, sigma2, newprior, oldprior)
        self.assertEqual(alpha.size, 1)
        
    def test_likelihood_goes_up(self):
        MA, ss1, ss2, newprior, oldprior, sigma2 = self.setup_size_2()
        alpha1 = MA.evaluate_likelihood_function(ss1, ss2, sigma2, newprior, oldprior)
        
        ss3 = np.array([0.4, 0.5])
        alpha2 = MA.evaluate_likelihood_function(ss3, ss2, sigma2, newprior, oldprior)
        self.assertTrue(alpha1 < alpha2)
        
    def test_likelihood_goes_down(self):
        MA, ss1, ss2, newprior, oldprior, sigma2 = self.setup_size_2()
        alpha1 = MA.evaluate_likelihood_function(ss1, ss2, sigma2, newprior, oldprior)
        
        ss3 = np.array([1.4, 2.5])
        alpha2 = MA.evaluate_likelihood_function(ss3, ss2, sigma2, newprior, oldprior)
        self.assertTrue(alpha2 < alpha1)
        
    def test_alpha_value(self):
        MA, ss1, ss2, newprior, oldprior, sigma2 = self.setup_size_2()
        alpha1 = MA.evaluate_likelihood_function(ss1, ss2, sigma2, newprior, oldprior)
        self.assertTrue(np.allclose(alpha1, np.array([2.568050833375483])), msg = str('alpha = {}'.format(alpha1)))
               
# --------------------------
def setup_CL(theta = 1.0, ss = 1.0, prior = 0.0, sigma2 = 0.0):
    return {'theta':theta, 'ss': ss, 'prior': prior, 'sigma2': sigma2}

class RunMetropolisStep(unittest.TestCase):
    @classmethod
    def setup_rms(cls, CL):
        sos_object, prior_object, parameters = gf.setup_mcmc_case_mh()
        MS = Metropolis()
        R = np.array([[0.4, 0.2],[0, 0.3]])
        parset = ParameterSet(theta = CL['theta'], ss = CL['ss'], prior = CL['prior'], sigma2 = CL['sigma2'])
        accept, _, outbound, npar_sample_from_normal = MS.run_metropolis_step(old_set = parset, parameters = parameters, R = R, prior_object = prior_object, sos_object = sos_object)
        
        return accept, outbound

    @patch('pymcmcstat.samplers.Metropolis.is_sample_outside_bounds', return_value = True)
    def test_run_step_outside_bounds(self, mock_1):
        accept, outbound = self.setup_rms(setup_CL(sigma2 = 1.0))
        self.assertEqual(outbound, 1, msg = 'outbound set to 1')
        self.assertEqual(accept, 0, msg = 'Not accepted because outside bounds')
        
    @patch('pymcmcstat.samplers.Metropolis.is_sample_outside_bounds', return_value = False)
    @patch('pymcmcstat.samplers.Metropolis.Metropolis.evaluate_likelihood_function', return_value = 1.1)
    def test_run_step_inside_bounds(self, mock_1, mock_2):
        accept, outbound = self.setup_rms(setup_CL())
        self.assertEqual(outbound, 0, msg = 'outbound set to 0')
        self.assertEqual(accept, 1, msg = 'Accepted because likelihood > 1')
        
    @patch('pymcmcstat.samplers.Metropolis.is_sample_outside_bounds', return_value = False)
    @patch('pymcmcstat.samplers.Metropolis.Metropolis.evaluate_likelihood_function', return_value = 0.5)
    @patch('numpy.random.rand', return_value = 0.4)
    def test_run_step_inside_bounds_test_accept(self, mock_1, mock_2, mock_3):
        accept, outbound = self.setup_rms(setup_CL())
        self.assertEqual(outbound, 0, msg = 'outbound set to 0')
        self.assertEqual(accept, 1, msg = 'Accepted because 0.5 > 0.4')
        
    @patch('pymcmcstat.samplers.Metropolis.is_sample_outside_bounds', return_value = False)
    @patch('pymcmcstat.samplers.Metropolis.Metropolis.evaluate_likelihood_function', return_value = 0.3)
    @patch('numpy.random.rand', return_value = 0.4)
    def test_run_step_inside_bounds_test_accept_fail(self, mock_1, mock_2, mock_3):
        accept, outbound = self.setup_rms(setup_CL())
        self.assertEqual(outbound, 0, msg = 'outbound set to 0')
        self.assertEqual(accept, 0, msg = 'Accepted because 0.3 < 0.4')