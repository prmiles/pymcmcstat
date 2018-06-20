#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 08:48:00 2018

@author: prmiles
"""

from pymcmcstat.samplers.DelayedRejection import DelayedRejection
from pymcmcstat.structures.ParameterSet import ParameterSet
from pymcmcstat.MCMC import MCMC

import unittest
from mock import patch
import numpy as np

# define test model function
def modelfun(xdata, theta):
    m = theta[0]
    b = theta[1]
    nrow = xdata.shape[0]
    y = np.zeros([nrow,1])
    y[:,0] = m*xdata.reshape(nrow,) + b
    return y

def ssfun(theta, data, local = None):
    xdata = data.xdata[0]
    ydata = data.ydata[0]
    # eval model
    ymodel = modelfun(xdata, theta)
    # calc sos
    ss = sum((ymodel[:,0] - ydata[:,0])**2)
    return ss

def setup_mcmc():
    # Initialize MCMC object
    mcstat = MCMC()
    # Add data
    nds = 100
    x = np.linspace(2, 3, num=nds)
    y = 2.*x + 3. + 0.1*np.random.standard_normal(x.shape)
    mcstat.data.add_data_set(x, y)

    mcstat.simulation_options.define_simulation_options(nsimu = int(2.0e2), updatesigma = 1, method = 'dram', verbosity = 0)
    
    # update model settings
    mcstat.model_settings.define_model_settings(sos_function = ssfun)
    
    mcstat.parameters.add_model_parameter(name = 'm', theta0 = 2., minimum = -10, maximum = np.inf, sample = 1)
    mcstat.parameters.add_model_parameter(name = 'b', theta0 = -5., minimum = -10, maximum = 100, sample = 1)
    mcstat._initialize_simulation()
    
    # extract components
    model = mcstat.model_settings
    options = mcstat.simulation_options
    parameters = mcstat.parameters
    data = mcstat.data
    covariance = mcstat._covariance
    rejected = {'total': 10, 'outside_bounds': 2}
    chain = np.zeros([options.nsimu, 2])
    s2chain = np.zeros([options.nsimu, 1])
    sschain = np.zeros([options.nsimu, 1])
    return model, options, parameters, data, covariance, rejected, chain, s2chain, sschain

# --------------------------
class OutsideBounds(unittest.TestCase):
    def test_outsidebounds_p1_below(self):
        DR = DelayedRejection()
        self.assertTrue(DR._is_sample_outside_bounds(theta = np.array([-1.0, 1.5]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])), msg = 'p1 below lower limit')
        
    def test_outsidebounds_p2_below(self):
        DR = DelayedRejection()
        self.assertTrue(DR._is_sample_outside_bounds(theta = np.array([1.0, 0.5]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])), msg = 'p2 below lower limit')
        
    def test_outsidebounds_p1_above(self):
        DR = DelayedRejection()
        self.assertTrue(DR._is_sample_outside_bounds(theta = np.array([1.5, 1.5]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])), msg = 'p1 above upper limit')
        
    def test_outsidebounds_p2_above(self):
        DR = DelayedRejection()
        self.assertTrue(DR._is_sample_outside_bounds(theta = np.array([1.0, 2.5]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])), msg = 'p2 above upper limit')
        
    def test_not_outsidebounds_p1_on_lowlim(self):
        DR = DelayedRejection()
        self.assertFalse(DR._is_sample_outside_bounds(theta = np.array([0, 1.5]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])), msg = 'p1 on lower limit')
        
    def test_not_outsidebounds_p2_on_lowlim(self):
        DR = DelayedRejection()
        self.assertFalse(DR._is_sample_outside_bounds(theta = np.array([0.5, 1]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])), msg = 'p2 on lower limit')
        
    def test_not_outsidebounds_p1_on_upplim(self):
        DR = DelayedRejection()
        self.assertFalse(DR._is_sample_outside_bounds(theta = np.array([1, 1.5]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])), msg = 'p1 on upper limit')
        
    def test_not_outsidebounds_p2_on_upplim(self):
        DR = DelayedRejection()
        self.assertFalse(DR._is_sample_outside_bounds(theta = np.array([0.5, 2]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])), msg = 'p2 on upper limit')
        
    def test_not_outsidebounds_all(self):
        DR = DelayedRejection()
        self.assertFalse(DR._is_sample_outside_bounds(theta = np.array([0.5, 1.5]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])), msg = 'within limits')
        
# --------------------------        
class Acceptance(unittest.TestCase):
    def test_accept_alpha_gt_1(self):
        DR = DelayedRejection()
        DR.iacce = np.zeros([1])
        self.assertEqual(DR.acceptance_test(alpha = 1.1, old_set = 1, next_set = 1, itry = 0)[0], 1, msg = 'accept alpha >= 1')
        
    def test_accept_alpha_eq_1(self):
        DR = DelayedRejection()
        DR.iacce = np.zeros([1])
        self.assertEqual(DR.acceptance_test(alpha = 1, old_set = 1, next_set = 1, itry = 0)[0], 1, msg = 'accept alpha >= 1')
        
    def test_not_accept_alpha_lt_0(self):
        DR = DelayedRejection()
        self.assertEqual(DR.acceptance_test(alpha = -0.1, old_set = 1, next_set = 1, itry = 0)[0], 0, msg = 'Reject alpha <= 0')
        
    def test_not_accept_alpha_eq_0(self):
        DR = DelayedRejection()
        self.assertEqual(DR.acceptance_test(alpha = 0, old_set = 1, next_set = 1, itry = 0)[0], 0, msg = 'Reject alpha <= 0')
        
    def test_not_accept_alpha_based_on_rand(self):
        DR = DelayedRejection()
        np.random.seed(0)
        self.assertEqual(DR.acceptance_test(alpha = 0.4, old_set = 1, next_set = 1, itry = 0)[0], 0, msg = 'Reject alpha < u (0.4 < 0.5488135)')
        
    def test_accept_alpha_based_on_rand(self):
        DR = DelayedRejection()
        DR.iacce = np.zeros([1])
        np.random.seed(0)
        self.assertEqual(DR.acceptance_test(alpha = 0.6, old_set = 1, next_set = 1, itry = 0)[0], 1, msg = 'Accept alpha > u (0.6 > 0.5488135)')
        
    def test_set_assignment_not_accept(self):
        DR = DelayedRejection()
        old_set = {'testvar1':'abc', 'testvar2':100.1}
        self.assertDictEqual(DR.acceptance_test(alpha = 0, old_set = old_set, next_set = 1, itry = 0)[1], old_set, msg = 'Reject alpha -> new_set = old_set')
        
    def test_set_assignment_accept(self):
        DR = DelayedRejection()
        DR.iacce = np.zeros([1])
        new_set = {'testvar1':'abc', 'testvar2':100.1}
        self.assertDictEqual(DR.acceptance_test(alpha = 1, old_set = 1, next_set = new_set, itry = 0)[1], new_set, msg = 'Accept alpha -> new_set = next_set')
        
# --------------------------        
class InitializeDRMetrics(unittest.TestCase):
    def test_dr_metrics(self):
        DR = DelayedRejection()
        model, options, parameters, data, covariance, rejected, chain, s2chain, sschain = setup_mcmc()
        DR._initialize_dr_metrics(options = options)
        self.assertTrue(np.array_equal(DR.iacce, np.zeros(options.ntry, dtype = int)), msg = 'Arrays should match')
        self.assertEqual(DR.dr_step_counter, 0, msg = 'Counter initialized to zero')
        
# --------------------------        
class InitializeNextMetropolisStep(unittest.TestCase):
    @patch('numpy.random.randn', return_value = np.array([0.2, 0.5]))
    def test_next_set(self, mock_1):
        DR = DelayedRejection()
        npar = 2
        old_theta = np.array([0.1, 0.2])
        RDR = np.array([[0.4, 0.2],[0, 0.3]])
        sigma2 = 0.24
        next_set = DR.initialize_next_metropolis_step(npar = npar, old_theta = old_theta, sigma2 = sigma2, RDR = RDR)
        self.assertEqual(next_set.sigma2, sigma2, msg = 'sigma2 should be 0.24')
        self.assertTrue(np.array_equal(next_set.theta, (old_theta + np.dot(np.array([0.2,0.5]),RDR)).reshape(npar)), msg = 'Arrays should match')
        
# --------------------------        
class SetOutsideBounds(unittest.TestCase):
    def test_set_outsidebounds(self):
        DR = DelayedRejection()
        CL = {'theta':1.0, 'ss': 1.0, 'prior':0.0, 'sigma2': 0.0}
        old_set = ParameterSet(theta = CL['theta'], ss = CL['ss'], prior = CL['prior'], sigma2 = CL['sigma2'])
        next_set = ParameterSet()
        out_set, next_set, trypath, outbound = DR._outside_bounds(old_set = old_set, next_set = next_set, trypath = [])
        self.assertEqual(trypath[0], next_set, msg = 'next_set should be element of trypath')
        self.assertEqual(outbound, 1, msg = 'outbound should be 1')
        self.assertEqual(out_set, old_set, msg = 'out_set should equal old_set since outside of bounds')
        