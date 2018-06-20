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

    mcstat.simulation_options.define_simulation_options(nsimu = int(5.0e3), updatesigma = 1, method = 'dram', verbosity = 0)
    
    # update model settings
    mcstat.model_settings.define_model_settings(sos_function = ssfun)
    
    mcstat.parameters.add_model_parameter(name = 'm', theta0 = 2., minimum = -10, maximum = np.inf, sample = 1)
    mcstat.parameters.add_model_parameter(name = 'b', theta0 = -5., minimum = -10, maximum = 100, sample = 0)
    mcstat.parameters.add_model_parameter(name = 'b2', theta0 = -5., minimum = -10, maximum = 100, sample = 1)
    
    mcstat._initialize_simulation()
    
    # extract components
    sos_object = mcstat._MCMC__sos_object
    prior_object = mcstat._MCMC__prior_object
    parameters = mcstat.parameters
    return sos_object, prior_object, parameters

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
# --------------------------        
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
# --------------------------        
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
        
# --------------------------
class RunMetropolisStep(unittest.TestCase):
    @patch('pymcmcstat.samplers.Metropolis.Metropolis.is_sample_outside_bounds')
    def test_run_step_outside_bounds(self, mock_simple_func):
        mock_simple_func.return_value = True
        sos_object, prior_object, parameters = setup_mcmc()
        MS = Metropolis()
        R = np.array([[0.4, 0.2],[0, 0.3]])
        CL = {'theta':1.0, 'ss': 1.0, 'prior':0.0, 'sigma2': 0.0}
        parset = ParameterSet(theta = CL['theta'], ss = CL['ss'], prior = CL['prior'], sigma2 = CL['sigma2'])
        accept, newset, outbound, npar_sample_from_normal = MS.run_metropolis_step(old_set = parset, parameters = parameters, R = R, prior_object = prior_object, sos_object = sos_object)
        self.assertEqual(accept, 0, msg = 'Not accepted because outside bounds')
        self.assertEqual(outbound, 1, msg = 'outbound set to 1')
        
    @patch('pymcmcstat.samplers.Metropolis.Metropolis.is_sample_outside_bounds', return_value = False)
    @patch('pymcmcstat.samplers.Metropolis.Metropolis.evaluate_likelihood_function', return_value = 1.1)
    def test_run_step_inside_bounds(self, mock_1, mock_2):
        sos_object, prior_object, parameters = setup_mcmc()
        MS = Metropolis()
        R = np.array([[0.4, 0.2],[0, 0.3]])
        CL = {'theta':1.0, 'ss': 1.0, 'prior':0.0, 'sigma2': 0.0}
        parset = ParameterSet(theta = CL['theta'], ss = CL['ss'], prior = CL['prior'], sigma2 = CL['sigma2'])
        accept, newset, outbound, npar_sample_from_normal = MS.run_metropolis_step(old_set = parset, parameters = parameters, R = R, prior_object = prior_object, sos_object = sos_object)
        self.assertEqual(outbound, 0, msg = 'outbound set to 0')
        self.assertEqual(accept, 1, msg = 'Accepted because likelihood > 1')
        
    @patch('pymcmcstat.samplers.Metropolis.Metropolis.is_sample_outside_bounds', return_value = False)
    @patch('pymcmcstat.samplers.Metropolis.Metropolis.evaluate_likelihood_function', return_value = 0.5)
    @patch('numpy.random.rand', return_value = 0.4)
    def test_run_step_inside_bounds_test_accept(self, mock_1, mock_2, mock_3):
        sos_object, prior_object, parameters = setup_mcmc()
        MS = Metropolis()
        R = np.array([[0.4, 0.2],[0, 0.3]])
        CL = {'theta':1.0, 'ss': 1.0, 'prior':0.0, 'sigma2': 0.0}
        parset = ParameterSet(theta = CL['theta'], ss = CL['ss'], prior = CL['prior'], sigma2 = CL['sigma2'])
        accept, newset, outbound, npar_sample_from_normal = MS.run_metropolis_step(old_set = parset, parameters = parameters, R = R, prior_object = prior_object, sos_object = sos_object)
        self.assertEqual(outbound, 0, msg = 'outbound set to 0')
        self.assertEqual(accept, 1, msg = 'Accepted because 0.5 > 0.4')
        
    @patch('pymcmcstat.samplers.Metropolis.Metropolis.is_sample_outside_bounds', return_value = False)
    @patch('pymcmcstat.samplers.Metropolis.Metropolis.evaluate_likelihood_function', return_value = 0.3)
    @patch('numpy.random.rand', return_value = 0.4)
    def test_run_step_inside_bounds_test_accept_fail(self, mock_1, mock_2, mock_3):
        sos_object, prior_object, parameters = setup_mcmc()
        MS = Metropolis()
        R = np.array([[0.4, 0.2],[0, 0.3]])
        CL = {'theta':1.0, 'ss': 1.0, 'prior':0.0, 'sigma2': 0.0}
        parset = ParameterSet(theta = CL['theta'], ss = CL['ss'], prior = CL['prior'], sigma2 = CL['sigma2'])
        accept, newset, outbound, npar_sample_from_normal = MS.run_metropolis_step(old_set = parset, parameters = parameters, R = R, prior_object = prior_object, sos_object = sos_object)
        self.assertEqual(outbound, 0, msg = 'outbound set to 0')
        self.assertEqual(accept, 0, msg = 'Accepted because 0.3 < 0.4')
        
# --------------------------
class SampleCandidate(unittest.TestCase):
    @patch('numpy.random.randn')
    def test_sample_candidate(self, mock_simple_func):
        mock_simple_func.return_value = np.array([0.1, 0.2])
        MS = Metropolis()
        oldpar = np.array([0.1, 0.4])
        R = np.array([[0.4, 0.2],[0, 0.3]])
        newpar, npar_sample_from_normal = MS.sample_candidate_from_gaussian_proposal(npar = 2, oldpar = oldpar, R = R)
        self.assertEqual(newpar.size, 2, msg = 'Size of parameter array is 2')
        self.assertEqual(npar_sample_from_normal.size, 2, msg = 'Size of sample is 2')
        self.assertTrue(np.array_equal(newpar, (oldpar + np.dot(np.array([0.1, 0.2]), R)).reshape(2)), msg = 'Arrays should match')

# --------------------------
class ValuesOutsideBounds(unittest.TestCase):
    def test_values_ob(self):
        MS = Metropolis()
        ss = 0.24
        accept, newprior, alpha, ss1, ss2, outbound = MS.values_for_outsidebounds(ss = ss)
        self.assertEqual(accept, 0, msg = 'Do not accept -> 0')
        self.assertEqual(newprior, 0, msg = 'New prior is 0')
        self.assertEqual(alpha, 0, msg = 'Alpha set to 0')
        self.assertEqual(ss1, np.inf, msg = 'ss1 set to np.inf')
        self.assertEqual(ss2, ss, msg = 'ss2 set to ss')
        self.assertEqual(outbound, 1, msg = 'outbound set to 1')      