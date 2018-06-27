#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 08:48:00 2018

@author: prmiles
"""

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
        
    def test_alpha_value(self):
        MA = Metropolis()
        ss1 = np.array([1., 2.])
        ss2 = np.array([1.1, 2.4])
        newprior = np.array([0.,0.])
        oldprior = np.array([0.,0.])
        sigma2 = np.array([1.,1.])
        alpha1 = MA.evaluate_likelihood_function(ss1, ss2, sigma2, newprior, oldprior)
        self.assertTrue(np.allclose(alpha1, np.array([2.568050833375483])), msg = str('alpha = {}'.format(alpha1)))
               
# --------------------------
class RunMetropolisStep(unittest.TestCase):
    @patch('pymcmcstat.samplers.Metropolis.is_sample_outside_bounds', return_value = True)
    def test_run_step_outside_bounds(self, mock_1):
        sos_object, prior_object, parameters = setup_mcmc()
        MS = Metropolis()
        R = np.array([[0.4, 0.2],[0, 0.3]])
        CL = {'theta':1.0, 'ss': 1.0, 'prior': 0.0, 'sigma2': 1.0}
        parset = ParameterSet(theta = CL['theta'], ss = CL['ss'], prior = CL['prior'], sigma2 = CL['sigma2'])
        accept, newset, outbound, npar_sample_from_normal = MS.run_metropolis_step(old_set = parset, parameters = parameters, R = R, prior_object = prior_object, sos_object = sos_object)
        self.assertEqual(outbound, 1, msg = 'outbound set to 1')
        self.assertEqual(accept, 0, msg = 'Not accepted because outside bounds')
        
    @patch('pymcmcstat.samplers.Metropolis.is_sample_outside_bounds', return_value = False)
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
        
    @patch('pymcmcstat.samplers.Metropolis.is_sample_outside_bounds', return_value = False)
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
        
    @patch('pymcmcstat.samplers.Metropolis.is_sample_outside_bounds', return_value = False)
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