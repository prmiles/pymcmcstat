#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 08:48:00 2018

@author: prmiles
"""

from pymcmcstat.samplers.DelayedRejection import update_set_based_on_acceptance
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
        
# -------------------------------------------
class UpdateSetBasedOnAcceptance(unittest.TestCase):
    def test_set_based_on_accept_false(self):
        next_set = ParameterSet(theta = np.random.random_sample(size = (2,1)), ss = 0.4)
        old_set = ParameterSet(theta = np.random.random_sample(size = (2,1)), ss = 0.6)
        out_set = update_set_based_on_acceptance(accept = False, old_set = old_set, next_set = next_set)
        self.assertEqual(out_set, old_set)
        
    def test_set_based_on_accept_true(self):
        next_set = ParameterSet(theta = np.random.random_sample(size = (2,1)), ss = 0.4)
        old_set = ParameterSet(theta = np.random.random_sample(size = (2,1)), ss = 0.6)
        out_set = update_set_based_on_acceptance(accept = True, old_set = old_set, next_set = next_set)
        self.assertEqual(out_set, next_set)