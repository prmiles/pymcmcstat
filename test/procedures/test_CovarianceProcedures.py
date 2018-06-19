#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 06:55:18 2018

@author: prmiles
"""

from pymcmcstat.procedures.CovarianceProcedures import CovarianceProcedures
from pymcmcstat.MCMC import MCMC
import unittest
import numpy as np

def removekey(d, key):
        r = dict(d)
        del r[key]
        return r

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
    model = mcstat.model_settings
    options = mcstat.simulation_options
    parameters = mcstat.parameters
    data = mcstat.data
    return model, options, parameters, data

# --------------------------
class InitializeCP(unittest.TestCase):

    def test_init_CP(self):
        CP = CovarianceProcedures()
        self.assertTrue(hasattr(CP, 'description'))
        
# --------------------------
class UpdateCovarianceFromAdaptation(unittest.TestCase):

    def test_update_cov(self):
        model, options, parameters, data = setup_mcmc()
        CP = CovarianceProcedures()
        CP._initialize_covariance_settings(parameters = parameters, options = options)
        
        check = {
        'R': np.random.random_sample(size = (2,2)),
        'covchain': np.random.random_sample(size = (2,2)),
        'meanchain': np.random.random_sample(size = (1,2)),
        'wsum': np.random.random_sample(size = (2,1)),
        'last_index_since_adaptation': 0,
        'iiadapt': 100
        }
        
        CP._update_covariance_from_adaptation(**check)
        CPD = CP.__dict__
        items = ['last_index_since_adaptation', 'iiadapt']
        for ii, ai in enumerate(items):
            self.assertEqual(CPD[str('_{}'.format(ai))], check[ai], msg = str('{}: {} != {}'.format(ai, CPD[str('_{}'.format(ai))], check[ai])))
            
        array_items = ['R', 'covchain', 'meanchain', 'wsum']
        for ii, ai in enumerate(array_items):
            self.assertTrue(np.array_equal(CPD[str('_{}'.format(ai))], check[ai]), msg = str('{}: {} != {}'.format(ai, CPD[str('_{}'.format(ai))], check[ai])))
            
# --------------------------
class UpdateCovarianceFromDelayedRejection(unittest.TestCase):

    def test_update_cov(self):
        model, options, parameters, data = setup_mcmc()
        CP = CovarianceProcedures()
        CP._initialize_covariance_settings(parameters = parameters, options = options)
        
        check = {
        'RDR': np.random.random_sample(size = (2,2)),
        'invR': np.random.random_sample(size = (2,2)),
        }
        
        CP._update_covariance_for_delayed_rejection_from_adaptation(**check)
        CPD = CP.__dict__
        array_items = ['RDR', 'invR']
        for ii, ai in enumerate(array_items):
            self.assertTrue(np.array_equal(CPD[str('_{}'.format(ai))], check[ai]), msg = str('{}: {} != {}'.format(ai, CPD[str('_{}'.format(ai))], check[ai])))
            
# --------------------------
class UpdateCovarianceSettings(unittest.TestCase):

    def test_update_cov_wsum_none(self):
        model, options, parameters, data = setup_mcmc()
        CP = CovarianceProcedures()
        CP._initialize_covariance_settings(parameters = parameters, options = options)
        
        theta = np.array([2., 5.])
        CP._wsum = None
        CP._covchain = 1
        CP._meanchain = 2
        CP._qcov = 0
        CP._update_covariance_settings(parameter_set = theta)
        
        CPD = CP.__dict__
        self.assertEqual(CPD['_covchain'], 1, msg = '_covchain unchanged.')
        self.assertEqual(CPD['_meanchain'], 2, msg = '_meanchain unchanged.')
        
    def test_update_cov_wsum_not_none(self):
        model, options, parameters, data = setup_mcmc()
        CP = CovarianceProcedures()
        CP._initialize_covariance_settings(parameters = parameters, options = options)
        
        theta = np.array([2., 5.])
        CP._wsum = 10
        CP._covchain = 1
        CP._meanchain = 2
        CP._qcov = 0
        CP._update_covariance_settings(parameter_set = theta)
        
        CPD = CP.__dict__
        self.assertEqual(CPD['_covchain'], 0, msg = '_covchain = _qcov.')
        self.assertTrue(np.array_equal(CPD['_meanchain'], theta), msg = '_meanchain = parameter_set.')
        
# -------------------------------------------
class DisplayCovarianceSettings(unittest.TestCase):
    
    def test_print_these_none(self):
        model, options, parameters, data = setup_mcmc()
        CP = CovarianceProcedures()
        CP._initialize_covariance_settings(parameters = parameters, options = options)
        
        print_these = CP.display_covariance_settings(print_these = None)
        self.assertEqual(print_these, ['qcov', 'R', 'RDR', 'invR', 'last_index_since_adaptation', 'covchain'], msg = 'Default print keys')
        
    def test_print_these_not_none(self):
        model, options, parameters, data = setup_mcmc()
        CP = CovarianceProcedures()
        CP._initialize_covariance_settings(parameters = parameters, options = options)
        
        print_these = ['qcov', 'R', 'RDR', 'invR', 'last_index_since_adaptation', 'covchain']
        for ii, ptii in enumerate(print_these):
            self.assertEqual(CP.display_covariance_settings(print_these=[ptii]), [ptii], msg = 'Specified print keys')
# -------------------------------------------