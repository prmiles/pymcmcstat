#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 05:12:38 2018

@author: prmiles
"""

from pymcmcstat.procedures.SumOfSquares import SumOfSquares
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
    nrow, ncol = xdata.shape
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
    
    mcstat.simulation_options.define_simulation_options(nsimu = int(5.0e3), updatesigma = 1, method = 'dram')
    
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
class InitializeSOS(unittest.TestCase):

    def test_init_sos(self):
        model, options, parameters, data = setup_mcmc()
        SOS = SumOfSquares(model = model, data = data, parameters = parameters)
        SOSD = SOS.__dict__
        check = {'nbatch': 1, 'parind': parameters._parind, 'value': parameters._value,
                 'local': parameters._local, 'data': data, 'model_function': model.model_function,
                 'sos_function': model.sos_function, 'sos_style': 1}
        items = ['nbatch', 'sos_style', 'sos_function', 'model_function', 'data']
        for ii, ai in enumerate(items):
            self.assertEqual(SOSD[ai], check[ai], msg = str('{}: {} != {}'.format(ai, SOSD[ai], check[ai])))
            
        array_items = ['parind', 'value', 'local']
        for ii, ai in enumerate(array_items):
            self.assertTrue(np.array_equal(SOSD[ai], check[ai]), msg = str('{}: {} != {}'.format(ai, SOSD[ai], check[ai])))
            
    def test_init_sos_function_is_none(self):
        model, options, parameters, data = setup_mcmc()
        model.sos_function = None
        model.model_function = modelfun
        SOS = SumOfSquares(model = model, data = data, parameters = parameters)
        SOSD = SOS.__dict__
        check = {'nbatch': 1, 'parind': parameters._parind, 'value': parameters._value,
                 'local': parameters._local, 'data': data, 'model_function': modelfun,
                 'sos_function': None, 'sos_style': 4}
        items = ['nbatch', 'sos_style', 'sos_function', 'model_function', 'data']
        for ii, ai in enumerate(items):
            self.assertEqual(SOSD[ai], check[ai], msg = str('{}: {} != {}'.format(ai, SOSD[ai], check[ai])))
            
        array_items = ['parind', 'value', 'local']
        for ii, ai in enumerate(array_items):
            self.assertTrue(np.array_equal(SOSD[ai], check[ai]), msg = str('{}: {} != {}'.format(ai, SOSD[ai], check[ai])))
        
    def test_init_sos_model_function_is_none(self):
        model, options, parameters, data = setup_mcmc()
        model.sos_function = None
        model.model_function = None
        with self.assertRaises(SystemExit):
            SumOfSquares(model = model, data = data, parameters = parameters)        
            
class EvaluateSOS(unittest.TestCase):

    def test_eval_sos(self):
        model, options, parameters, data = setup_mcmc()
        SOS = SumOfSquares(model = model, data = data, parameters = parameters)
        theta = np.array([2., 5.])
        ss = SOS.evaluate_sos_function(theta = theta)
        self.assertTrue(isinstance(ss, np.ndarray), msg = 'Expect numpy array return')
        self.assertEqual(ss.size, 1, msg = 'Size of array is 1')
        self.assertTrue(isinstance(ss[0], float), msg = 'Numerical result returned')
        self.assertTrue(np.array_equal(SOS.value[SOS.parind], theta), msg = 'Value(s) updated')
        
    def test_eval_sos_none(self):
        model, options, parameters, data = setup_mcmc()
        model.sos_function = None
        model.model_function = modelfun
        SOS = SumOfSquares(model = model, data = data, parameters = parameters)
        theta = np.array([2., 5.])
        ss = SOS.evaluate_sos_function(theta = theta)
        self.assertTrue(isinstance(ss, np.ndarray), msg = 'Expect numpy array return')
        self.assertEqual(ss.size, 1, msg = 'Size of array is 1')
        self.assertTrue(isinstance(ss[0], float), msg = 'Numerical result returned')
        self.assertTrue(np.array_equal(SOS.value[SOS.parind], theta), msg = 'Value(s) updated')
        
    def test_eval_sos_model_none(self):
        model, options, parameters, data = setup_mcmc()
        SOS = SumOfSquares(model = model, data = data, parameters = parameters)
        SOS.sos_style = 2
        theta = np.array([2., 5.])
        ss = SOS.evaluate_sos_function(theta = theta)
        self.assertTrue(isinstance(ss, np.ndarray), msg = 'Expect numpy array return')
        self.assertEqual(ss.size, 1, msg = 'Size of array is 1')
        self.assertTrue(isinstance(ss[0], float), msg = 'Numerical result returned')
        self.assertTrue(np.array_equal(SOS.value[SOS.parind], theta), msg = 'Value(s) updated')
#        for (k,v) in SOSD.items():
#            self.assertEqual(v, defaults[k], msg = str('Default {} is {}'.format(k, defaults[k])))
            
#    def test_PS_defaul_priorfun(self):
#        key = 'priorfun'
#        PF = PriorFunction()
#        PFD = PF.__dict__
#        self.assertEqual(PFD[key], PF.default_priorfun, msg = str('Expected {} = {}'.format(key, 'PriorFunction.default_priorfun')))
#        
#class Evaluation_Default_Prior_Function(unittest.TestCase):
#    
#    def test_PF_evaluation_with_single_set(self):
#        PF = PriorFunction()
#        theta = [0.]
#        mu = [0.]
#        sigma = [0.1]
#        self.assertEqual(PF.default_priorfun(theta = theta, mu = mu, sigma=sigma), 0, msg = 'Prior should be 0')
#        
#    def test_PF_evaluation_with_double_set(self):
#        PF = PriorFunction()
#        theta = [0., 0.]
#        mu = [0., 0.]
#        sigma = [0.1, 0.1]
#        self.assertEqual(PF.default_priorfun(theta = theta, mu = mu, sigma=sigma), 0, msg = 'Prior should be 0')
#        
#    def test_PF_evaluation_with_single_set_and_nonzero_answer(self):
#        PF = PriorFunction()
#        theta = [1.]
#        mu = [45.]
#        sigma = [0.1]
#        self.assertTrue(PF.default_priorfun(theta = theta, mu = mu, sigma=sigma) > 0, msg = 'Prior should be >=0')
#        
#    def test_PF_evaluation_with_double_set_and_nonzero_answer(self):
#        PF = PriorFunction()
#        theta = [1., 91.]
#        mu = [45., 27.]
#        sigma = [0.1, 16.]
#        self.assertTrue(PF.default_priorfun(theta = theta, mu = mu, sigma=sigma) > 0, msg = 'Prior should be >=0')
        
#class EvaluatePriorFunction(unittest.TestCase):
#    
#    def test_EPF_with_single_set(self):
#        PF = PriorFunction()
#        theta = [0.]
#        self.assertEqual(PF.evaluate_prior(theta = theta), 0, msg = 'Prior should be 0')
#        
#    def test_EPF_with_double_set(self):
#        PF = PriorFunction()
#        theta = [0., 0.]
#        self.assertEqual(PF.evaluate_prior(theta = theta), 0, msg = 'Prior should be 0')
#        
#    def test_EPF_with_single_set_and_nonzero_answer(self):
#        PF = PriorFunction()
#        theta = [1.]
#        self.assertTrue(PF.evaluate_prior(theta = theta) > 0, msg = 'Prior should be >=0')