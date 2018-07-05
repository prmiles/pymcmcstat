#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 05:12:38 2018

@author: prmiles
"""

from pymcmcstat.procedures.SumOfSquares import SumOfSquares
import test.general_functions as gf
import unittest
import numpy as np

# --------------------------
class InitializeSOS(unittest.TestCase):

    def test_init_sos(self):
        model, __, parameters, data = gf.setup_mcmc()
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
        model, options, parameters, data = gf.setup_mcmc()
        model.sos_function = None
        model.model_function = gf.modelfun
        SOS = SumOfSquares(model = model, data = data, parameters = parameters)
        SOSD = SOS.__dict__
        check = {'nbatch': 1, 'parind': parameters._parind, 'value': parameters._value,
                 'local': parameters._local, 'data': data, 'model_function': gf.modelfun,
                 'sos_function': None, 'sos_style': 4}
        items = ['nbatch', 'sos_style', 'sos_function', 'model_function', 'data']
        for ii, ai in enumerate(items):
            self.assertEqual(SOSD[ai], check[ai], msg = str('{}: {} != {}'.format(ai, SOSD[ai], check[ai])))
            
        array_items = ['parind', 'value', 'local']
        for ii, ai in enumerate(array_items):
            self.assertTrue(np.array_equal(SOSD[ai], check[ai]), msg = str('{}: {} != {}'.format(ai, SOSD[ai], check[ai])))
        
    def test_init_sos_model_function_is_none(self):
        model, options, parameters, data = gf.setup_mcmc()
        model.sos_function = None
        model.model_function = None
        with self.assertRaises(SystemExit):
            SumOfSquares(model = model, data = data, parameters = parameters)
            
class EvaluateSOS(unittest.TestCase):

    def test_eval_sos(self):
        model, options, parameters, data = gf.setup_mcmc()
        SOS = SumOfSquares(model = model, data = data, parameters = parameters)
        theta = np.array([2., 5.])
        ss = SOS.evaluate_sos_function(theta = theta)
        self.assertTrue(isinstance(ss, np.ndarray), msg = 'Expect numpy array return')
        self.assertEqual(ss.size, 1, msg = 'Size of array is 1')
        self.assertTrue(isinstance(ss[0], float), msg = 'Numerical result returned')
        self.assertTrue(np.array_equal(SOS.value[SOS.parind], theta), msg = 'Value(s) updated')
        
    def test_eval_sos_none(self):
        model, options, parameters, data = gf.setup_mcmc()
        model.sos_function = None
        model.model_function = gf.modelfun
        SOS = SumOfSquares(model = model, data = data, parameters = parameters)
        theta = np.array([2., 5.])
        ss = SOS.evaluate_sos_function(theta = theta)
        self.assertTrue(isinstance(ss, np.ndarray), msg = 'Expect numpy array return')
        self.assertEqual(ss.size, 1, msg = 'Size of array is 1')
        self.assertTrue(isinstance(ss[0], float), msg = 'Numerical result returned')
        self.assertTrue(np.array_equal(SOS.value[SOS.parind], theta), msg = 'Value(s) updated')
        
    def test_eval_sos_model_none(self):
        model, options, parameters, data = gf.setup_mcmc()
        SOS = SumOfSquares(model = model, data = data, parameters = parameters)
        SOS.sos_style = 2
        theta = np.array([2., 5.])
        ss = SOS.evaluate_sos_function(theta = theta)
        self.assertTrue(isinstance(ss, np.ndarray), msg = 'Expect numpy array return')
        self.assertEqual(ss.size, 1, msg = 'Size of array is 1')
        self.assertTrue(isinstance(ss[0], float), msg = 'Numerical result returned')
        self.assertTrue(np.array_equal(SOS.value[SOS.parind], theta), msg = 'Value(s) updated')