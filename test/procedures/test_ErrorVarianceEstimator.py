#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 05:12:38 2018

@author: prmiles
"""

from pymcmcstat.procedures.SumOfSquares import SumOfSquares
from pymcmcstat.procedures.ErrorVarianceEstimator import ErrorVarianceEstimator
import test.general_functions as gf
import unittest
import numpy as np

# --------------------------
class InitializeEVE(unittest.TestCase):

    def test_init_EVE(self):
        EVE = ErrorVarianceEstimator()
        self.assertTrue(hasattr(EVE, 'description'))
        
# --------------------------
class UpdateEVE(unittest.TestCase):
    
    def test_eve_update_with_nsos_1(self):
        model, __, parameters, data = gf.setup_mcmc()
        SOS = SumOfSquares(model = model, data = data, parameters = parameters)
        theta = np.array([2., 5.])
        ss = SOS.evaluate_sos_function(theta = theta)
        EVE = ErrorVarianceEstimator()
        sigma2 = EVE.update_error_variance(sos = ss, model = model)
        self.assertTrue(isinstance(sigma2, np.ndarray), msg = 'Expect array return.')
        self.assertEqual(sigma2.size, 1, msg = 'Size of array is 1')
        self.assertTrue(isinstance(sigma2[0], float), msg = 'Numerical result returned')
        
    def test_eve_update_with_nsos_2(self):
        model, options, parameters, data = gf.setup_mcmc()
        nsos = 2
        model._check_dependent_model_settings_wrt_nsos(nsos = nsos)
        ss = np.array([0.1, 0.2])
        EVE = ErrorVarianceEstimator()
        sigma2 = EVE.update_error_variance(sos = ss, model = model)
        self.assertTrue(isinstance(sigma2, np.ndarray), msg = 'Expect array return.')
        self.assertEqual(sigma2.size, 2, msg = 'Size of array is 2')
        self.assertTrue(isinstance(sigma2[0], float), msg = 'Numerical result returned')
        
# --------------------------
class Gammar(unittest.TestCase):
    
    def test_gammar_for_a_0(self):
        EVE = ErrorVarianceEstimator()
        m = 4
        n = 3
        ret = EVE.gammar(m = m, n = 3, a = 0)
        self.assertTrue(np.array_equal(ret, np.zeros([m,n])), msg = 'Expect equal array return.')
        
    def test_gammar_for_a_lt_0(self):
        EVE = ErrorVarianceEstimator()
        m = 4
        n = 3
        ret = EVE.gammar(m = m, n = 3, a = -10.)
        self.assertTrue(np.array_equal(ret, np.zeros([m,n])), msg = 'Expect equal array return.')
        
# --------------------------
class Gammar_MT1(unittest.TestCase):
    
    def test_gammar_for_a_lt_1(self):
        EVE = ErrorVarianceEstimator()
        y = EVE._gammar_mt1(a = 0.1, b = 1)
        self.assertTrue(isinstance(y, np.ndarray), msg = 'Expect equal array return.')
        
#    def test_gammar_for_a_lt_0(self):
#        EVE = ErrorVarianceEstimator()
#        m = 4
#        n = 3
#        ret = EVE.gammar_mt1(a = -10.)
#        self.assertTrue(np.array_equal(ret, np.zeros([m,n])), msg = 'Expect equal array return.')