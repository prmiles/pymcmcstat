# -*- coding: utf-8 -*-

"""
Created on Mon Jan. 15, 2018

Description: This file contains a series of utilities designed to test the
features in the 'PriorFunction.py" class of the pymcmcstat module.

@author: prmiles
"""
from pymcmcstat.procedures.PriorFunction import PriorFunction
import unittest
import test.general_functions as gf

# --------------------------
# PriorFunction
# --------------------------
class Initialize_Prior_Function(unittest.TestCase):

    def test_PF_default_mu_sigma(self):
        PF = PriorFunction()
        PFD = PF.__dict__
        PFD = gf.removekey(PFD, 'priorfun')
        defaults = {'mu': None, 'sigma': None}
        for (k,v) in PFD.items():
            self.assertEqual(v, defaults[k], msg = str('Default {} is {}'.format(k, defaults[k])))

    def test_PS_defaul_priorfun(self):
        key = 'priorfun'
        PF = PriorFunction()
        PFD = PF.__dict__
        self.assertEqual(PFD[key], PF.default_priorfun, msg = str('Expected {} = {}'.format(key, 'PriorFunction.default_priorfun')))
        
class Evaluation_Default_Prior_Function(unittest.TestCase):
    
    def test_PF_evaluation_with_single_set(self):
        PF = PriorFunction()
        theta = [0.]
        mu = [0.]
        sigma = [0.1]
        self.assertEqual(PF.default_priorfun(theta = theta, mu = mu, sigma=sigma), 0, msg = 'Prior should be 0')
        
    def test_PF_evaluation_with_double_set(self):
        PF = PriorFunction()
        theta = [0., 0.]
        mu = [0., 0.]
        sigma = [0.1, 0.1]
        self.assertEqual(PF.default_priorfun(theta = theta, mu = mu, sigma=sigma), 0, msg = 'Prior should be 0')
        
    def test_PF_evaluation_with_single_set_and_nonzero_answer(self):
        PF = PriorFunction()
        theta = [1.]
        mu = [45.]
        sigma = [0.1]
        self.assertTrue(PF.default_priorfun(theta = theta, mu = mu, sigma=sigma) > 0, msg = 'Prior should be >=0')
        
    def test_PF_evaluation_with_double_set_and_nonzero_answer(self):
        PF = PriorFunction()
        theta = [1., 91.]
        mu = [45., 27.]
        sigma = [0.1, 16.]
        self.assertTrue(PF.default_priorfun(theta = theta, mu = mu, sigma=sigma) > 0, msg = 'Prior should be >=0')
        
class EvaluatePriorFunction(unittest.TestCase):
    
    def test_EPF_with_single_set(self):
        PF = PriorFunction()
        theta = [0.]
        self.assertEqual(PF.evaluate_prior(theta = theta), 0, msg = 'Prior should be 0')
        
    def test_EPF_with_double_set(self):
        PF = PriorFunction()
        theta = [0., 0.]
        self.assertEqual(PF.evaluate_prior(theta = theta), 0, msg = 'Prior should be 0')
        
    def test_EPF_with_single_set_and_nonzero_answer(self):
        PF = PriorFunction()
        theta = [1.]
        self.assertTrue(PF.evaluate_prior(theta = theta) > 0, msg = 'Prior should be >=0')