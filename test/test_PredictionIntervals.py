# -*- coding: utf-8 -*-

"""
Created on Mon Jan. 15, 2018

Description: This file contains a series of utilities designed to test the
features in the 'PredictionIntervals.py" package of the pymcmcstat module.  The 
functions tested include:
    - empirical_quantiles(x, p = np.array([0.25, 0.5, 0.75])):

@author: prmiles
"""
from pymcmcstat.plotting.PredictionIntervals import PredictionIntervals
import unittest
import numpy as np

PI = PredictionIntervals()
empirical_quantiles = PI._empirical_quantiles
observation_sample = PI._observation_sample
analyze_s2chain = PI._analyze_s2chain

# --------------------------
# empirical_quantiles 
# --------------------------
class Empirical_Quantiles_Test(unittest.TestCase):

    def test_does_default_empirical_quantiles_return_3_element_array(self):
        test_out = empirical_quantiles(np.random.rand(10,1))
        self.assertEqual(test_out.shape, (3,1), msg = 'Default output shape is (3,1)')
        
    def test_does_non_default_empirical_quantiles_return_2_element_array(self):
        test_out = empirical_quantiles(np.random.rand(10,1), p = np.array([0.2, 0.5]))
        self.assertEqual(test_out.shape, (2,1), msg = 'Non-default output shape should be (2,1)')
        
    def test_empirical_quantiles_should_not_support_list_input(self):
#        test_out = empirical_quantiles(np.random.rand(10,1))
        with self.assertRaises(AttributeError):
#            empirical_quantiles(test_out)
            empirical_quantiles([-1,0,1])
            
    def test_empirical_quantiles_vector(self):
        out = empirical_quantiles(np.linspace(10,20, num = 10).reshape(10,1), p = np.array([0.22, 0.57345]))
        exact = np.array([[12.2], [15.7345]])
        comp = np.linalg.norm(out - exact)
        self.assertAlmostEqual(comp, 0)
        
## --------------------------
## analyze_s2chain
## --------------------------
#class Analyze_S2chain_Test(unittest.TestCase):
#
#    def test_does_analyze_s2chain_return_3_element_array(self):
#        test_out = analyze_s2chain(np.random.rand(10,1))
#        self.assertEqual(test_out.shape, (3,1), msg = 'Default output shape is (3,1)')
        
            
class Observation_Sample_Test(unittest.TestCase):
    
    def test_does_observation_sample_unknown_sstype_cause_system_exit(self):
        s2elem = np.array([[2.0]])
        ypred = np.linspace(2.0, 3.0, num = 5)
        ypred = ypred.reshape(5,1)
        sstype = 4
        with self.assertRaises(SystemExit):
            observation_sample(s2elem, ypred, sstype)
            
    def test_does_observation_sample_return_right_size_array(self):
        s2elem = np.array([[2.0]])
        ypred = np.linspace(2.0, 3.0, num = 5)
        ypred = ypred.reshape(5,1)
        sstype = 0
        opred = observation_sample(s2elem, ypred, sstype)
        self.assertEqual(opred.shape, ypred.shape)
        
    def test_does_observation_sample_wrong_size_s2elem_break_right_size_array(self):
        s2elem = np.array([[2.0, 1.0]])
        ypred = np.linspace(2.0, 3.0, num = 5)
        ypred = ypred.reshape(5,1)
        sstype = 0
        with self.assertRaises(SystemExit):
            observation_sample(s2elem, ypred, sstype)
            
    def test_does_observation_sample_2_column_ypred_right_size_array(self):
        s2elem = np.array([[2.0, 1.0]])
        ypred = np.linspace(2.0, 3.0, num = 10)
        ypred = ypred.reshape(5,2)
        sstype = 0
        opred = observation_sample(s2elem, ypred, sstype)
        self.assertEqual(opred.shape, ypred.shape)
        
    def test_does_observation_sample_2_column_ypred_with_1_s2elem_right_size_array(self):
        s2elem = np.array([[2.0]])
        ypred = np.linspace(2.0, 3.0, num = 10)
        ypred = ypred.reshape(5,2)
        sstype = 0
        opred = observation_sample(s2elem, ypred, sstype)
        self.assertEqual(opred.shape, ypred.shape)
        
    def test_does_observation_sample_off_s2elem_greater_than_1_cause_system_exit(self):
        s2elem = np.array([[2.0, 1.0]])
        ypred = np.linspace(2.0, 3.0, num = 15)
        ypred = ypred.reshape(5,3)
        sstype = 0
        with self.assertRaises(SystemExit):
            observation_sample(s2elem, ypred, sstype)