# -*- coding: utf-8 -*-

"""
Created on Mon Jan. 15, 2018

Description: This file contains a series of utilities designed to test the
features in the 'generalfunctions.py" package of the pymcmcstat module.  The 
functions tested include:
    - less_than_or_equal_to_zero(x):
    - replace_list_elements(x, testfunction, value):
    - message(verbosity, level, printthis):
    - is_semi_pos_def_chol(x):
    - print_mod(string, value, flag):
    - nordf(x, mu = 0, sigma2 = 1):
    - empirical_quantiles(x, p = np.array([0.25, 0.5, 0.75])):

@author: prmiles
"""
from pymcmcstat.generalfunctions import less_than_or_equal_to_zero, replace_list_elements
from pymcmcstat.generalfunctions import message, is_semi_pos_def_chol, print_mod, nordf, empirical_quantiles
import unittest
import numpy as np

# --------------------------
# less_than_or_equal_to_zero       
# --------------------------
class Less_Than_Or_Equal_To_Zero_Test(unittest.TestCase):
    
    def test_is_negative_one_less_than_or_equal_to_zero(self):
        self.assertTrue(less_than_or_equal_to_zero(-1), msg = '-1 is less than or equal to 0')    
        
    def test_is_zero_less_than_or_equal_to_zero(self):
        self.assertTrue(less_than_or_equal_to_zero(0), msg = '0 is less than or equal to 0')
        
    def test_is_one_less_than_or_equal_to_zero(self):
        self.assertFalse(less_than_or_equal_to_zero(1), msg = '1 is not less than or equal to 0')

# --------------------------
# replace_list_elements        
# --------------------------
class Replace_List_Elements_Test(unittest.TestCase):
    
    def test_no_replace_list_elements(self):
        self.assertEqual(replace_list_elements(np.array([1, 2, 3], dtype=float), less_than_or_equal_to_zero, np.Inf).all(), np.array([1,2,3]).all(), msg = 'All elements greater than 0')
    
    def test_replace_list_elements(self):
        self.assertEqual(replace_list_elements(np.array([1, -2, 3], dtype=float), less_than_or_equal_to_zero, np.Inf).all(), np.array([1,np.Inf,3]).all(), msg = '2nd element switched to np.Inf')

# --------------------------
# message        
# --------------------------
class MessageTest(unittest.TestCase):
    
    def test_does_message_not_print(self):
        self.assertFalse(message(0, 1, 'Message Not Printed'), msg = 'Should not print')
        
    def test_does_message_print(self):
        self.assertTrue(message(1, 0, 'Message Printed'), msg = 'Should print')

# --------------------------
# is_semi_pos_def_chol
# --------------------------
class Is_Semi_Pos_Def_Chol_Test(unittest.TestCase): 
       
    def test_is_semi_pos_def_chol(self):
        flag, c = is_semi_pos_def_chol(np.array([[2, 1],[1, 2]]))
        self.assertTrue(flag, msg = 'flag is True')
        self.assertIsInstance(c, np.ndarray, msg = 'c is matrix')
        
    def test_is_not_semi_pos_def_chol(self):
        flag, c = is_semi_pos_def_chol(np.array([[1, 2],[2, 1]]))
        self.assertFalse(flag, msg = 'flag is False')
        self.assertIsNone(c, msg = 'c is None')
        
# --------------------------
# print_mod       
# --------------------------
class Print_Mod_Test(unittest.TestCase):
    
    def test_does_print_mod_not_print(self):
        self.assertFalse(print_mod('Variable = ', 0, 0), msg = 'Should not print')
        
    def test_does_print_mod_print(self):
        self.assertTrue(print_mod('Variable = ', 0, 1), msg = 'Should print')
        
# --------------------------
# nordf      
# --------------------------
class Nordf_Test(unittest.TestCase):

    def test_is_default_nordf_at_zero_one_half(self):
        self.assertEqual(nordf(0), 0.5, msg = 'Default at 0 is 0.5')
        
    def test_does_nordf_return_float_for_scalar_input(self):
        self.assertIsInstance(nordf(0), float)
        
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