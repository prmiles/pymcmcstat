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
from pymcmcstat.generalfunctions import less_than_or_equal_to_zero, replace_list_elements, message, is_semi_pos_def_chol
import unittest
import numpy as np

# --------------------------
# less_than_or_equal_to_zero       
# --------------------------
class Less_Than_Or_Equal_To_Zero_Test(unittest.TestCase):
    
    def test_is_negative_one_less_than_or_equal_to_zero(self):
        self.assertTrue(less_than_or_equal_to_zero(-1), '-1 is less than or equal to 0')
        
    def test_is_zero_less_than_or_equal_to_zero(self):
        self.assertTrue(less_than_or_equal_to_zero(0), '0 is less than or equal to 0')
        
    def test_is_one_less_than_or_equal_to_zero(self):
        self.assertFalse(less_than_or_equal_to_zero(1), '1 is not less than or equal to 0')

# --------------------------
# replace_list_elements        
# --------------------------
class Replace_List_Elements_Test(unittest.TestCase):
    def test_no_replace_list_elements(self):
        self.assertEqual(replace_list_elements(np.array([1, 2, 3], dtype=float), less_than_or_equal_to_zero, np.Inf).all(), np.array([1,2,3]).all(), 'All elements greater than 0')
    def test_replace_list_elements(self):
        self.assertEqual(replace_list_elements(np.array([1, -2, 3], dtype=float), less_than_or_equal_to_zero, np.Inf).all(), np.array([1,np.Inf,3]).all(), '2nd element switched to np.Inf')

# --------------------------
# message        
# --------------------------
class MessageTest(unittest.TestCase):
    def test_does_message_not_print(self):
        self.assertFalse(message(0, 1, 'Message Not Printed'), 'Should not print')
    def test_does_message_print(self):
        self.assertTrue(message(1, 0, 'Message Printed'), 'Should print')

# --------------------------
# is_semi_pos_def_chol
# --------------------------
class Is_Semi_Pos_Def_Chol_Test(unittest.TestCase):        
    def test_is_semi_pos_def_chol(self):
        self.assertTrue(is_semi_pos_def_chol(np.array([[2, 1],[1, 2]])), 'Matrix is positive definite')
    def test_is_not_semi_pos_def_chol(self):
        self.assertFalse(is_semi_pos_def_chol(np.array([[1, 2],[2, 1]])), 'Matrix is not positive definite')