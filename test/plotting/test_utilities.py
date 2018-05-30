#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 05:57:34 2018

@author: prmiles
"""

from pymcmcstat.plotting import utilities
import unittest
import numpy as np

# --------------------------
class GenerateDefaultNames(unittest.TestCase):
    
    def test_size_of_default_names(self):
        nparam = 8
        names = utilities.generate_default_names(nparam = nparam)
        self.assertEqual(len(names), nparam, msg = 'Length of names should match number of parameters')
        
    def test_value_of_default_names(self):
        names = utilities.generate_default_names(nparam = 3)
        expected_names = ['$p_{0}$','$p_{1}$','$p_{2}$']
        self.assertEqual(names, expected_names, 
                         msg = str('Names do not match: Expected - {}, Received - {}'.format(expected_names, names)))

# --------------------------        
class ExtendNamesToMatchNparam(unittest.TestCase):
    
    def test_initially_empty_name_set(self):
        nparam = 3
        names = utilities.extend_names_to_match_nparam(names = None, nparam = nparam)
        expected_names = ['$p_{0}$','$p_{1}$','$p_{2}$']
        self.assertEqual(names, expected_names, 
                         msg = str('Names do not match: Expected - {}, Received - {}'.format(expected_names, names)))
        
    def test_single_entry_name_set(self):
        nparam = 3
        names = ['aa']
        names = utilities.extend_names_to_match_nparam(names = names, nparam = nparam)
        expected_names = ['aa','$p_{1}$','$p_{2}$']
        self.assertEqual(names, expected_names, 
                         msg = str('Names do not match: Expected - {}, Received - {}'.format(expected_names, names)))
        
    def test_double_entry_name_set(self):
        nparam = 3
        names = ['aa', 'zz']
        names = utilities.extend_names_to_match_nparam(names = names, nparam = nparam)
        expected_names = ['aa','zz','$p_{2}$']
        self.assertEqual(names, expected_names, 
                         msg = str('Names do not match: Expected - {}, Received - {}'.format(expected_names, names)))
    
# --------------------------    
class MakeXGrid(unittest.TestCase):
    
    def test_shape_of_output(self):
        x = np.linspace(0, 10, num = 50)
        npts = 20
        xgrid = utilities.make_x_grid(x = x, npts = 20)
        self.assertEqual(xgrid.shape, (npts, 1), msg = str('Expected return dimension of ({}, 1)'.format(npts)))
        
    def test_default_shape_of_output(self):
        x = np.linspace(0, 10, num = 50)
        xgrid = utilities.make_x_grid(x = x)
        self.assertEqual(xgrid.shape, (100, 1), msg = 'Expected return dimension of (100, 1)')
        
    def test_shape_of_output_for_dense_x(self):
        x = np.linspace(0, 10, num = 500)
        npts = 20
        xgrid = utilities.make_x_grid(x = x, npts = 20)
        self.assertEqual(xgrid.shape, (npts, 1), msg = 'Expected return dimension of (npts, 1)')
        
    def test_default_shape_of_output_for_dense_x(self):
        x = np.linspace(0, 10, num = 500)
        xgrid = utilities.make_x_grid(x = x)
        self.assertEqual(xgrid.shape, (100, 1), msg = 'Expected return dimension of (100, 1)')
    
# --------------------------
class GenerateEllipse(unittest.TestCase):

    def test_does_non_square_matrix_return_error(self):
        cmat = np.zeros([3,2])
        mu = np.zeros([2,1])
        with self.assertRaises(SystemExit):
            utilities.generate_ellipse(mu, cmat)
            
    def test_does_non_symmetric_matrix_return_error(self):
        cmat = np.array([[3,2],[1,3]])
        mu = np.zeros([2,1])
        with self.assertRaises(SystemExit):
            utilities.generate_ellipse(mu, cmat)
            
    def test_does_non_positive_definite_matrix_return_error(self):
        cmat = np.zeros([2,2])
        mu = np.zeros([2,1])
        with self.assertRaises(SystemExit):
            utilities.generate_ellipse(mu, cmat)
            
    def test_does_good_matrix_return_equal_sized_xy_arrays(self):
        cmat = np.eye(2)
        mu = np.zeros([2,1])
        x,y = utilities.generate_ellipse(mu, cmat)
        self.assertEqual(x.shape,y.shape)
        
    def test_does_good_matrix_return_correct_size_array(self):
        cmat = np.eye(2)
        mu = np.zeros([2,1])
        ndp = 50 # number of oints to generate ellipse shape
        x,y = utilities.generate_ellipse(mu, cmat, ndp)
        self.assertEqual(x.size,ndp)
        
# --------------------------
class GaussianDensityFunction(unittest.TestCase):
    
    def test_float_return_with_float_input(self):
        self.assertTrue(isinstance(utilities.gaussian_density_function(x = 0.), float), 
                             msg = 'Expected float return')
        
    def test_float_return_with_int_input(self):
        self.assertTrue(isinstance(utilities.gaussian_density_function(x = 0), float), 
                             msg = 'Expected float return')
        
    def test_float_return_with_float_input_at_nondefault_mean(self):
        self.assertTrue(isinstance(utilities.gaussian_density_function(x = 0., mu = 100), float), 
                             msg = 'Expected float return')
        
# --------------------------
class IQrange(unittest.TestCase):
    
    def test_array_return_with_column_vector_input(self):
        x = np.random.random_sample(size = (100,1))
        q = utilities.iqrange(x = x)
        self.assertTrue(isinstance(q, np.ndarray), msg = 'Expected array return')
        
    def test_array_return_with_row_vector_input(self):
        x = np.random.random_sample(size = (1,100))
        q = utilities.iqrange(x = x)
        self.assertTrue(isinstance(q, np.ndarray), msg = 'Expected array return')
        
# --------------------------
class ScaleBandWidth(unittest.TestCase):
    
    def test_array_return_with_column_vector_input(self):
        x = np.random.random_sample(size = (100,1))
        s = utilities.scale_bandwidth(x = x)
        self.assertTrue(isinstance(s, np.ndarray), msg = 'Expected array return')
        
    def test_array_return_with_row_vector_input(self):
        x = np.random.random_sample(size = (1,100))
        s = utilities.scale_bandwidth(x = x)
        self.assertTrue(isinstance(s, np.ndarray), msg = 'Expected array return')
        
