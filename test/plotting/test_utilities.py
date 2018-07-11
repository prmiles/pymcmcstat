#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 05:57:34 2018

@author: prmiles
"""

from pymcmcstat.plotting import utilities
import unittest
from mock import patch
import numpy as np
import math

# --------------------------
class GenerateSubplotGrid(unittest.TestCase):
    def test_generate_subplot_grid(self):
        nparam = 5
        ns1, ns2 = utilities.generate_subplot_grid(nparam = nparam)
        self.assertEqual(ns1, math.ceil(math.sqrt(nparam)), msg = 'Expect 3')
        self.assertEqual(ns2, round(math.sqrt(nparam)), msg = 'Expect 2')

    def test_generate_subplot_grid_1(self):
        nparam = 1
        ns1, ns2 = utilities.generate_subplot_grid(nparam = nparam)
        self.assertEqual(ns1, math.ceil(math.sqrt(nparam)), msg = 'Expect 1')
        self.assertEqual(ns2, round(math.sqrt(nparam)), msg = 'Expect 1')

# --------------------------
class GenerateNames(unittest.TestCase):
    
    def test_default_names(self):
        nparam = 8
        names = utilities.generate_names(nparam = nparam, names = None)
        self.assertEqual(len(names), nparam, msg = 'Length of names should match number of parameters')
        for ii in range(nparam):
            self.assertEqual(names[ii], str('$p_{{{}}}$'.format(ii)))

    def test_names_partial(self):
        nparam = 8
        names = ['hi']
        names = utilities.generate_names(nparam = nparam, names = names)
        self.assertEqual(names[0], 'hi', msg = 'First name is hi')
        for ii in range(1, nparam):
            self.assertEqual(names[ii], str('$p_{{{}}}$'.format(ii)))
 
# --------------------------
class SetupPlotFeatures(unittest.TestCase):
    def test_default_features(self):
        nparam = 2
        ns1, ns2, names, figsizeinches = utilities.setup_plot_features(nparam = nparam, names = None, figsizeinches = None)
        self.assertEqual(ns1, math.ceil(math.sqrt(nparam)), msg = 'Expect 3')
        self.assertEqual(ns2, round(math.sqrt(nparam)), msg = 'Expect 2')
        for ii in range(nparam):
            self.assertEqual(names[ii], str('$p_{{{}}}$'.format(ii)))
        self.assertEqual(figsizeinches, [5,4], msg = 'Default figure size is [5,4]')
        
    def test_nondefault_features(self):
        nparam = 2
        ns1, ns2, names, figsizeinches = utilities.setup_plot_features(nparam = nparam, names = ['hi'], figsizeinches = [7,2])
        self.assertEqual(ns1, math.ceil(math.sqrt(nparam)), msg = 'Expect 3')
        self.assertEqual(ns2, round(math.sqrt(nparam)), msg = 'Expect 2')
        self.assertEqual(names[0], 'hi', msg = 'First name is hi')
        for ii in range(1, nparam):
            self.assertEqual(names[ii], str('$p_{{{}}}$'.format(ii)))
        self.assertEqual(figsizeinches, [7,2], msg = 'Default figure size is [7,2]')
        
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
        ndp = 50 # number of points to generate ellipse shape
        x,y = utilities.generate_ellipse(mu, cmat, ndp)
        self.assertEqual(x.size, ndp)
        self.assertEqual(y.size, ndp)
        
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
        self.assertTrue(isinstance(q, np.ndarray), msg = 'Expected array return - received {}'.format(type(q)))
        self.assertEqual(q.size, 1, msg = 'Expect single element array')
        
    def test_array_return_with_row_vector_input(self):
        x = np.random.random_sample(size = (1,100))
        q = utilities.iqrange(x = x)
        self.assertTrue(isinstance(q, np.ndarray), msg = 'Expected array return - received {}'.format(type(q)))
        self.assertEqual(q.size, 1, msg = 'Expect single element array')
        
# --------------------------
class ScaleBandWidth(unittest.TestCase):
    
    def test_array_return_with_column_vector_input(self):
        x = np.random.random_sample(size = (100,1))
        s = utilities.scale_bandwidth(x = x)
        self.assertTrue(isinstance(s, np.ndarray), msg = 'Expected array return - received {}'.format(type(s)))
        self.assertEqual(s.size, 1, msg = 'Expect single element array')
        
    def test_array_return_with_row_vector_input(self):
        x = np.random.random_sample(size = (1,100))
        s = utilities.scale_bandwidth(x = x)
        self.assertTrue(isinstance(s, np.ndarray), msg = 'Expected array return - received {}'.format(type(s)))
        self.assertEqual(s.size, 1, msg = 'Expect single element array')
        
    @patch('pymcmcstat.plotting.utilities.iqrange', return_value = -1.0)
    def test_array_return_with_iqrange_lt_0(self, mock_iqrange):
        x = np.random.random_sample(size = (1,100))
        s = utilities.scale_bandwidth(x = x)
        self.assertTrue(isinstance(s, np.ndarray), msg = 'Expected array return - received {}'.format(type(s)))
        self.assertEqual(s.size, 1, msg = 'Expect single element array')
        
    @patch('pymcmcstat.plotting.utilities.iqrange', return_value = 1.0)
    def test_array_return_with_iqrange_gt_0(self, mock_iqrange):
        x = np.random.random_sample(size = (1,100))
        s = utilities.scale_bandwidth(x = x)
        self.assertTrue(isinstance(s, np.ndarray), msg = 'Expected array return - received {}'.format(type(s)))
        self.assertEqual(s.size, 1, msg = 'Expect single element array')
        
# --------------------------
class AppendToNrowNcolBasedOnShape(unittest.TestCase):
    def test_shape_is_2d(self):
        nrow = []
        ncol = []
        sh = (2,1)
        nrow, ncol = utilities.append_to_nrow_ncol_based_on_shape(sh = sh, nrow = nrow, ncol = ncol)
        self.assertEqual(nrow, [2], msg = 'Expect [2]')
        self.assertEqual(ncol, [1], msg = 'Expect [1]')
        
    def test_shape_is_1d(self):
        nrow = []
        ncol = []
        sh = (2,)
        nrow, ncol = utilities.append_to_nrow_ncol_based_on_shape(sh = sh, nrow = nrow, ncol = ncol)
        self.assertEqual(nrow, [2], msg = 'Expect [2]')
        self.assertEqual(ncol, [1], msg = 'Expect [1]')
        
# --------------------------
class ConvertFlagToBoolean(unittest.TestCase):
    def test_boolean_conversion(self):
        self.assertTrue(utilities.convert_flag_to_boolean(flag = 'on'), msg = 'on -> True')
        self.assertFalse(utilities.convert_flag_to_boolean(flag = 'off'), msg = 'off -> False')
        
# --------------------------
class SetLocalParameters(unittest.TestCase):
    def test_set_local_parameters(self):
        slp = utilities.set_local_parameters
        self.assertTrue(np.array_equal(slp(ii = 0, local = np.array([0, 0])), np.array([True, True])), msg = 'Expect Array [True, True]')
        self.assertTrue(np.array_equal(slp(ii = 0, local = np.array([0, 1])), np.array([True, False])), msg = 'Expect Array [True, False]')
        self.assertTrue(np.array_equal(slp(ii = 0, local = np.array([1, 0])), np.array([False, True])), msg = 'Expect Array [False, True]')

        self.assertTrue(np.array_equal(slp(ii = 1, local = np.array([0, 1])), np.array([True, True])), msg = 'Expect Array [True, True]')
        self.assertTrue(np.array_equal(slp(ii = 1, local = np.array([1, 0])), np.array([True, True])), msg = 'Expect Array [True, True]')
        
        self.assertTrue(np.array_equal(slp(ii = 1, local = np.array([2, 2])), np.array([False, False])), msg = 'Expect Array [False, False]')
        self.assertTrue(np.array_equal(slp(ii = 2, local = np.array([1, 2])), np.array([False, True])), msg = 'Expect Array [False, True]')
        
# --------------------------------------------
class Empirical_Quantiles_Test(unittest.TestCase):

    def test_does_default_empirical_quantiles_return_3_element_array(self):
        test_out = utilities.empirical_quantiles(np.random.rand(10,1))
        self.assertEqual(test_out.shape, (3,1), msg = 'Default output shape is (3,1)')
        
    def test_does_non_default_empirical_quantiles_return_2_element_array(self):
        test_out = utilities.empirical_quantiles(np.random.rand(10,1), p = np.array([0.2, 0.5]))
        self.assertEqual(test_out.shape, (2,1), msg = 'Non-default output shape should be (2,1)')
        
    def test_empirical_quantiles_should_not_support_list_input(self):
        with self.assertRaises(AttributeError):
            utilities.empirical_quantiles([-1,0,1])
            
    def test_empirical_quantiles_vector(self):
        out = utilities.empirical_quantiles(np.linspace(10,20, num = 10).reshape(10,1), p = np.array([0.22, 0.57345]))
        exact = np.array([[12.2], [15.7345]])
        comp = np.linalg.norm(out - exact)
        self.assertAlmostEqual(comp, 0)
        
# --------------------------------------------
class CheckDefaults(unittest.TestCase):
    def test_check_defaults(self):
        defaults = {'model_display': '-r'}
        kwargs = {'model_display': '--k', 'hi': 3}
        kwargsout = utilities.check_defaults(kwargs, defaults)
        self.assertEqual(kwargsout['model_display'], '--k', msg = 'Expect --k')
        self.assertEqual(kwargsout['hi'], 3, msg = 'Expect 3')
        
    def test_check_defaults_used_defaults(self):
        defaults = {'model_display': '-r'}
        kwargs = {'hi': 3}
        kwargsout = utilities.check_defaults(kwargs, defaults)
        self.assertEqual(kwargsout['model_display'], '-r', msg = 'Expect --k')
        self.assertEqual(kwargsout['hi'], 3, msg = 'Expect 3')