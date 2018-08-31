#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 11:14:18 2018

Description: This file contains a series of utilities designed to test the
features in the 'DataStructure.py" package of the pymcmcstat module.

@author: prmiles
"""
from pymcmcstat.settings import DataStructure
import unittest
import numpy as np

# --------------------------
# initialization
# --------------------------
class Data_Structure_Initialization(unittest.TestCase):
    def test_does_initialization_yield_xdata(self):
        data = DataStructure.DataStructure()
        self.assertTrue(hasattr(data,'xdata'))
    def test_does_initialization_yield_ydata(self):
        data = DataStructure.DataStructure()
        self.assertTrue(hasattr(data,'ydata'))
    def test_does_initialization_yield_n(self):
        data = DataStructure.DataStructure()
        self.assertTrue(hasattr(data,'n'))
    def test_does_initialization_yield_shape(self):
        data = DataStructure.DataStructure()
        self.assertTrue(hasattr(data,'shape'))
    def test_does_initialization_yield_weight(self):
        data = DataStructure.DataStructure()
        self.assertTrue(hasattr(data,'weight'))
    def test_does_initialization_yield_udobj(self):
        data = DataStructure.DataStructure()
        self.assertTrue(hasattr(data,'user_defined_object'))

# --------------------------
# adding data
# --------------------------
class Add_Data_Set(unittest.TestCase):
    
    def test_is_xdata_a_list(self):
        data = DataStructure.DataStructure()
        x = np.zeros([2])
        y = np.zeros([2])
        data.add_data_set(x,y)
        self.assertIsInstance(data.xdata,list, msg = 'xdata should be a list')
        
    def test_is_ydata_a_list(self):
        data = DataStructure.DataStructure()
        x = np.zeros([2])
        y = np.zeros([2])
        data.add_data_set(x,y)
        self.assertIsInstance(data.ydata,list, msg = 'ydata should be a list')
    
    def test_is_x_in_xdata_a_numpy_array_if_numpy_array_sent(self):
        data = DataStructure.DataStructure()
        x = np.zeros([2])
        y = np.zeros([2])
        data.add_data_set(x,y)
        self.assertIsInstance(data.xdata[0], np.ndarray, msg = 'xdata[0] should be a numpy array')
        
    def test_is_x_in_xdata_a_numpy_array_if_list_sent(self):
        data = DataStructure.DataStructure()
        x = [0]
        y = np.zeros([2])
        data.add_data_set(x,y)
        self.assertIsInstance(data.xdata[0], np.ndarray, msg = 'xdata[0] should be a numpy array')
        
    def test_is_y_in_ydata_a_numpy_array_if_numpy_array_sent(self):
        data = DataStructure.DataStructure()
        x = np.zeros([2])
        y = np.zeros([2])
        data.add_data_set(x,y)
        self.assertIsInstance(data.ydata[0], np.ndarray, msg = 'ydata[0] should be a numpy array')
        
    def test_is_y_in_ydata_a_numpy_array_if_list_sent(self):
        data = DataStructure.DataStructure()
        x = [0]
        y = [0]
        data.add_data_set(x,y)
        self.assertIsInstance(data.ydata[0], np.ndarray, msg = 'ydata[0] should be a numpy array')
    
    def test_is_y_in_ydata_a_numpy_array_if_element_sent(self):
        data = DataStructure.DataStructure()
        x = [0]
        y = 0
        data.add_data_set(x,y)
        self.assertIsInstance(data.ydata[0], np.ndarray, msg = 'ydata[0] should be a numpy array')
    
    def test_is_x_in_xdata_a_numpy_array_if_numpy_array_of_size_1(self):
        data = DataStructure.DataStructure()
        x = np.zeros(2)
        y = np.zeros([2])
        data.add_data_set(x,y)
        self.assertIsInstance(data.xdata[0], np.ndarray, msg = 'xdata[0] should be a numpy array')
        
    def test_does_shape_match_y_shape(self):
        data = DataStructure.DataStructure()
        x = np.zeros([2])
        y = np.zeros([2])
        data.add_data_set(x,y)
        self.assertEqual(data.shape[0], data.ydata[0].shape, msg = 'shapes should match')
        
    def test_2d_y_output_2d_shape(self):
        data = DataStructure.DataStructure()
        x = np.zeros([2])
        y = np.zeros([2,2])
        data.add_data_set(x,y)
        self.assertEqual(len(data.shape[0]), len(data.ydata[0].shape), msg = 'lengths should match')
        
# --------------------------
# get_number_of_batches
# --------------------------
class Get_Number_Of_Batches(unittest.TestCase):
    def test_does_nbatch_equal_length_of_shape_for_1d_y(self):
        data = DataStructure.DataStructure()
        x = np.zeros([2])
        y = np.zeros([2])
        data.add_data_set(x,y)
        data.get_number_of_batches()
        self.assertEqual(data.nbatch,1, msg = 'nbatch should match number of calls to ''add_data_set''')
        
    def test_does_nbatch_match_for_1d_y_with_multiple_data_sets(self):
        data = DataStructure.DataStructure()
        x = np.zeros([2])
        y = np.zeros([2])
        data.add_data_set(x,y)
        data.add_data_set(x,y)
        data.get_number_of_batches()
        self.assertEqual(data.nbatch,2, msg = 'nbatch should match number of calls to ''add_data_set''')
    
    def test_does_nbatch_match_for_1d_y_with_multiple_data_sets_with_different_size(self):
        data = DataStructure.DataStructure()
        x = np.zeros([2])
        y = np.zeros([2])
        data.add_data_set(x,y)
        y = np.zeros([2,2])
        data.add_data_set(x,y)
        data.get_number_of_batches()
        self.assertEqual(data.nbatch,2, msg = 'nbatch should match number of calls to ''add_data_set''')
    
    def test_does_nbatch_equal_length_of_shape_for_1d_y_v2(self):
        data = DataStructure.DataStructure()
        x = np.zeros([2,])
        y = np.zeros([2,])
        data.add_data_set(x,y)
        data.get_number_of_batches()
        self.assertEqual(data.nbatch,1, msg = 'nbatch should match number of calls to ''add_data_set''')
    
    def test_does_nbatch_equal_length_of_shape_for_2d_y(self):
        data = DataStructure.DataStructure()
        x = np.zeros([2,])
        y = np.zeros([2,2])
        data.add_data_set(x,y)
        data.get_number_of_batches()
        self.assertEqual(data.nbatch, 1, msg = 'nbatch should match number of calls to ''add_data_set''')
        
# --------------------------
# get_number_of_data_sets
# --------------------------
class Get_Number_Of_Data_Sets(unittest.TestCase):
    def setup_data_set(self, xsh = [2], ysh = [2], nsets = 1):
        data = DataStructure.DataStructure()
        x = np.zeros(xsh)
        y = np.zeros(ysh)
        for _ in range(nsets):
            data.add_data_set(x,y)
        data.get_number_of_data_sets()
        return data
    def add_data_set(self, data, xsh = [2], ysh = [2], nsets = 1):
        x = np.zeros(xsh)
        y = np.zeros(ysh)
        for _ in range(nsets):
            data.add_data_set(x,y)
        data.get_number_of_data_sets()
        return data
    def test_does_single_data_set_match(self):
        data = self.setup_data_set()
        self.assertEqual(data.ndatasets, 1, msg = 'ndatasets should match total number of columns of elements of ydata')
        
    def test_does_double_data_set_match(self):
        data = self.setup_data_set(nsets = 2)
        self.assertEqual(data.ndatasets, 2, msg = 'ndatasets should match total number of columns of elements of ydata')
        
    def test_does_double_data_set_match_with_2d_sets(self):
        data = self.setup_data_set(ysh = [2,2], nsets = 2)
        self.assertEqual(data.ndatasets, 4, msg = 'ndatasets should match total number of columns of elements of ydata')
     
    def test_does_double_data_set_match_with_2d_sets_of_different_size(self):
        data = self.setup_data_set(ysh = [2,2])
        data = self.add_data_set(data, ysh = [3,4])
        self.assertEqual(data.ndatasets, 6, msg = 'ndatasets should match total number of columns of elements of ydata')
     
    def test_does_double_data_set_match_with_different_sizes(self):
        data = self.setup_data_set(ysh = [2])
        data = self.add_data_set(data, ysh = [2,2])
        self.assertEqual(data.ndatasets, 3, msg = 'ndatasets should match total number of columns of elements of ydata')
        
        
# --------------------------
# get_number_of_observations
# --------------------------
class Get_Number_Of_Observations(unittest.TestCase):
    def test_number_of_observations_match_sum_of_n(self):
        data = DataStructure.DataStructure()
        x = np.zeros([2])
        y = np.zeros([2])
        data.add_data_set(x,y)
        nds = data.get_number_of_observations()
        self.assertEqual(nds[0], 2, msg = 'total number of observations is number of rows in y')
        
    def test_number_of_observations_match_sum_of_n_with_2d_y(self):
        data = DataStructure.DataStructure()
        x = np.zeros([2])
        y = np.zeros([2,2])
        data.add_data_set(x,y)
        nds = data.get_number_of_observations()
        self.assertEqual(nds[0], 2, msg = 'total number of observations is number of rows in y')
        
    def test_number_of_observations_match_sum_of_n_with_multiple_sets(self):
        data = DataStructure.DataStructure()
        x = np.zeros([2])
        y = np.zeros([2])
        data.add_data_set(x,y)
        data.add_data_set(x,y)
        nds = data.get_number_of_observations()
        self.assertEqual(nds[0], 4, msg = 'total number of observations is number of rows in y - summed over all sets of y')
        
    def test_number_of_observations_match_sum_of_n_with_multiple_sets_with_2d_y(self):
        data = DataStructure.DataStructure()
        x = np.zeros([2])
        y = np.zeros([2])
        data.add_data_set(x,y)
        y = np.zeros([2,2])
        data.add_data_set(x,y)
        nds = data.get_number_of_observations()
        self.assertEqual(nds[0], 4, msg = 'total number of observations is number of rows in y - summed over all sets of y')
        
    def test_number_of_observations_match_sum_of_n_with_multiple_2d_sets(self):
        data = DataStructure.DataStructure()
        x = np.zeros([2])
        y = np.zeros([1,2])
        data.add_data_set(x,y)
        y = np.zeros([2,2])
        data.add_data_set(x,y)
        nds = data.get_number_of_observations()
        self.assertEqual(nds[0], 3, msg = 'total number of observations is number of rows in y - summed over all sets of y')