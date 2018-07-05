#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 13:17:23 2018

Description: This file contains a series of utilities designed to test the
features in the 'ModelSettings.py" package of the pymcmcstat module.

@author: prmiles
"""

from pymcmcstat.settings.DataStructure import DataStructure
from pymcmcstat.settings.SimulationOptions import SimulationOptions
from pymcmcstat.settings.ModelSettings import ModelSettings
import unittest
import numpy as np

MS = ModelSettings()

def setup_model_settings(**kwargs):
    # create settings
    MS = ModelSettings()
    MS.define_model_settings(**kwargs)
    return MS

def setup_simulation_options(**kwargs):
    # create options
    options = SimulationOptions()
    options.define_simulation_options(**kwargs)
    return options

# --------------------------
# initialization
# --------------------------
class Model_Settings_Initialization(unittest.TestCase):

    def test_does_initialization_yield_description(self):
        ms = ModelSettings()
        self.assertTrue(hasattr(ms,'description'))

# --------------------------
# define_model_settings
# --------------------------
class Define_Model_Settings(unittest.TestCase):
       
    def test_default_sos_function(self):
        ms = setup_model_settings()
        self.assertEqual(ms.sos_function, None, msg = 'Should be ''None''')
        
    def test_default_prior_function(self):
        ms = setup_model_settings()
        self.assertEqual(ms.prior_function, None, msg = 'Should be ''None''')
        
    def test_default_prior_type(self):
        ms = setup_model_settings()
        self.assertEqual(ms.prior_type, 1, msg = 'Should be 1')
        
    def test_default_prior_update_function(self):
        ms = setup_model_settings()
        self.assertEqual(ms.prior_update_function, None, msg = 'Should be ''None''')
        
    def test_default_prior_pars(self):
        ms = setup_model_settings()
        self.assertEqual(ms.prior_pars, None, msg = 'Should be ''None''')
        
    def test_default_model_function(self):
        ms = setup_model_settings()
        self.assertEqual(ms.model_function, None, msg = 'Should be ''None''')
        
    def test_default_sigma2(self):
        ms = setup_model_settings()
        self.assertEqual(ms.sigma2, None, msg = 'Should be ''None''')
        
    def test_default_N(self):
        ms = setup_model_settings()
        self.assertEqual(ms.N, None, msg = 'Should be ''None''')
        
    def test_default_S20(self):
        ms = setup_model_settings()
        self.assertTrue(np.isnan(ms.S20), msg = 'Should be ''nan''')
        
    def test_default_N0(self):
        ms = setup_model_settings()
        self.assertEqual(ms.N0, None, msg = 'Should be ''None''')
        
    def test_default_nbatch(self):
        ms = setup_model_settings()
        self.assertEqual(ms.nbatch, None, msg = 'Should be ''None''')
        
    def test_list_N_returned_as_numpy_array(self):
        ms = setup_model_settings(N = [1, 1])
        self.assertIsInstance(ms.N, np.ndarray, msg = 'Should be a numpy array')
        
    def test_float_N_returned_as_numpy_array(self):
        ms = setup_model_settings(N = 1.2)
        self.assertIsInstance(ms.N, np.ndarray, msg = 'Should be a numpy array')
        
    def test_int_N_returned_as_numpy_array(self):
        ms = setup_model_settings(N = 1)
        self.assertIsInstance(ms.N, np.ndarray, msg = 'Should be a numpy array')
        
    def test_tuple_N_returns_error(self):
        ms = ModelSettings()
        with self.assertRaises(SystemExit, msg = 'Tuple input not accepted'):
            ms.define_model_settings(N = (1,2))
            
# --------------------------
# _check_dependent_model_settings
# --------------------------
class Check_Dependent_Model_Settings(unittest.TestCase):
    
    def test_nbatch_calc_from_data(self):
        # create test data
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2])
        data.add_data_set(x,y)
        
        options = setup_simulation_options(nsimu = int(1000))
        ms = setup_model_settings()
        
        # calculate dependencies
        ms._check_dependent_model_settings(data, options)
        
        self.assertEqual(ms.nbatch, 1, msg = 'nbatch should equal number of calls to ''add_data_set''')
        
    def test_nbatch_calc_from_data_y_2d(self):
        # create test data
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2,2])
        data.add_data_set(x,y)
        
        options = setup_simulation_options(nsimu = int(1000))
        ms = setup_model_settings()
        
        # calculate dependencies
        ms._check_dependent_model_settings(data, options)
        
        self.assertEqual(ms.nbatch, 1, msg = 'nbatch should equal number of calls to ''add_data_set''')
        
    def test_nbatch_calc_from_data_multiple_sets(self):
        # create test data
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2])
        data.add_data_set(x,y)
        data.add_data_set(x,y)
        
        options = setup_simulation_options(nsimu = int(1000))
        ms = setup_model_settings()
        
        # calculate dependencies
        ms._check_dependent_model_settings(data, options)
        
        self.assertEqual(ms.nbatch, 2, msg = 'nbatch should equal number of calls to ''add_data_set''')
        
    def test_nbatch_calc_from_data_multiple_data_sets_of_different_size(self):
        # create test data
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2,2])
        data.add_data_set(x,y)
        y = np.zeros([2,])
        data.add_data_set(x,y)
        y = np.zeros([5])
        data.add_data_set(x,y)
        
        options = setup_simulation_options(nsimu = int(1000))
        ms = setup_model_settings()
        
        # calculate dependencies
        ms._check_dependent_model_settings(data, options)
        
        self.assertEqual(ms.nbatch, 3, msg = 'nbatch should equal number of calls to ''add_data_set''')
        
    def test_N_calc_from_data(self):
        # create test data
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2,2])
        data.add_data_set(x,y)
        
        options = setup_simulation_options(nsimu = int(1000))
        ms = setup_model_settings()
        
        # calculate dependencies
        ms._check_dependent_model_settings(data, options)
        
        self.assertTrue(np.array_equal(ms.N, np.array([2])), msg = 'N should equal total number of rows in each y set')
        
    def test_N_calc_from_data_with_multiple_sets(self):
        # create test data
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2,2])
        data.add_data_set(x,y)
        y = np.zeros([3,3])
        data.add_data_set(x,y)
        
        options = setup_simulation_options(nsimu = int(1000))
        ms = setup_model_settings()
        
        # calculate dependencies
        ms._check_dependent_model_settings(data, options)
        
        self.assertTrue(np.array_equal(ms.N, np.array([2,3])), msg = 'N should equal total number of rows in each y set')
        
    def test_N_calc_from_data_with_multiple_sets_of_different_size(self):
        # create test data
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2,2])
        data.add_data_set(x,y)
        y = np.zeros([3,3])
        data.add_data_set(x,y)
        y = np.zeros([4,7])
        data.add_data_set(x,y)
        
        options = setup_simulation_options(nsimu = int(1000))
        ms = setup_model_settings()
        
        # calculate dependencies
        ms._check_dependent_model_settings(data, options)
        
        self.assertTrue(np.array_equal(ms.N, np.array([2,3,4])), msg = 'N should equal total number of rows in each y set')
        
    def test_N_conflict_produces_error_message(self):
        # create test data
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2,2])
        data.add_data_set(x,y)
        
        options = setup_simulation_options(nsimu = int(1000))
        ms = setup_model_settings(N = 1)
                
        # calculate dependencies
        with self.assertRaises(SystemExit, msg = 'Conflicting N'):
            ms._check_dependent_model_settings(data, options)
            
    def test_default_sigma2_dependent(self):
        # create test data
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2,2])
        data.add_data_set(x,y)
        
        options = setup_simulation_options(nsimu = int(1000))
        ms = setup_model_settings()
        
        # calculate dependencies
        ms._check_dependent_model_settings(data, options)
        
        self.assertEqual(ms.sigma2, np.ones([1]), msg='default is 1');
        
    def test_default_N0_dependent(self):
        # create test data
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2,2])
        data.add_data_set(x,y)
        
        options = setup_simulation_options(nsimu = int(1000))
        ms = setup_model_settings()
        
        # calculate dependencies
        ms._check_dependent_model_settings(data, options)
        
        self.assertEqual(ms.N0, np.zeros([1]), msg='default is 0');
        
    def test_default_N0_dependent_for_non_default_sigma2(self):
        # create test data
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2,2])
        data.add_data_set(x,y)
        
        options = setup_simulation_options(nsimu = int(1000))
        ms = setup_model_settings(sigma2 = 1)
        
        # calculate dependencies
        ms._check_dependent_model_settings(data, options)
        
        self.assertEqual(ms.N0, np.ones([1]), msg='default is 1');
        
    def test_default_sigma2_dependent_for_non_default_N0(self):
        # create test data
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2,2])
        data.add_data_set(x,y)
        
        options = setup_simulation_options(nsimu = int(1000))
        ms = setup_model_settings(N0 = 1)
        
        # calculate dependencies
        ms._check_dependent_model_settings(data, options)
        
        self.assertEqual(ms.sigma2, np.ones(ms.nbatch), msg='default is ones(ms.nbatch)');
        
    def test_default_sigma2_dependent_for_non_default_N0_S20(self):
        # create test data
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2,2])
        data.add_data_set(x,y)
        
        options = setup_simulation_options(nsimu = int(1000))
        ms = setup_model_settings(N0 = 1, S20 = 1)
        
        # calculate dependencies
        ms._check_dependent_model_settings(data, options)
        
        self.assertEqual(ms.sigma2, ms.S20, msg='default is S20');
        
    def test_default_sigma2_dependent_for_non_default_N0_S20_multiple_data_sets(self):
        # create test data
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2,2])
        data.add_data_set(x,y)
        y = np.zeros([3,2])
        data.add_data_set(x,y)
        
        options = setup_simulation_options(nsimu = int(1000))
        ms = setup_model_settings(N0 = 1, S20 = 1)
        
        # calculate dependencies
        ms._check_dependent_model_settings(data, options)
        
        comp = np.linalg.norm(ms.sigma2 - ms.S20)
        self.assertAlmostEqual(comp, 0, msg='default is S20')
        
    def test_default_S20_dependent(self):
        # create test data
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2,2])
        data.add_data_set(x,y)
        
        options = setup_simulation_options(nsimu = int(1000))
        ms = setup_model_settings()
        
        # calculate dependencies
        ms._check_dependent_model_settings(data, options)
        
        self.assertAlmostEqual(ms.S20, ms.sigma2, msg='default is sigma2')
        
    def test_default_S20_dependent_for_multiple_data_sets(self):
        # create test data
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2,2])
        data.add_data_set(x,y)
        y = np.zeros([3,2])
        data.add_data_set(x,y)
        
        options = setup_simulation_options(nsimu = int(1000))
        ms = setup_model_settings()
        
        # calculate dependencies
        ms._check_dependent_model_settings(data, options)
        
        comp = np.linalg.norm(ms.sigma2 - ms.S20)
        self.assertAlmostEqual(comp, 0, msg='default is sigma2')
        
# --------------------------
# _check_dependent_model_settings_wrt_nsos
# --------------------------
class Check_Dependent_Model_Settings_WRT_Nsos(unittest.TestCase):
    
    def test_size_S20_wrt_nsos(self):
        # create test data
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2])
        data.add_data_set(x,y)
        
        options = setup_simulation_options(nsimu = int(1000))
        ms = setup_model_settings()
        
        # calculate dependencies
        ms._check_dependent_model_settings(data, options)
        
        nsos = 2
        ms._check_dependent_model_settings_wrt_nsos(nsos = nsos)
        
        self.assertEqual(len(ms.S20), nsos, msg = 'length of S20 should equal number of elements returned from sos function')
        
    def test_size_N0_wrt_nsos(self):
        # create test data
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2])
        data.add_data_set(x,y)
        
        options = setup_simulation_options(nsimu = int(1000))
        ms = setup_model_settings()
        
        # calculate dependencies
        ms._check_dependent_model_settings(data, options)
        
        nsos = 2
        ms._check_dependent_model_settings_wrt_nsos(nsos = nsos)
        
        self.assertEqual(len(ms.N0), nsos, msg = 'length of N0 should equal number of elements returned from sos function')
        
    def test_size_sigma2_wrt_nsos(self):
        # create test data
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2])
        data.add_data_set(x,y)
        
        options = setup_simulation_options(nsimu = int(1000))
        ms = setup_model_settings()
        
        # calculate dependencies
        ms._check_dependent_model_settings(data, options)
        
        nsos = 2
        ms._check_dependent_model_settings_wrt_nsos(nsos = nsos)
        
        self.assertEqual(len(ms.sigma2), nsos, msg = 'length of sigma2 should equal number of elements returned from sos function')
    
    def test_size_S20_wrt_nsos_for_non_default_S20(self):
        # create test data
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2])
        data.add_data_set(x,y)
        
        options = setup_simulation_options(nsimu = int(1000))
        ms = setup_model_settings(S20 = [1,2])
        
        # calculate dependencies
        ms._check_dependent_model_settings(data, options)
        
        nsos = 2
        ms._check_dependent_model_settings_wrt_nsos(nsos = nsos)
        
        self.assertEqual(len(ms.S20), nsos, msg = 'length of S20 should equal number of elements returned from sos function')
        
    def test_size_S20_wrt_nsos_for_non_default_S20_raises_error(self):
        # create test data
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2])
        data.add_data_set(x,y)
        
        options = setup_simulation_options(nsimu = int(1000))
        ms = setup_model_settings(S20 = [1,2])
        
        # calculate dependencies
        ms._check_dependent_model_settings(data, options)
        
        nsos = 3
        with self.assertRaises(SystemExit, msg = 'S20 size mismatch'):
            ms._check_dependent_model_settings_wrt_nsos(nsos = nsos)
            
    def test_size_N0_wrt_nsos_for_non_default_N0(self):
        # create test data
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2])
        data.add_data_set(x,y)
        
        options = setup_simulation_options(nsimu = int(1000))
        ms = setup_model_settings(N0 = [1,2])
        
        # calculate dependencies
        ms._check_dependent_model_settings(data, options)
        
        nsos = 2
        ms._check_dependent_model_settings_wrt_nsos(nsos = nsos)
        
        self.assertEqual(len(ms.N0), nsos, msg = 'length of N0 should equal number of elements returned from sos function')
    
    def test_size_N0_wrt_nsos_for_non_default_N0_raises_error(self):
        # create test data
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2])
        data.add_data_set(x,y)
        
        options = setup_simulation_options(nsimu = int(1000))
        ms = setup_model_settings(N0 = [1,2])
        
        # calculate dependencies
        ms._check_dependent_model_settings(data, options)
        
        nsos = 3
        with self.assertRaises(SystemExit, msg = 'N0 size mismatch'):
            ms._check_dependent_model_settings_wrt_nsos(nsos = nsos)
        
    def test_size_sigma2_wrt_nsos_for_non_default_sigma2(self):
        # create test data
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2])
        data.add_data_set(x,y)
        
        options = setup_simulation_options(nsimu = int(1000))
        ms = setup_model_settings(sigma2 = [1,2])
        
        # calculate dependencies
        ms._check_dependent_model_settings(data, options)
        
        nsos = 2
        ms._check_dependent_model_settings_wrt_nsos(nsos = nsos)
        
        self.assertEqual(len(ms.sigma2), nsos, msg = 'length of sigma2 should equal number of elements returned from sos function')
        
    def test_size_sigma2_wrt_nsos_for_non_default_sigma2_raises_error(self):
        # create test data
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2])
        data.add_data_set(x,y)
        
        options = setup_simulation_options(nsimu = int(1000))
        ms = setup_model_settings(sigma2 = [1,2])
        
        # calculate dependencies
        ms._check_dependent_model_settings(data, options)
        
        nsos = 3
        with self.assertRaises(SystemExit, msg = 'sigma2 size mismatch'):
            ms._check_dependent_model_settings_wrt_nsos(nsos = nsos)
            
    def test_value_S20_wrt_nsos_raises_error(self):
        # create test data
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2])
        data.add_data_set(x,y)
        
        # create options
        options = SimulationOptions()
        options.define_simulation_options(nsimu = int(1000))
        
        # create settings
        ms = setup_model_settings(S20 = [1,2])
        # calculate dependencies
        ms._check_dependent_model_settings(data, options)
        nsos = 1
        with self.assertRaises(SystemExit, msg = 'S20 should not be larger than nsos'):
            ms._check_dependent_model_settings_wrt_nsos(nsos = nsos)
            
    def test_value_N0_wrt_nsos_raises_error(self):
        # create test data
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2])
        data.add_data_set(x,y)
        
        options = setup_simulation_options(nsimu = int(1000))
        ms = setup_model_settings(N0 = [1,2])
        # calculate dependencies
        ms._check_dependent_model_settings(data, options)
        nsos = 1
        with self.assertRaises(SystemExit, msg = 'N0 should not be larger than nsos'):
            ms._check_dependent_model_settings_wrt_nsos(nsos = nsos)
            
    def test_value_sigma2_wrt_nsos_raises_error(self):
        # create test data
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2])
        data.add_data_set(x,y)
        
        options = setup_simulation_options(nsimu = int(1000))
        ms = setup_model_settings(sigma2 = [1,2])
        # calculate dependencies
        ms._check_dependent_model_settings(data, options)
        nsos = 1
        with self.assertRaises(SystemExit, msg = 'sigma2 should not be larger than nsos'):
            ms._check_dependent_model_settings_wrt_nsos(nsos = nsos)
     
#   CHECK CALCULATION OF NUMBER OF OBSERVATIONS
    def test_size_N_wrt_nsos(self):
        # create test data
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2])
        data.add_data_set(x,y)
        
        options = setup_simulation_options(nsimu = int(1000))
        ms = setup_model_settings()
        
        # calculate dependencies
        ms._check_dependent_model_settings(data, options)
        
        nsos = 2
        ms._check_dependent_model_settings_wrt_nsos(nsos = nsos)
        
        self.assertEqual(len(ms.N), nsos, msg = 'length of N should equal number of elements returned from sos function')
       
    def test_size_N_wrt_nsos_for_2_data_sets(self):
        # create test data
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2])
        data.add_data_set(x,y)
        data.add_data_set(x,y)
        
        options = setup_simulation_options(nsimu = int(1000))
        ms = setup_model_settings()
        
        # calculate dependencies
        ms._check_dependent_model_settings(data, options)
        
        nsos = 2
        print('N = {}, nsos = {}'.format(ms.N, nsos))
        ms._check_dependent_model_settings_wrt_nsos(nsos = nsos)
        print('N = {}, nsos = {}'.format(ms.N, nsos))
        self.assertEqual(len(ms.N), nsos, msg = 'length of N should equal number of elements returned from sos function')
       
    def test_size_N_wrt_nsos_for_non_default_N(self):
        # create test data
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2])
        data.add_data_set(x,y)
        y = np.zeros([4,2])
        data.add_data_set(x,y)
        
        options = setup_simulation_options(nsimu = int(1000))
        ms = setup_model_settings(N = [2,4])
        
        # calculate dependencies
        ms._check_dependent_model_settings(data, options)
        
        nsos = 3
        
        with self.assertRaises(SystemExit, msg = 'length of N should equal number of elements returned from sos function'):
            ms._check_dependent_model_settings_wrt_nsos(nsos = nsos)
            
    def test_size_N_wrt_nsos_for_non_default_N_and_nsos_equal_1(self):
        # create test data
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2])
        data.add_data_set(x,y)
        y = np.zeros([4,2])
        data.add_data_set(x,y)
        
        options = setup_simulation_options(nsimu = int(1000))
        ms = setup_model_settings(N = [2,4])
        
        # calculate dependencies
        ms._check_dependent_model_settings(data, options)
        
        nsos = 1
        ms._check_dependent_model_settings_wrt_nsos(nsos = nsos)
        
        self.assertEqual(len(ms.N), nsos, msg = 'length of N should equal number of elements returned from sos function')
    
    def test_value_N_wrt_nsos_for_non_default_N_and_nsos_equal_1(self):
        # create test data
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2])
        data.add_data_set(x,y)
        y = np.zeros([4,2])
        data.add_data_set(x,y)
        
        options = setup_simulation_options(nsimu = int(1000))
        ms = setup_model_settings(N = [2,4])
        
        # calculate dependencies
        ms._check_dependent_model_settings(data, options)
        
        nsos = 1
        ms._check_dependent_model_settings_wrt_nsos(nsos = nsos)
        
        self.assertEqual(ms.N, 6, msg = 'length of N should equal number of elements returned from sos function')

# -------------------------------------------
class ArrayType(unittest.TestCase):
    
    def test_none(self):
        self.assertTrue(MS._array_type(x = None) is None)
    
    def test_int(self):
        self.assertTrue(np.array_equal(MS._array_type(x = 3), np.array([3])))
        
    def test_float(self):
        self.assertTrue(np.array_equal(MS._array_type(x = 3.), np.array([3.])))
        
    def test_list(self):
        self.assertTrue(np.array_equal(MS._array_type(x = [3., 2.]), np.array([3., 2.])))
    
    def test_nparray(self):
        self.assertTrue(np.array_equal(MS._array_type(x = np.array([3., 2.])), np.array([3., 2.])))
        
    def test_dict(self):
        with self.assertRaises(SystemExit, msg = 'Dictionary not expected.'):
            MS._array_type(x = {'hello': 'world'})
            
# -------------------------------------------
class NumberOfObservations(unittest.TestCase):
    
    def test_array_equal(self):
        ms = ModelSettings()
        
        udN = np.ones([1,2])
        dsN = np.ones([1,2])
        
        N = ms._check_number_of_observations(udN = udN, dsN = dsN)
        
        self.assertTrue((N == udN).all() and (N == dsN).all())
        
    def test_len_dsN_gt_udN_case_1(self):
        ms = ModelSettings()
        udN = np.ones([1])
        dsN = np.ones([1,2])
        N = ms._check_number_of_observations(udN = udN, dsN = dsN)
        
        self.assertTrue((N == dsN).all())
        
    def test_len_dsN_gt_udN_case_2(self):
        ms = ModelSettings()
        udN = np.ones([1,2])
        dsN = np.ones([1,3])
        with self.assertRaises(SystemExit, msg = 'Mismatching dimensions'):
            ms._check_number_of_observations(udN = udN, dsN = dsN)
            
    def test_len_udN_gt_dsN_case_1(self):
        ms = ModelSettings()
        udN = np.ones([1,2])
        dsN = np.ones([1])
        N = ms._check_number_of_observations(udN = udN, dsN = dsN)
        
        self.assertTrue((N == dsN).all())
        
    def test_len_udN_gt_dsN_case_2(self):
        ms = ModelSettings()
        udN = np.ones([1,3])
        dsN = np.ones([1,2])
        with self.assertRaises(SystemExit, msg = 'Mismatching dimensions'):
            ms._check_number_of_observations(udN = udN, dsN = dsN)
            
    def test_len_udN_dsN_gt_1(self):
        ms = ModelSettings()
        udN = np.ones([1,3])
        dsN = np.zeros([1,3])
        with self.assertRaises(SystemExit, msg = 'Dimensions match, but values differ.'):
            ms._check_number_of_observations(udN = udN, dsN = dsN)
        
# -------------------------------------------
class DisplayModelSettings(unittest.TestCase):
    
    def test_print_these_none(self):
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2])
        data.add_data_set(x,y)
        
        options = SimulationOptions()
        
        ms = ModelSettings()
        ms.define_model_settings()
        ms._check_dependent_model_settings(data, options)
        ms._check_dependent_model_settings_wrt_nsos(nsos = 1)
        
        print_these = ms.display_model_settings(print_these = None)
        self.assertEqual(print_these, ['sos_function', 'model_function', 'sigma2', 'N', 'N0', 'S20', 'nsos', 'nbatch'], msg = 'Default print keys')
        
    def test_print_these_not_none(self):
        data = DataStructure()
        x = np.zeros([2])
        y = np.zeros([2])
        data.add_data_set(x,y)
        
        options = SimulationOptions()
        
        ms = ModelSettings()
        ms.define_model_settings()
        ms._check_dependent_model_settings(data, options)
        ms._check_dependent_model_settings_wrt_nsos(nsos = 1)
        
        print_these = ms.display_model_settings(print_these = ['model_function'])
        self.assertEqual(print_these, ['model_function'], msg = 'Specified print keys')
# -------------------------------------------