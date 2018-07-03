#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 14:30:05 2018

@author: prmiles
"""

from pymcmcstat.ParallelMCMC import ParallelMCMC
from pymcmcstat.ParallelMCMC import check_options_output, check_directory
from pymcmcstat.ParallelMCMC import run_serial_simulation, assign_number_of_cores, generate_initial_values
from pymcmcstat.ParallelMCMC import check_shape_of_users_initial_values, check_users_initial_values_wrt_limits
from pymcmcstat.samplers.utilities import is_sample_outside_bounds
import test.general_functions as gf
import unittest
from mock import patch
import os
import shutil
import numpy as np

# -------------------------
class CheckOptionsOutput(unittest.TestCase):
    def test_check_options_output(self):
        mcstat = gf.basic_mcmc()
        mcstat.simulation_options.save_to_bin = True
        mcstat.simulation_options.save_to_txt = True
        options = check_options_output(options = mcstat.simulation_options)
        self.assertTrue(options.save_to_bin, msg = 'Expect True')
        self.assertTrue(options.save_to_txt, msg = 'Expect True')
        
    def test_check_options_output_with_bin_false(self):
        mcstat = gf.basic_mcmc()
        mcstat.simulation_options.save_to_bin = False
        mcstat.simulation_options.save_to_txt = True
        options = check_options_output(options = mcstat.simulation_options)
        self.assertFalse(options.save_to_bin, msg = 'Expect False')
        self.assertTrue(options.save_to_txt, msg = 'Expect True')
        
    def test_check_options_output_with_txt_false(self):
        mcstat = gf.basic_mcmc()
        mcstat.simulation_options.save_to_bin = True
        mcstat.simulation_options.save_to_txt = False
        options = check_options_output(options = mcstat.simulation_options)
        self.assertTrue(options.save_to_bin, msg = 'Expect True')
        self.assertFalse(options.save_to_txt, msg = 'Expect False')
        
    def test_check_options_output_with_both_false(self):
        mcstat = gf.basic_mcmc()
        mcstat.simulation_options.save_to_bin = False
        mcstat.simulation_options.save_to_txt = False
        options = check_options_output(options = mcstat.simulation_options)
        self.assertTrue(options.save_to_bin, msg = 'Expect True')
        self.assertFalse(options.save_to_txt, msg = 'Expect False')
        
# -------------------------
class CheckDirectory(unittest.TestCase):
    def test_check_directory(self):
        tmpfolder = gf.generate_temp_folder()
        check_directory(tmpfolder)
        self.assertTrue(os.path.exists(tmpfolder), msg = 'Directory should exist')
        shutil.rmtree(tmpfolder)
        
# -------------------------
class RunSerialSimulation(unittest.TestCase):
    def test_run_serial_simulation(self):
        mcstat = gf.basic_mcmc()
        mcstat.simulation_options.nsimu = 100
        mcstat.simulation_options.save_to_bin = False
        mcstat.simulation_options.save_to_txt = False
        simulation_results = run_serial_simulation(mcstat)
        results = simulation_results.results
        self.assertTrue(isinstance(results, dict), msg = 'Expect dictionary return item')
        self.assertEqual(results['nsimu'], 100, msg = 'Expect 100 simulations')
        
# -------------------------
class AssignNumberOfCores(unittest.TestCase):
    @patch('pymcmcstat.ParallelMCMC.cpu_count', return_value = 4)
    def test_assign_number_of_cores(self, mock_cpu_count):
        for nc in range(4):
            num_cores = assign_number_of_cores(num_cores = nc)
            self.assertEqual(num_cores, nc, msg = str('Expect num_cores = {}'.format(nc))) 
            
    @patch('pymcmcstat.ParallelMCMC.cpu_count', return_value = 2)
    def test_assign_number_of_cores_gte_cpu_count(self, mock_cpu_count):
        for nc in range(2, 4):
            num_cores = assign_number_of_cores(num_cores = nc)
            self.assertEqual(num_cores, 2, msg = str('Expect num_cores = {}'.format(2))) 
            
# -------------------------
class CheckShapeOfUsersInitialValues(unittest.TestCase):
    def test_check_shape_of_uiv(self):
        initial_values = np.random.random_sample(size = (4,3))
        num_chain = 4
        npar = 3
        out = check_shape_of_users_initial_values(initial_values = initial_values, num_chain = num_chain, npar = npar)
        self.assertEqual(out[0], num_chain, msg = str('Expect no change to num_chain: {} neq {}'.format(out[0],num_chain)))
        self.assertTrue(np.array_equal(out[1], initial_values), msg = str('Expect no change to initial_values: {} neq {}'.format(out[1],initial_values)))
        
    def test_check_shape_of_uiv_not_same(self):
        initial_values = np.random.random_sample(size = (4,3))
        num_chain = 3
        npar = 2
        out = check_shape_of_users_initial_values(initial_values = initial_values, num_chain = num_chain, npar = npar)
        self.assertEqual(out[0], 4, msg = str('Expect change to num_chain: {} neq {}'.format(out[0],4)))
        self.assertTrue(np.array_equal(out[1], initial_values[:,:npar]), msg = str('Expect change to initial_values: {} neq {}'.format(out[1],initial_values[:,:npar])))
        
# -------------------------
class CheckUsersInitialValuesWRTLimits(unittest.TestCase):
    @patch('pymcmcstat.ParallelMCMC.is_sample_outside_bounds', return_value = False)
    def test_initial_values_wrt_limits_false(self, mock_outside):
        initial_values = np.random.random_sample(size = (4,3))
        low_lim = np.random.random_sample(size = (4,3))
        upp_lim = np.random.random_sample(size = (4,3))
        out = check_users_initial_values_wrt_limits(initial_values = initial_values, low_lim = low_lim, upp_lim = upp_lim)
        self.assertTrue(np.array_equal(out, initial_values), msg = str('Expect no change to initial_values: {} neq {}'.format(out,initial_values)))

    @patch('pymcmcstat.ParallelMCMC.is_sample_outside_bounds', return_value = True)
    def test_initial_values_wrt_limits_true(self, mock_outside):
        initial_values = np.random.random_sample(size = (4,3))
        low_lim = np.random.random_sample(size = (4,3))
        upp_lim = np.random.random_sample(size = (4,3))
        with self.assertRaises(SystemExit, msg = 'Outside bounds not acceptable for initial values'):
            check_users_initial_values_wrt_limits(initial_values = initial_values, low_lim = low_lim, upp_lim = upp_lim)
# -------------------------
class GenerateInitialValues(unittest.TestCase):
    def test_generate_initial_values(self):
        num_chain = 3
        npar = 3
        low_lim = np.zeros([3])
        upp_lim = np.ones([3])
        initial_values = generate_initial_values(num_chain = num_chain, npar = npar, low_lim = low_lim, upp_lim = upp_lim)
        self.assertFalse(is_sample_outside_bounds(initial_values, low_lim, upp_lim), msg = 'Expect initial values to be inside bounds')