#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 14:30:05 2018

@author: prmiles
"""

from pymcmcstat.ParallelMCMC import ParallelMCMC
from pymcmcstat.ParallelMCMC import check_options_output, check_directory, check_initial_values
from pymcmcstat.ParallelMCMC import run_serial_simulation, assign_number_of_cores, generate_initial_values
from pymcmcstat.ParallelMCMC import check_shape_of_users_initial_values, check_users_initial_values_wrt_limits
from pymcmcstat.ParallelMCMC import get_parameter_features, unpack_mcmc_set
from pymcmcstat.samplers.utilities import is_sample_outside_bounds
from pymcmcstat.settings.ModelParameters import ModelParameters
import test.general_functions as gf
import unittest
from mock import patch
import os
import shutil
import numpy as np

def setup_par_mcmc_basic():
    mcstat = gf.basic_mcmc()
    tmpfolder = gf.generate_temp_folder()
    mcstat.simulation_options.savedir = tmpfolder
    return mcstat, tmpfolder
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

# -------------------------
class CheckInitialValues(unittest.TestCase):
    def test_check_initial_values_none(self):
        num_chain = 3
        npar = 3
        low_lim = np.zeros([3])
        upp_lim = np.ones([3])
        initial_values, num_chain_out = check_initial_values(initial_values = None, num_chain = num_chain, npar = npar, low_lim = low_lim, upp_lim = upp_lim)
        self.assertFalse(is_sample_outside_bounds(initial_values, low_lim, upp_lim), msg = 'Expect initial values to be inside bounds')
        self.assertEqual(initial_values.shape, (3,3), msg = 'Expect initial values array shape = (3,3)')
        self.assertEqual(num_chain_out, num_chain, msg = 'Expect num_chain to be unchanged')
        
    def test_check_shape_of_uiv(self):
        initial_values = np.random.random_sample(size = (4,3))
        num_chain = 4
        npar = 3
        low_lim = np.zeros([npar])
        upp_lim = np.ones([npar])
        out = check_initial_values(initial_values = initial_values, num_chain = num_chain, npar = npar, low_lim = low_lim, upp_lim = upp_lim)
        self.assertEqual(out[1], num_chain, msg = str('Expect no change to num_chain: {} neq {}'.format(out[1],num_chain)))
        self.assertTrue(np.array_equal(out[0], initial_values), msg = str('Expect no change to initial_values: {} neq {}'.format(out[0],initial_values)))
        
    def test_check_shape_of_uiv_not_same(self):
        initial_values = np.random.random_sample(size = (4,3))
        num_chain = 3
        npar = 2
        low_lim = np.zeros([npar])
        upp_lim = np.ones([npar])
        out = check_initial_values(initial_values = initial_values, num_chain = num_chain, npar = npar, low_lim = low_lim, upp_lim = upp_lim)
        self.assertEqual(out[1], 4, msg = str('Expect change to num_chain: {} neq {}'.format(out[1],4)))
        self.assertTrue(np.array_equal(out[0], initial_values[:,:npar]), msg = str('Expect change to initial_values: {} neq {}'.format(out[0],initial_values[:,:npar])))

# -------------------------
class GetParameterFeatures(unittest.TestCase):
    def test_get_parameter_features(self):
        MP = ModelParameters()
        MP.add_model_parameter(name = 'a1', theta0=0.1, minimum=0, maximum=1)
        MP.add_model_parameter(name = 'a2', theta0=0.2, minimum=0, maximum=1)
        MP.add_model_parameter(name = 'a3', theta0=0.3, minimum=0, maximum=1)
        npar, low_lim, upp_lim = get_parameter_features(MP.parameters)
        self.assertEqual(npar, 3, msg = 'Expect to find 3 parameters')
        self.assertTrue(np.array_equal(low_lim, np.zeros([3])), msg = str('Expect low_lim all zero: {} neq {}'.format(low_lim,np.zeros([3]))))
        self.assertTrue(np.array_equal(upp_lim, np.ones([3])), msg = str('Expect upp_lim all one: {} neq {}'.format(upp_lim,np.ones([3]))))
        
    def test_get_parameter_features_with_inf_bounds(self):
        MP = ModelParameters()
        MP.add_model_parameter(name = 'a1', theta0=0.1, minimum=-np.inf, maximum=1)
        MP.add_model_parameter(name = 'a2', theta0=0.2, minimum=0, maximum=1)
        MP.add_model_parameter(name = 'a3', theta0=0.3, minimum=0, maximum=np.inf)
        npar, low_lim, upp_lim = get_parameter_features(MP.parameters)
        expect_low_lim = np.array([0.1 - 100*0.1, 0, 0])
        expect_upp_lim = np.array([1, 1, 0.3 + 100*0.3])
        self.assertEqual(npar, 3, msg = 'Expect to find 3 parameters')
        self.assertTrue(np.array_equal(low_lim, expect_low_lim), msg = str('Expect low_lim finite: {} neq {}'.format(low_lim,expect_low_lim)))
        self.assertTrue(np.array_equal(upp_lim, expect_upp_lim), msg = str('Expect upp_lim finite: {} neq {}'.format(upp_lim,expect_upp_lim)))
        
# -------------------------
class UnpackMCMCSet(unittest.TestCase):
    def test_unpack_mcmc_set(self):
        mcstat = gf.basic_mcmc()
        data, options, model, parameters = unpack_mcmc_set(mcset = mcstat)
        self.assertEqual(data, mcstat.data, msg = 'Expect structures to match')
        self.assertEqual(options, mcstat.simulation_options, msg = 'Expect structures to match')
        self.assertEqual(model, mcstat.model_settings, msg = 'Expect structures to match')
        self.assertEqual(parameters, mcstat.parameters, msg = 'Expect structures to match')

# -------------------------
class ParallelMCMCInit(unittest.TestCase):
    def test_par_mc_init(self):
        PMC = ParallelMCMC()
        check_these = ['setup_parallel_simulation', 'run_parallel_simulation', 'display_individual_chain_statistics', 'description']
        for ct in check_these:
            self.assertTrue(hasattr(PMC, ct), msg = str('Expect class to have attribute: {}'.format(ct)))
            
# -------------------------
class SetupParallelMCMC(unittest.TestCase):
    def test_setup_parmc(self):
        PMC = ParallelMCMC()
        mcstat, tmpfolder = setup_par_mcmc_basic()
        PMC.setup_parallel_simulation(mcset = mcstat)
        self.assertEqual(PMC.num_cores, 1, msg = 'Default cores = 1')
        self.assertEqual(PMC.num_chain, 1, msg = 'Default chains = 1')
        self.assertTrue(isinstance(PMC.parmc, list), msg = 'Expect list')
        self.assertEqual(len(PMC.parmc), 1, msg = 'Expect length of 1')
        shutil.rmtree(tmpfolder)
        
    def test_setup_parmc_with_initial_values(self):
        PMC = ParallelMCMC()
        mcstat, tmpfolder = setup_par_mcmc_basic()
        initial_values = np.array([[0.5, 0.5, 0.5]])
        PMC.setup_parallel_simulation(mcset = mcstat, initial_values = initial_values)
        self.assertEqual(PMC.num_cores, 1, msg = 'Default cores = 1')
        self.assertEqual(PMC.num_chain, 1, msg = 'Default chains = 1')
        self.assertTrue(isinstance(PMC.parmc, list), msg = 'Expect list')
        self.assertEqual(len(PMC.parmc), 1, msg = 'Expect length of 1')
        self.assertTrue(np.array_equal(PMC.initial_values, initial_values), msg = str('Expect arrays to match: {} neq {}'.format(PMC.initial_values, initial_values)))
        shutil.rmtree(tmpfolder)
        
    @patch('pymcmcstat.ParallelMCMC.cpu_count', return_value = 2)
    def test_setup_parmc_with_initial_values_num_cores_chain_2(self, mock_cpu_count):
        PMC = ParallelMCMC()
        mcstat, tmpfolder = setup_par_mcmc_basic()
        initial_values = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
        PMC.setup_parallel_simulation(mcset = mcstat, initial_values = initial_values, num_cores = 2, num_chain = 2)
        self.assertEqual(PMC.num_cores, 2, msg = 'Default cores = 2')
        self.assertEqual(PMC.num_chain, 2, msg = 'Default chains = 2')
        self.assertTrue(isinstance(PMC.parmc, list), msg = 'Expect list')
        self.assertEqual(len(PMC.parmc), 2, msg = 'Expect length of 2')
        self.assertTrue(np.array_equal(PMC.initial_values, initial_values), msg = str('Expect arrays to match: {} neq {}'.format(PMC.initial_values, initial_values)))
        shutil.rmtree(tmpfolder)
# -------------------------
class RunParallelMCMC(unittest.TestCase):
    def test_setup_parmc(self):
        PMC = ParallelMCMC()
        mcstat, tmpfolder = setup_par_mcmc_basic()
        PMC.setup_parallel_simulation(mcset = mcstat)
        PMC.run_parallel_simulation()
        self.assertEqual(PMC.num_cores, 1, msg = 'Default cores = 1')
        self.assertEqual(PMC.num_chain, 1, msg = 'Default chains = 1')
        self.assertTrue(isinstance(PMC.parmc, list), msg = 'Expect list')
        self.assertEqual(len(PMC.parmc), 1, msg = 'Expect length of 1')
        self.assertTrue(hasattr(PMC.parmc[0], 'simulation_results'), msg = 'Expect results added')
        self.assertEqual(PMC.parmc[0].simulation_results.results['nsimu'], 5000, msg = 'Expect nsimu = 5000')
        shutil.rmtree(tmpfolder)