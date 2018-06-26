#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 08:33:47 2018

@author: prmiles
"""

from pymcmcstat.MCMC import MCMC
from pymcmcstat.structures.ParameterSet import ParameterSet
import unittest
from mock import patch
import io
import sys
import numpy as np

# define test model function
def modelfun(xdata, theta):
    m = theta[0]
    b = theta[1]
    nrow = xdata.shape[0]
    y = np.zeros([nrow,1])
    y[:,0] = m*xdata.reshape(nrow,) + b
    return y

def ssfun(theta, data, local = None):
    xdata = data.xdata[0]
    ydata = data.ydata[0]
    # eval model
    ymodel = modelfun(xdata, theta)
    # calc sos
    ss = sum((ymodel[:,0] - ydata[:,0])**2)
    return ss

def setup_mcmc(initialize = True):
    # Initialize MCMC object
    mcstat = MCMC()
    # Add data
    nds = 100
    x = np.linspace(2, 3, num=nds)
    y = 2.*x + 3. + 0.1*np.random.standard_normal(x.shape)
    mcstat.data.add_data_set(x, y)

    mcstat.simulation_options.define_simulation_options(nsimu = int(5.0e3), updatesigma = 1, method = 'dram', verbosity = 0)
    
    # update model settings
    mcstat.model_settings.define_model_settings(sos_function = ssfun)
    
    mcstat.parameters.add_model_parameter(name = 'm', theta0 = 2., minimum = -10, maximum = np.inf, sample = 1)
    mcstat.parameters.add_model_parameter(name = 'b', theta0 = -5., minimum = -10, maximum = 100, sample = 0)
    mcstat.parameters.add_model_parameter(name = 'b2', theta0 = -5., minimum = -10, maximum = 100, sample = 1)
    
    if initialize:
        mcstat._initialize_simulation()
    
    return mcstat

def setup_pseudo_results(initialize = True):
    mcstat = setup_mcmc(initialize = initialize)
    rejectedin = {'total': 10, 'in_adaptation_interval': 4, 'outside_bounds': 1}
    mcstat._MCMC__rejected = rejectedin.copy()
    mcstat._MCMC__simulation_time = 0.1
    return mcstat

# --------------------------
class MCMCInitialization(unittest.TestCase):
    
    def test_initialization(self):
        MC = MCMC()
        
        check_these = ['data', 'model_settings', 'simulation_options', 'parameters',
                       '_error_variance', '_covariance', '_sampling_methods', '_mcmc_status']
        for ct in check_these:
            self.assertTrue(hasattr(MC, ct), msg = str('Object missing attribute: {}'.format(ct)))
        
        self.assertFalse(MC._mcmc_status, msg = 'Status is False')

# --------------------------
class DisplayCurrentMCMCSettings(unittest.TestCase):
    
    def test_standard_print(self):
        mcstat = setup_mcmc()
        capturedOutput = io.StringIO()                  # Create StringIO object
        sys.stdout = capturedOutput                     #  and redirect stdout.
        mcstat._MCMC__display_current_mcmc_settings()
        sys.stdout = sys.__stdout__                     # Reset redirect.
        self.assertTrue(isinstance(capturedOutput.getvalue(), str), msg = 'Caputured string')
        
# --------------------------
class PrintRejectionStatistics(unittest.TestCase):
    def test_print_stats(self):
        rejected = {'total': 10, 'in_adaptation_interval': 4, 'outside_bounds': 1}
        isimu = 100
        iiadapt = 10
        verbosity = 3
        MC = MCMC()
        capturedOutput = io.StringIO()                  # Create StringIO object
        sys.stdout = capturedOutput                     #  and redirect stdout.
        MC._MCMC__print_rejection_statistics(rejected = rejected, isimu = isimu, iiadapt = iiadapt, verbosity = verbosity)
        sys.stdout = sys.__stdout__
        
        self.assertEqual(capturedOutput.getvalue(), str('i:{} ({},{},{})\n\n'.format(
                isimu, rejected['total']*isimu**(-1)*100, rejected['in_adaptation_interval']*iiadapt**(-1)*100,
                rejected['outside_bounds']*isimu**(-1)*100)), msg = 'Strings should match')
        
# --------------------------
class UpdateChain(unittest.TestCase):
    def test_chain_accepted(self):
        mcstat = setup_mcmc()
        accept = 1
        outsidebounds = 0
        CL = {'theta':np.array([1.0, 2.0]), 'ss': 1.0, 'prior':0.0, 'sigma2': 0.0}
        parset = ParameterSet(theta = CL['theta'], ss = CL['ss'], prior = CL['prior'], sigma2 = CL['sigma2'])
        
        mcstat._MCMC__chain_index = mcstat.simulation_options.nsimu - 1
        mcstat._MCMC__initialize_chains(chainind = 0, nsimu = mcstat.simulation_options.nsimu, npar = mcstat.parameters.npar, nsos = mcstat.model_settings.nsos, updatesigma = mcstat.simulation_options.updatesigma, sigma2 = mcstat.model_settings.sigma2)
        mcstat._MCMC__update_chain(accept = accept, new_set = parset, outsidebounds = outsidebounds)
        
        self.assertTrue(np.array_equal(mcstat._MCMC__chain[-1,:], parset.theta), msg = str('theta added to end of chain - {}'.format(mcstat._MCMC__chain[-1,:])))
        self.assertEqual(mcstat._MCMC__old_set, parset, msg = 'old_set updated to new set')
        
    def test_chain_not_accepted_within_bounds(self):
        mcstat = setup_mcmc()
        accept = 0
        outsidebounds = 0
        CL = {'theta':np.array([1.0, 2.0]), 'ss': 1.0, 'prior':0.0, 'sigma2': 0.0}
        parset = ParameterSet(theta = CL['theta'], ss = CL['ss'], prior = CL['prior'], sigma2 = CL['sigma2'])
        
        mcstat._MCMC__old_set = ParameterSet(theta = CL['theta'], ss = CL['ss'], prior = CL['prior'], sigma2 = CL['sigma2'])
        mcstat._MCMC__chain_index = mcstat.simulation_options.nsimu - 1
        mcstat._MCMC__initialize_chains(chainind = 0, nsimu = mcstat.simulation_options.nsimu, npar = mcstat.parameters.npar, nsos = mcstat.model_settings.nsos, updatesigma = mcstat.simulation_options.updatesigma, sigma2 = mcstat.model_settings.sigma2)
        mcstat._MCMC__rejected = {'total': 10, 'in_adaptation_interval': 4, 'outside_bounds': 1}
        mcstat._MCMC__update_chain(accept = accept, new_set = parset, outsidebounds = outsidebounds)
        
        self.assertTrue(np.array_equal(mcstat._MCMC__chain[-1,:], mcstat._MCMC__old_set.theta), msg = str('theta added to end of chain - {}'.format(mcstat._MCMC__chain[-1,:])))
        
# --------------------------
class UpdateRejected(unittest.TestCase):
    def test_update_rejection_stats_not_outsidebounds(self):
        mcstat = setup_mcmc()
        rejectedin = {'total': 10, 'in_adaptation_interval': 4, 'outside_bounds': 1}
        mcstat._MCMC__rejected = rejectedin.copy()
        mcstat._MCMC__update_rejected(outsidebounds = 0)
        
        self.assertEqual(mcstat._MCMC__rejected['total'], rejectedin['total']+1, msg = 'Adds one to counter')
        self.assertEqual(mcstat._MCMC__rejected['in_adaptation_interval'], rejectedin['in_adaptation_interval']+1, msg = 'Adds one to counter')
        self.assertEqual(mcstat._MCMC__rejected['outside_bounds'], rejectedin['outside_bounds'], msg = 'This counter stays the same.')
        
    def test_update_rejection_stats_outsidebounds(self):
        mcstat = setup_mcmc()
        rejectedin = {'total': 10, 'in_adaptation_interval': 4, 'outside_bounds': 1}
        mcstat._MCMC__rejected = rejectedin.copy()
        mcstat._MCMC__update_rejected(outsidebounds = 1)
        
        self.assertEqual(mcstat._MCMC__rejected['total'], rejectedin['total']+1, msg = 'Adds one to counter')
        self.assertEqual(mcstat._MCMC__rejected['in_adaptation_interval'], rejectedin['in_adaptation_interval']+1, msg = 'Adds one to counter')
        self.assertEqual(mcstat._MCMC__rejected['outside_bounds'], rejectedin['outside_bounds']+1, msg = 'Adds one to counter')
        
# --------------------------
class InitializeChain(unittest.TestCase):
    def test_initialize_chain_updatesigma_1(self):
        mcstat = setup_mcmc()
        CL = {'theta':np.array([1.0, 2.0]), 'ss': 1.0, 'prior':0.0, 'sigma2': 0.0}
        mcstat._MCMC__old_set = ParameterSet(theta = CL['theta'], ss = CL['ss'], prior = CL['prior'], sigma2 = CL['sigma2'])
        mcstat._MCMC__chain_index = mcstat.simulation_options.nsimu - 1
        mcstat._MCMC__initialize_chains(chainind = 0, nsimu = mcstat.simulation_options.nsimu, npar = mcstat.parameters.npar, nsos = mcstat.model_settings.nsos, updatesigma = mcstat.simulation_options.updatesigma, sigma2 = mcstat.model_settings.sigma2)
        self.assertEqual(mcstat._MCMC__chain.shape, (mcstat.simulation_options.nsimu, 2), msg = str('Shape should be (nsimu,2) -> {}'.format(mcstat._MCMC__chain.shape)))
        self.assertEqual(mcstat._MCMC__sschain.shape, (mcstat.simulation_options.nsimu, 1), msg = str('Shape should be (nsimu,1) -> {}'.format(mcstat._MCMC__sschain.shape)))
        self.assertEqual(mcstat._MCMC__s2chain.shape, (mcstat.simulation_options.nsimu, 1), msg = str('Shape should be (nsimu,1) -> {}'.format(mcstat._MCMC__s2chain.shape)))
        
    def test_initialize_chain_updatesigma_0(self):
        mcstat = setup_mcmc()
        mcstat.simulation_options.updatesigma = 0
        CL = {'theta':np.array([1.0, 2.0]), 'ss': 1.0, 'prior':0.0, 'sigma2': 0.0}
        mcstat._MCMC__old_set = ParameterSet(theta = CL['theta'], ss = CL['ss'], prior = CL['prior'], sigma2 = CL['sigma2'])
        mcstat._MCMC__chain_index = mcstat.simulation_options.nsimu - 1
        mcstat._MCMC__initialize_chains(chainind = 0, nsimu = mcstat.simulation_options.nsimu, npar = mcstat.parameters.npar, nsos = mcstat.model_settings.nsos, updatesigma = mcstat.simulation_options.updatesigma, sigma2 = mcstat.model_settings.sigma2)
        self.assertEqual(mcstat._MCMC__chain.shape, (mcstat.simulation_options.nsimu, 2), msg = str('Shape should be (nsimu,2) -> {}'.format(mcstat._MCMC__chain.shape)))
        self.assertEqual(mcstat._MCMC__sschain.shape, (mcstat.simulation_options.nsimu, 1), msg = str('Shape should be (nsimu,1) -> {}'.format(mcstat._MCMC__sschain.shape)))
        self.assertEqual(mcstat._MCMC__s2chain, None, msg = str('s2chain should be None -> {}'.format(mcstat._MCMC__s2chain)))
        
    def test_initialize_chain_updatesigma_1_nsos_2(self):
        mcstat = setup_mcmc()
        mcstat.model_settings.nsos = 2
        CL = {'theta':np.array([1.0, 2.0]), 'ss': 1.0, 'prior':0.0, 'sigma2': 0.0}
        mcstat._MCMC__old_set = ParameterSet(theta = CL['theta'], ss = CL['ss'], prior = CL['prior'], sigma2 = CL['sigma2'])
        mcstat._MCMC__chain_index = mcstat.simulation_options.nsimu - 1
        mcstat._MCMC__initialize_chains(chainind = 0, nsimu = mcstat.simulation_options.nsimu, npar = mcstat.parameters.npar, nsos = mcstat.model_settings.nsos, updatesigma = mcstat.simulation_options.updatesigma, sigma2 = mcstat.model_settings.sigma2)
        self.assertEqual(mcstat._MCMC__chain.shape, (mcstat.simulation_options.nsimu, 2), msg = str('Shape should be (nsimu,2) -> {}'.format(mcstat._MCMC__chain.shape)))
        self.assertEqual(mcstat._MCMC__sschain.shape, (mcstat.simulation_options.nsimu, 2), msg = str('Shape should be (nsimu,2) -> {}'.format(mcstat._MCMC__sschain.shape)))
        self.assertEqual(mcstat._MCMC__s2chain.shape, (mcstat.simulation_options.nsimu, 2), msg = str('Shape should be (nsimu,2) -> {}'.format(mcstat._MCMC__s2chain.shape)))
     
# --------------------------
class ExpandChain(unittest.TestCase):
    def test_expand_chain_updatesigma_1(self):
        mcstat = setup_mcmc()
        CL = {'theta':np.array([1.0, 2.0]), 'ss': 1.0, 'prior':0.0, 'sigma2': 0.0}
        mcstat._MCMC__old_set = ParameterSet(theta = CL['theta'], ss = CL['ss'], prior = CL['prior'], sigma2 = CL['sigma2'])
        mcstat._MCMC__chain_index = mcstat.simulation_options.nsimu - 1
        mcstat._MCMC__initialize_chains(chainind = 0, nsimu = mcstat.simulation_options.nsimu, npar = mcstat.parameters.npar, nsos = mcstat.model_settings.nsos, updatesigma = mcstat.simulation_options.updatesigma, sigma2 = mcstat.model_settings.sigma2)
        mcstat._MCMC__expand_chains(nsimu = mcstat.simulation_options.nsimu, npar = mcstat.parameters.npar, nsos = mcstat.model_settings.nsos, updatesigma = mcstat.simulation_options.updatesigma)
        self.assertEqual(mcstat._MCMC__chain.shape, (mcstat.simulation_options.nsimu*2 - 1, 2), msg = str('Shape should be (nsimu,2) -> {}'.format(mcstat._MCMC__chain.shape)))
        self.assertEqual(mcstat._MCMC__sschain.shape, (mcstat.simulation_options.nsimu*2 - 1, 1), msg = str('Shape should be (nsimu,1) -> {}'.format(mcstat._MCMC__sschain.shape)))
        self.assertEqual(mcstat._MCMC__s2chain.shape, (mcstat.simulation_options.nsimu*2 - 1, 1), msg = str('Shape should be (nsimu,1) -> {}'.format(mcstat._MCMC__s2chain.shape)))
        
    def test_expand_chain_updatesigma_0(self):
        mcstat = setup_mcmc()
        mcstat.simulation_options.updatesigma = 0
        CL = {'theta':np.array([1.0, 2.0]), 'ss': 1.0, 'prior':0.0, 'sigma2': 0.0}
        mcstat._MCMC__old_set = ParameterSet(theta = CL['theta'], ss = CL['ss'], prior = CL['prior'], sigma2 = CL['sigma2'])
        mcstat._MCMC__chain_index = mcstat.simulation_options.nsimu - 1
        mcstat._MCMC__initialize_chains(chainind = 0, nsimu = mcstat.simulation_options.nsimu, npar = mcstat.parameters.npar, nsos = mcstat.model_settings.nsos, updatesigma = mcstat.simulation_options.updatesigma, sigma2 = mcstat.model_settings.sigma2)
        mcstat._MCMC__expand_chains(nsimu = mcstat.simulation_options.nsimu, npar = mcstat.parameters.npar, nsos = mcstat.model_settings.nsos, updatesigma = mcstat.simulation_options.updatesigma)
        self.assertEqual(mcstat._MCMC__chain.shape, (mcstat.simulation_options.nsimu*2 - 1, 2), msg = str('Shape should be (nsimu,2) -> {}'.format(mcstat._MCMC__chain.shape)))
        self.assertEqual(mcstat._MCMC__sschain.shape, (mcstat.simulation_options.nsimu*2 - 1, 1), msg = str('Shape should be (nsimu,1) -> {}'.format(mcstat._MCMC__sschain.shape)))
        self.assertEqual(mcstat._MCMC__s2chain, None, msg = str('s2chain should be None -> {}'.format(mcstat._MCMC__s2chain)))
        
    def test_expand_chain_updatesigma_1_nsos_2(self):
        mcstat = setup_mcmc()
        mcstat.model_settings.nsos = 2
        CL = {'theta':np.array([1.0, 2.0]), 'ss': 1.0, 'prior':0.0, 'sigma2': 0.0}
        mcstat._MCMC__old_set = ParameterSet(theta = CL['theta'], ss = CL['ss'], prior = CL['prior'], sigma2 = CL['sigma2'])
        mcstat._MCMC__chain_index = mcstat.simulation_options.nsimu - 1
        mcstat._MCMC__initialize_chains(chainind = 0, nsimu = mcstat.simulation_options.nsimu, npar = mcstat.parameters.npar, nsos = mcstat.model_settings.nsos, updatesigma = mcstat.simulation_options.updatesigma, sigma2 = mcstat.model_settings.sigma2)
        mcstat._MCMC__expand_chains(nsimu = mcstat.simulation_options.nsimu, npar = mcstat.parameters.npar, nsos = mcstat.model_settings.nsos, updatesigma = mcstat.simulation_options.updatesigma)
        self.assertEqual(mcstat._MCMC__chain.shape, (mcstat.simulation_options.nsimu*2 - 1, 2), msg = str('Shape should be (nsimu,2) -> {}'.format(mcstat._MCMC__chain.shape)))
        self.assertEqual(mcstat._MCMC__sschain.shape, (mcstat.simulation_options.nsimu*2 - 1, 2), msg = str('Shape should be (nsimu,2) -> {}'.format(mcstat._MCMC__sschain.shape)))
        self.assertEqual(mcstat._MCMC__s2chain.shape, (mcstat.simulation_options.nsimu*2 - 1, 2), msg = str('Shape should be (nsimu,2) -> {}'.format(mcstat._MCMC__s2chain.shape)))
       
# --------------------------
class SetupSimulator(unittest.TestCase):
    def test_setup_simu_use_prev_false(self):
        mcstat = setup_mcmc(initialize=False)
        mcstat._MCMC__setup_simulator(use_previous_results = False)
        self.assertEqual(mcstat._MCMC__chain_index, 0, msg = 'Chain index should be 0')
        self.assertEqual(mcstat._MCMC__chain.shape, (mcstat.simulation_options.nsimu, 2), msg = str('Shape should be (nsimu,2) -> {}'.format(mcstat._MCMC__chain.shape)))
        self.assertEqual(mcstat._MCMC__sschain.shape, (mcstat.simulation_options.nsimu, 1), msg = str('Shape should be (nsimu,1) -> {}'.format(mcstat._MCMC__sschain.shape)))
        self.assertEqual(mcstat._MCMC__s2chain.shape, (mcstat.simulation_options.nsimu, 1), msg = str('Shape should be (nsimu,1) -> {}'.format(mcstat._MCMC__s2chain.shape)))
        
    def test_setup_simu_use_prev_true_causes_error(self):
        mcstat = setup_mcmc(initialize=False)
        with self.assertRaises(SystemExit, msg = 'No previous results exist'):
            mcstat._MCMC__setup_simulator(use_previous_results = True)
    
    def test_setup_simu_use_prev_true(self):
        mcstat = setup_pseudo_results(initialize = False)
        mcstat._MCMC__setup_simulator(use_previous_results = False)
        mcstat._MCMC__generate_simulation_results()
        mcstat._mcmc_status = True
        mcstat._MCMC__setup_simulator(use_previous_results = True)
        self.assertEqual(mcstat._MCMC__chain.shape, (mcstat.simulation_options.nsimu*2 - 1, 2), msg = str('Shape should be (nsimu,2) -> {}'.format(mcstat._MCMC__chain.shape)))
        self.assertEqual(mcstat._MCMC__sschain.shape, (mcstat.simulation_options.nsimu*2 - 1, 1), msg = str('Shape should be (nsimu,1) -> {}'.format(mcstat._MCMC__sschain.shape)))
        self.assertEqual(mcstat._MCMC__s2chain.shape, (mcstat.simulation_options.nsimu*2 - 1, 1), msg = str('Shape should be (nsimu,1) -> {}'.format(mcstat._MCMC__s2chain.shape)))
        
# --------------------------
class GenerateSimulationResults(unittest.TestCase):
    def test_results_generation_ntry_gt_1(self):
        mcstat = setup_pseudo_results(initialize = False)
        mcstat._MCMC__setup_simulator(use_previous_results = False)
        mcstat._MCMC__generate_simulation_results()
        results = mcstat.simulation_results.results
        self.assertTrue(mcstat.simulation_results.basic, msg = 'Basic successfully added if true')
        self.assertTrue(np.array_equal(results['R'], mcstat._covariance._R), msg = 'Arrays should match')
        self.assertTrue(np.array_equal(results['cov'], mcstat._covariance._covchain), msg = 'Arrays should match')
        check_for_these = ['simulation_options', 'model_settings', 'chain', 's2chain', 'sschain', 'drscale', 'iacce', 'RDR']
        for cft in check_for_these:
            self.assertTrue(cft in results, msg = str('{} assigned successfully'.format(cft)))