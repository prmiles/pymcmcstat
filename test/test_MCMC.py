#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 08:33:47 2018

@author: prmiles
"""

from pymcmcstat.MCMC import MCMC
from pymcmcstat.structures.ParameterSet import ParameterSet
import unittest
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

def setup_mcmc():
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
    
    mcstat._initialize_simulation()
    
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
class MessageDisplay(unittest.TestCase):
    
    def test_standard_print(self):
        MC = MCMC()
        capturedOutput = io.StringIO()                  # Create StringIO object
        sys.stdout = capturedOutput                     #  and redirect stdout.
        flag = MC._MCMC__message(verbosity = 1, level = 0, printthis = 'test')
        sys.stdout = sys.__stdout__                     # Reset redirect.
        self.assertEqual(capturedOutput.getvalue(), 'test\n', msg = 'Expected string')
        self.assertTrue(flag, msg = 'Statement was printed')
        
    def test_no_print(self):
        MC = MCMC()
        capturedOutput = io.StringIO()                  # Create StringIO object
        sys.stdout = capturedOutput                     #  and redirect stdout.
        flag = MC._MCMC__message(verbosity = 0, level = 1, printthis = 'test')
        sys.stdout = sys.__stdout__                     # Reset redirect.
        self.assertFalse(flag, msg = 'Statement not printed because verbosity less than level')
        
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
        mcstat._MCMC__initialize_chains(chainind = 0)
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
        mcstat._MCMC__initialize_chains(chainind = 0)
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