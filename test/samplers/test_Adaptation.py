#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 08:26:50 2018

@author: prmiles
"""

from pymcmcstat.MCMC import MCMC
from pymcmcstat.samplers.Adaptation import Adaptation
from pymcmcstat.samplers.Adaptation import is_semi_pos_def_chol
from pymcmcstat.samplers.Adaptation import unpack_simulation_options, unpack_covariance_settings
from pymcmcstat.samplers.Adaptation import below_burnin_threshold
from pymcmcstat.samplers.Adaptation import update_delayed_rejection
from pymcmcstat.samplers.Adaptation import update_cov_via_ram
from pymcmcstat.samplers.Adaptation import scale_cholesky_decomposition
from pymcmcstat.samplers.Adaptation import adjust_cov_matrix
from pymcmcstat.settings.SimulationOptions import SimulationOptions
from pymcmcstat.procedures.CovarianceProcedures import CovarianceProcedures
import unittest
from mock import patch
import numpy as np
import math
import io
import sys

def setup_options(**kwargs):
    SO = SimulationOptions()
    SO.define_simulation_options(**kwargs)
    return SO

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
    
    # extract components
    model = mcstat.model_settings
    options = mcstat.simulation_options
    parameters = mcstat.parameters
    data = mcstat.data
    return model, options, parameters, data

# --------------------------------------------
class Initialization(unittest.TestCase):
    def test_init_adapt(self):
        AD = Adaptation()
        ADD = AD.__dict__
        check_fields = ['qcov', 'qcov_scale', 'R', 'qcov_original', 'invR', 'iacce', 'covchain', 'meanchain']
        for ii, cf in enumerate(check_fields):
            self.assertTrue(ADD[cf] is None, msg = str('Initialize {} to None'.format(cf)))
# --------------------------------------------            
class CheckForSingularCovMatrix(unittest.TestCase):
    def test_not_singular(self):
        AD = Adaptation()
        
        upcov = np.ones([2,2])
        Rin = np.diag([2, 2])
        npar = 2
        qcov_adjust = 1e-8
        qcov_scale = 2.4*(math.sqrt(npar)**(-1)) # scale factor in R
        rejected = {'in_adaptation_interval': 10}
        iiadapt = 100
        verbosity = 0
        
        R = AD.check_for_singular_cov_matrix(upcov = upcov, R = Rin, npar = npar, qcov_adjust = qcov_adjust, qcov_scale = qcov_scale, rejected = rejected, iiadapt = iiadapt, verbosity = verbosity)
        self.assertTrue(isinstance(R, np.ndarray), msg = 'Expect numpy array output')
        self.assertEqual(R.shape, (2,2), msg = 'Expect 2x2 array')
        
    def test_singular(self):
        AD = Adaptation()
        
        upcov = np.array([[2, -1],[0, 0]])
        Rin = np.diag([2, 2])
        npar = 2
        qcov_adjust = 1e-8
        qcov_scale = 2.4*(math.sqrt(npar)**(-1)) # scale factor in R
        rejected = {'in_adaptation_interval': 10}
        iiadapt = 100
        verbosity = 0
        
        R = AD.check_for_singular_cov_matrix(upcov = upcov, R = Rin, npar = npar, qcov_adjust = qcov_adjust, qcov_scale = qcov_scale, rejected = rejected, iiadapt = iiadapt, verbosity = verbosity)
        self.assertTrue(isinstance(R, np.ndarray), msg = 'Expect numpy array output')
        self.assertEqual(R.shape, (2,2), msg = 'Expect 2x2 array')
# --------------------------------------------        
class IsSemiPositiveDefinite(unittest.TestCase):
    def test_semipositivedef(self):
        mtx = np.diag([2, 2])
        flag, c = is_semi_pos_def_chol(x = mtx)
        self.assertTrue(flag, msg = 'Expect true')
        self.assertTrue(isinstance(c, np.ndarray), msg = 'Expect numpy array')
        self.assertEqual(c.shape, (2,2), msg = 'Expect 2x2 array')
        
    def test_notsemipositivedef(self):
        mtx = np.array([[2, -1],[0, 0]])
        flag, c = is_semi_pos_def_chol(x = mtx)
        self.assertFalse(flag, msg = 'Expect false')
# --------------------------------------------        
class UnpackSimulationOptions(unittest.TestCase):
    def test_unpack_options(self):
        SO = setup_options()
        burnintime, burninscale, ntry, drscale, alphatarget, etaparam, qcov_adjust, doram, verbosity = unpack_simulation_options(options = SO)
        defaults = {'burnintime': 0, 'burninscale': 10, 'ntry': 2, 'drscale': np.array([5, 4, 3], dtype = float),
                    'alphatarget': 0.234, 'etaparam': 0.7, 'qcov_adjust': 1e-8, 'doram': 0, 'verbosity': 1}
        
        self.assertEqual(burnintime, defaults['burnintime'], msg = str('Expected {} = {}'.format('burnintime', defaults['burnintime'])))
        self.assertEqual(burninscale, defaults['burninscale'], msg = str('Expected {} = {}'.format('burninscale', defaults['burninscale'])))
        self.assertEqual(ntry, defaults['ntry'], msg = str('Expected {} = {}'.format('ntry', defaults['ntry'])))
        self.assertTrue(np.array_equal(drscale, defaults['drscale']), msg = str('Expected {} = {}'.format('drscale', defaults['drscale'])))
        self.assertEqual(alphatarget, defaults['alphatarget'], msg = str('Expected {} = {}'.format('alphatarget', defaults['alphatarget'])))
        self.assertEqual(etaparam, defaults['etaparam'], msg = str('Expected {} = {}'.format('etaparam', defaults['etaparam'])))
        self.assertEqual(qcov_adjust, defaults['qcov_adjust'], msg = str('Expected {} = {}'.format('qcov_adjust', defaults['qcov_adjust'])))
        self.assertEqual(doram, defaults['doram'], msg = str('Expected {} = {}'.format('doram', defaults['doram'])))
        self.assertEqual(verbosity, defaults['verbosity'], msg = str('Expected {} = {}'.format('verbosity', defaults['verbosity'])))
        
# --------------------------------------------        
class UnpackCovarianceSettings(unittest.TestCase):
    def test_unpack_covariance(self):
        model, options, parameters, data = setup_mcmc()
        CP = CovarianceProcedures()
        CP._initialize_covariance_settings(parameters = parameters, options = options)
        
        last_index_since_adaptation, R, oldcovchain, oldmeanchain, oldwsum, no_adapt_index, qcov_scale, qcov = unpack_covariance_settings(covariance = CP)     
        
        out = {'last_index_since_adaptation': last_index_since_adaptation, 'R': R, 'oldcovchain': oldcovchain, 'oldmeanchain': oldmeanchain,
                    'oldwsum': oldwsum, 'no_adapt_index': no_adapt_index, 'qcov_scale': qcov_scale, 'qcov': qcov}
        
        defaults = {'last_index_since_adaptation': 0, 'R': np.array([[0.1, 0.],[0., 0.25]]), 'oldcovchain': None, 'oldmeanchain': None,
                    'oldwsum': None, 'no_adapt_index': np.array([False, False]), 'qcov_scale': 2.4/np.sqrt(2), 'qcov': np.square(np.array([[0.1, 0.],[0., 0.25]]))}

        check_these = ['last_index_since_adaptation', 'oldcovchain', 'oldmeanchain', 'oldwsum']        
        for item in check_these:
            self.assertEqual(out[item], defaults[item], msg = str('Expected {} = {} ? {}'.format(item, defaults[item], out[item])))
            
        check_these = ['R', 'no_adapt_index', 'qcov_scale', 'qcov']        
        for item in check_these:
            self.assertTrue(np.array_equal(out[item], defaults[item]), msg = str('Expected {} = {} ? {}'.format(item, defaults[item], out[item])))
        
# --------------------------------------------
class BelowBurninThreshold(unittest.TestCase):
    def test_below_burnin_reject_gt_95(self):
        rejected = {'in_adaptation_interval': 96}
        iiadapt = 100
        R = np.array([[0.1, 0.],[0., 0.25]])
        burninscale = 0.5
        verbosity = 10
        capturedOutput = io.StringIO()                  # Create StringIO object
        sys.stdout = capturedOutput                     #  and redirect stdout.
        Rout = below_burnin_threshold(rejected = rejected, iiadapt = iiadapt, R = R, burninscale = burninscale, verbosity = verbosity)
        sys.stdout = sys.__stdout__

        self.assertEqual(capturedOutput.getvalue(), ' (burnin/down) 96.0\n', msg = 'Expected string {}'.format(' (burnin/down) 96.0\n'))
        self.assertTrue(np.array_equal(Rout, R/burninscale), msg = str('Expected R = {}'.format(R/burninscale)))
    
    def test_below_burnin_reject_lt_95(self):
        rejected = {'in_adaptation_interval': 4}
        iiadapt = 100
        R = np.array([[0.1, 0.],[0., 0.25]])
        burninscale = 0.5
        verbosity = 10
        capturedOutput = io.StringIO()                  # Create StringIO object
        sys.stdout = capturedOutput                     #  and redirect stdout.
        Rout = below_burnin_threshold(rejected = rejected, iiadapt = iiadapt, R = R, burninscale = burninscale, verbosity = verbosity)
        sys.stdout = sys.__stdout__

        self.assertEqual(capturedOutput.getvalue(), ' (burnin/up) 4.0\n', msg = 'Expected string {}'.format(' (burnin/up) 4.0\n'))
        self.assertTrue(np.array_equal(Rout, R*burninscale), msg = str('Expected R = {}'.format(R/burninscale)))
        
# --------------------------------------------
class UpdateDelayedRejection(unittest.TestCase):
    def test_update_dr_ntry_1(self):
        RDR, invR = update_delayed_rejection(R = None, npar = 2, ntry = 1, drscale = 0)
        self.assertEqual(RDR, None, msg = 'Expect None')
        self.assertEqual(invR, None, msg = 'Expect None')
        
    def test_update_dr_ntry_3(self):
        R = np.array([[0.1, 0.],[0., 0.25]])
        npar = 2
        ntry = 3
        drscale = np.array([5,4,3], dtype = float)
        RDR, invR = update_delayed_rejection(R = R, npar = npar, ntry = ntry, drscale = drscale)
        self.assertTrue(isinstance(RDR, list), msg = 'Expect list return')
        self.assertTrue(isinstance(invR, list), msg = 'Expect list return')
        self.assertTrue(np.array_equal(RDR[0], R), msg = 'Expect arrays to match')
        self.assertTrue(np.array_equal(invR[0], np.linalg.solve(R, np.eye(npar))), msg = 'Expect arrays to match')
        self.assertTrue(np.array_equal(RDR[1], RDR[0]*(drscale[0]**(-1))), msg = str('Expect arrays to match: {} neq {}'.format(RDR[1], RDR[0]/drscale[0])))
        self.assertTrue(np.array_equal(invR[1], invR[0]*drscale[0]), msg = str('Expect arrays to match: {} neq {}'.format(invR[1], invR[0]*drscale[0])))
        self.assertTrue(np.array_equal(RDR[2], RDR[1]*(drscale[1]**(-1))), msg = str('Expect arrays to match: {} neq {}'.format(RDR[2], RDR[1]/drscale[1])))
        self.assertTrue(np.array_equal(invR[2], invR[1]*drscale[1]), msg = str('Expect arrays to match: {} neq {}'.format(invR[2], invR[1]*drscale[1])))
        
# --------------------------------------------
class UpdateCovViaRam(unittest.TestCase):
    def test_update_via_ram(self):
        u = np.array([0.1, 0.5])
        isimu = 100
        etaparam = 0.7
        npar = 2
        alphatarget = 0.234
        alpha = 0.7
        R = np.array([[0.1, 0.],[0., 0.25]])
        upcov = update_cov_via_ram(u = u, isimu = isimu, etaparam = etaparam, npar = npar, alphatarget = alphatarget, alpha = alpha, R = R)
        
        self.assertEqual(upcov.shape, (2,2), msg = 'Expect shape = (2,2)')
        
        uu = u*(np.linalg.norm(u)**(-1))
        eta = (isimu**(etaparam))**(-1)
        ram = np.eye(npar) + eta*(min(1.0, alpha) - alphatarget)*(np.dot(uu.transpose(), uu))
        self.assertTrue(np.array_equal(upcov, np.dot(np.dot(R.transpose(),ram),R)), msg = str('Expect arrays to match: {} neq {}'.format(upcov, np.dot(np.dot(R.transpose(),ram),R))))
        
# --------------------------------------------
class ScaleCholeskyDecomposition(unittest.TestCase):
    def test_scale_chol_dec(self):
        Ra = np.random.random_sample(size = (2,2))
        qcov_scale = 0.2
        R = scale_cholesky_decomposition(Ra = Ra, qcov_scale = qcov_scale)
        self.assertTrue(np.array_equal(R, Ra*qcov_scale), msg = 'Expect arrays to match')
        
# --------------------------------------------
class AdjustCovMatrix(unittest.TestCase):
    def test_singular_cov_mat_successfully_adjusted(self):
        upcov = np.array([[0.1, 0.4],[0.4, 0.2]])
        R = np.array([[0.1, 0.],[0., 0.25]])
        npar = 2
        qcov_adjust = 0.9
        qcov_scale = 0.3
        rejected = {'in_adaptation_interval': 96}
        iiadapt = 100
        verbosity = 10
        
        capturedOutput = io.StringIO()                  # Create StringIO object
        sys.stdout = capturedOutput                     #  and redirect stdout.
        Rout = adjust_cov_matrix(upcov = upcov, R = R, npar = npar, qcov_adjust = qcov_adjust, qcov_scale = qcov_scale, rejected = rejected, iiadapt = iiadapt, verbosity = verbosity)
        sys.stdout = sys.__stdout__
        self.assertEqual(capturedOutput.getvalue(), 'adjusted covariance matrix\n', msg = 'Expected string {}'.format('adjusted covariance matrix\n'))
        tmp = upcov + np.eye(npar)*qcov_adjust
        pos_def_adjust, pRa = is_semi_pos_def_chol(tmp)
        self.assertTrue(pos_def_adjust, msg = 'Expect True')
        self.assertTrue(np.array_equal(pRa*qcov_scale, Rout), msg = str('Expect arrays to match: {} neq {}'.format(pRa*qcov_scale, Rout)))
        
    @patch('pymcmcstat.samplers.Adaptation.is_semi_pos_def_chol', return_value = (False, None))
    def test_singular_cov_mat_not_successfully_adjusted(self, mock_is):
        upcov = np.array([[0.1, 0.4],[0.4, 0.0]])
        R = np.array([[0.1, 0.],[0., 0.25]])
        npar = 2
        qcov_adjust = 0.9
        qcov_scale = 0.3
        rejected = {'in_adaptation_interval': 96}
        iiadapt = 100
        verbosity = 10
        
        capturedOutput = io.StringIO()                  # Create StringIO object
        sys.stdout = capturedOutput                     #  and redirect stdout.
        Rout = adjust_cov_matrix(upcov = upcov, R = R, npar = npar, qcov_adjust = qcov_adjust, qcov_scale = qcov_scale, rejected = rejected, iiadapt = iiadapt, verbosity = verbosity)
        sys.stdout = sys.__stdout__
        self.assertEqual(capturedOutput.getvalue(), 'covariance matrix singular, no adaptation 96.0\n', msg = 'Expected string {}'.format('covariance matrix singular, no adaptation 96.0\n'))
        self.assertTrue(np.array_equal(R, Rout), msg = str('Expect arrays to match: {} neq {}'.format(R, Rout)))