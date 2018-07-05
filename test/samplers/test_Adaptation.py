#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 08:26:50 2018

@author: prmiles
"""

from pymcmcstat.samplers.Adaptation import cholupdate, initialize_covariance_mean_sum
from pymcmcstat.samplers.Adaptation import Adaptation
from pymcmcstat.samplers.Adaptation import is_semi_pos_def_chol, update_covariance_mean_sum
from pymcmcstat.samplers.Adaptation import unpack_simulation_options, unpack_covariance_settings
from pymcmcstat.samplers.Adaptation import below_burnin_threshold
from pymcmcstat.samplers.Adaptation import update_delayed_rejection
from pymcmcstat.samplers.Adaptation import update_cov_via_ram
from pymcmcstat.samplers.Adaptation import scale_cholesky_decomposition
from pymcmcstat.samplers.Adaptation import adjust_cov_matrix
from pymcmcstat.samplers.Adaptation import check_for_singular_cov_matrix
from pymcmcstat.samplers.Adaptation import update_cov_from_covchain
from pymcmcstat.samplers.Adaptation import setup_w_R, setup_cholupdate
from pymcmcstat.settings.SimulationOptions import SimulationOptions
from pymcmcstat.procedures.CovarianceProcedures import CovarianceProcedures
import test.general_functions as gf
import unittest
from mock import patch
import numpy as np
import io
import sys

def setup_options(**kwargs):
    SO = SimulationOptions()
    SO.define_simulation_options(**kwargs)
    return SO

def setup_run_adapt_settings():
    __, options, parameters, data = gf.setup_mcmc()
    CP = CovarianceProcedures()
    CP._initialize_covariance_settings(parameters = parameters, options = options)
    rejected = {'in_adaptation_interval': 4, 'total': 10, 'outside_bounds': 1}
    isimu = 100
    iiadapt = 100
    chain = np.zeros([100,2])
    chain[:,0] = np.linspace(1,100,100)
    chain[:,1] = np.linspace(1,100,100)
    chainind = 100
    u = np.random.random_sample(size = (1,2))
    npar = 2
    alpha = 0.78
    return CP, options, isimu, iiadapt, rejected, chain, chainind, u, npar, alpha

# -------------------------------------------
class RunAdaptation(unittest.TestCase):
    def test_run_adapt_isimu_lt_burnintime(self):
        covariance, options, isimu, iiadapt, rejected, chain, chainind, u, npar, alpha = setup_run_adapt_settings()
        options.burnintime = 1000
        A = Adaptation()
        
        rejected['in_adaptation_interval'] =  96
        tstR = np.array([[0.1, 0.],[0., 0.25]])
        covariance._R = tstR.copy()
        options.burnin_scale = 0.5
        options.verbosity = 10
        capturedOutput = io.StringIO()                  # Create StringIO object
        sys.stdout = capturedOutput                     #  and redirect stdout.
        cout = A.run_adaptation(covariance, options, isimu, iiadapt, rejected, chain, chainind, u, npar, alpha)
        sys.stdout = sys.__stdout__

        self.assertEqual(capturedOutput.getvalue(), ' (burnin/down) 96.0\n', msg = 'Expected string {}'.format(' (burnin/down) 96.0\n'))
        self.assertTrue(np.isclose(cout._R, tstR/options.burnin_scale).all(), msg = str('Expect arrays to match within numerical precision: {} new {}'.format(cout._R, tstR/options.burnin_scale)))
        
    def test_run_adapt_update_covariance_mean_sum_R_is_not_none(self):
        covariance, options, isimu, iiadapt, rejected, chain, chainind, u, npar, alpha = setup_run_adapt_settings()
        
        oldcov = np.array([[0.5, 0.1],[0.1, 0.3]])
        oldmean = np.array([10.2, 2.4])
        oldwsum = 100*np.ones([1])
        oldR = np.array([[0.3, 0.1],[0, 0.4]])
        
        covariance._R = oldR.copy()
        covariance._wsum = oldwsum.copy()
        covariance._meanchain = oldmean.copy()
        covariance._covchain = oldcov.copy()
        
        A = Adaptation()
        cout = A.run_adaptation(covariance, options, isimu, iiadapt, rejected, chain, chainind, u, npar, alpha)
        
        tsmtx = np.array([[827.030150753768,	905.811055276382],[905.811055276382,	1000.17688442211]])
        tsRmtx = np.array([[48.8041682048865, 53.4531359748650],[0.,	4.82407313255805]])
        tsmean = np.array([30.3500000000000,	26.4500000000000])
        self.assertTrue(np.isclose(tsmtx, cout._covchain).all(), msg = str('Mean algorithms should agree: {} neq {}'.format(tsmtx, cout._covchain)))
        self.assertTrue(np.isclose(tsmean, cout._meanchain).all(), msg = str('Mean algorithms should agree: {} neq {}'.format(tsmean, cout._meanchain)))
        self.assertTrue(np.isclose(tsRmtx, cout._R).all(), msg = str('Mean algorithms should agree: {} neq {}'.format(tsRmtx, cout._R)))
        
    @patch('pymcmcstat.samplers.Adaptation.update_cov_via_ram', return_value = np.array([[0.26917331, 0.09473752],[0.09473752, 0.49657009]]))
    def test_run_adapt_update_covariance_mean_sum_doram_is_true(self, mock_ram):
        covariance, options, isimu, iiadapt, rejected, chain, chainind, u, npar, alpha = setup_run_adapt_settings()
        
        options.doram = True
        
        oldcov = np.array([[0.5, 0.1],[0.1, 0.3]])
        oldmean = np.array([10.2, 2.4])
        oldwsum = 100*np.ones([1])
        oldR = np.array([[0.3, 0.1],[0, 0.4]])
        
        covariance._R = oldR.copy()
        covariance._wsum = oldwsum.copy()
        covariance._meanchain = oldmean.copy()
        covariance._covchain = oldcov.copy()
        
        A = Adaptation()
        cout = A.run_adaptation(covariance, options, isimu, iiadapt, rejected, chain, chainind, u, npar, alpha)
        
        tsmtx = np.array([[827.030150753768,	905.811055276382],[905.811055276382,	1000.17688442211]])
        tsRmtx = np.array([[0.880465293353463,	0.309886215458656],[0.,	1.15502917394701]])
        tsmean = np.array([30.3500000000000,	26.4500000000000])
        self.assertTrue(np.isclose(tsmtx, cout._covchain).all(), msg = str('Mean algorithms should agree: {} neq {}'.format(tsmtx, cout._covchain)))
        self.assertTrue(np.isclose(tsmean, cout._meanchain).all(), msg = str('Mean algorithms should agree: {} neq {}'.format(tsmean, cout._meanchain)))
        self.assertTrue(np.isclose(tsRmtx, cout._R).all(), msg = str('Mean algorithms should agree: {} neq {}'.format(tsRmtx, cout._R)))
   
# --------------------------------------------
class CholUpdate(unittest.TestCase):
    def test_cholupate(self):
        R = np.diag([1., 1., 1.])
        R1 = cholupdate(R = R, x = np.array([1,1,1], dtype = float))
        tsmtx = np.array([[1.41421356237310,	0.707106781186548, 0.707106781186548],[0,	1.22474487139159,	0.408248290463863],[0,	0,	1.15470053837925]])
        self.assertTrue(np.isclose(R1, tsmtx).all(), msg = str('Expect arrays to match within numerical precision: {} new {}'.format(R1, tsmtx)))

# --------------------------------------------
class SetupWR(unittest.TestCase):
    def test_setup_w_R_none_input(self):
        w = None
        oldR = None
        n = 100
        wout, Rout = setup_w_R(w = w, oldR = oldR, n = n)
        self.assertTrue(np.array_equal(wout, np.ones(n)*np.ones(1)), msg = 'Expect np.ones([n])')
        self.assertEqual(Rout, None, msg = 'Expect None')
        
    def test_setup_w_R_not_none_input(self):
        w = 100*np.ones(1)
        oldR = np.random.random_sample(size = (3,3))
        n = 100
        wout, Rout = setup_w_R(w = w, oldR = oldR, n = n)
        self.assertTrue(np.array_equal(wout, 100*np.ones(n)*np.ones(1)), msg = 'Expect 100*np.ones([n])')
        self.assertTrue(np.array_equal(Rout, oldR), msg = 'Expect arrays to match')

# --------------------------------------------
class SetupCholUpdate(unittest.TestCase):
    def test_setup_cholupdate(self):
        x = np.zeros([100,2])
        x[:,0] = np.linspace(1,100,100)
        x[:,1] = np.linspace(1,100,100)
        xi = x[0,:]
        wsum = np.ones([1])
        oldmean = np.array([10.2, 2.4])
        oldwsum = 100*np.ones([1])
        R = np.array([[0.3, 0.1],[0, 0.4]])
        Rin, xin = setup_cholupdate(R = R, oldwsum = oldwsum, wsum = wsum, oldmean = oldmean, xi = xi)
        
        tstRin = np.array([[0.298496231131986, 0.0994987437106620],[0, 0.397994974842648]])
        tstxin = np.array([-0.915434214993190,-0.139305206629398])
        self.assertTrue(np.isclose(Rin, tstRin).all(), msg = str('Mean algorithms should agree: {} neq {}'.format(Rin, tstRin)))
        self.assertTrue(np.isclose(xin, tstxin).all(), msg = str('Mean algorithms should agree: {} neq {}'.format(xin, tstxin)))

# --------------------------------------------
class UpdateCovarianceMeanSum(unittest.TestCase):

    def test_update_covariance_mean_sum_none(self):
        x = np.array([[]])
        w = np.ones([1])
        xcov, xmean, wsum = update_covariance_mean_sum(x = x, w = w, oldcov = None, oldmean = None, oldwsum = None, oldR = None)
        self.assertEqual(xcov, None, msg = 'Expect None')
        self.assertEqual(xmean, None, msg = 'Expect None')
        self.assertEqual(wsum, None, msg = 'Expect None')
        
    def test_update_covariance_mean_sum_initialize(self):
        x = np.zeros([100,2])
        x[:,0] = np.linspace(1,100,100)
        x[:,1] = np.linspace(1,100,100)
        w = np.ones([1])
        xcov, xmean, wsum = update_covariance_mean_sum(x = x, w = w, oldcov = None, oldmean = None, oldwsum = None, oldR = None)
        tsmtx = np.array([[8.4167e+02, 8.4167e+02], [8.4167e+02,   8.4167e+02]])
        tsmean = np.array([5.0500e+01, 5.0500e+01])
        self.assertTrue(isinstance(xcov, np.ndarray), msg = 'Expect numpy array')
        self.assertEqual(xcov.shape, (2,2), msg = 'Expect shape = (2,2)')
        self.assertTrue(np.isclose(tsmtx, xcov).all(), msg = str('Mean algorithms should agree: {} neq {}'.format(tsmtx, xcov)))
        self.assertTrue(np.isclose(tsmean, xmean).all(), msg = str('Mean algorithms should agree: {} neq {}'.format(tsmean, xmean)))
        
        self.assertTrue(np.isclose(xmean, np.mean(x, axis = 0)).all(), msg = str('Mean algorithms should agree: {} neq {}'.format(xmean, np.mean(x, axis = 0))))
        self.assertEqual(wsum, 100, msg = 'Expect wsum = 100')
        
    def test_update_covariance_mean_sum_R_is_none(self):
        x = np.zeros([100,2])
        x[:,0] = np.linspace(1,100,100)
        x[:,1] = np.linspace(1,100,100)
        w = np.ones([1])
        oldcov = np.array([[0.5, 0.1],[0.1, 0.3]])
        oldmean = np.array([10.2, 2.4])
        oldwsum = 100*np.ones(1)
        xcov, xmean, wsum = update_covariance_mean_sum(x = x, w = w, oldcov = oldcov, oldmean = oldmean, oldwsum = oldwsum, oldR = None)
        tsmtx = np.array([[827.030150753768,	905.811055276382],[905.811055276382,	1000.17688442211]])
        tsmean = np.array([30.3500000000000,	26.4500000000000])
        self.assertTrue(isinstance(xcov, np.ndarray), msg = 'Expect numpy array')
        self.assertEqual(xcov.shape, (2,2), msg = 'Expect shape = (2,2)')
        self.assertTrue(np.isclose(tsmtx, xcov).all(), msg = str('Mean algorithms should agree: {} neq {}'.format(tsmtx, xcov)))
        self.assertTrue(np.isclose(tsmean, xmean).all(), msg = str('Mean algorithms should agree: {} neq {}'.format(tsmean, xmean)))
        self.assertEqual(wsum, 200, msg = 'Expect wsum = 200')
        
    def test_update_covariance_mean_sum_R_is_not_none(self):
        x = np.zeros([100,2])
        x[:,0] = np.linspace(1,100,100)
        x[:,1] = np.linspace(1,100,100)
        w = np.ones([1])
        oldcov = np.array([[0.5, 0.1],[0.1, 0.3]])
        oldmean = np.array([10.2, 2.4])
        oldwsum = 100*np.ones([1])
        oldR = np.array([[0.3, 0.1],[0, 0.4]])
        xcov, xmean, wsum = update_covariance_mean_sum(x = x, w = w, oldcov = oldcov, oldmean = oldmean, oldwsum = oldwsum, oldR = oldR)
        tsmtx = np.array([[827.030150753768,	905.811055276382],[905.811055276382,	1000.17688442211]])
        tsmean = np.array([30.3500000000000,	26.4500000000000])
        self.assertTrue(isinstance(xcov, np.ndarray), msg = 'Expect numpy array')
        self.assertEqual(xcov.shape, (2,2), msg = 'Expect shape = (2,2)')
        self.assertTrue(np.isclose(tsmtx, xcov).all(), msg = str('Mean algorithms should agree: {} neq {}'.format(tsmtx, xcov)))
        self.assertTrue(np.isclose(tsmean, xmean).all(), msg = str('Mean algorithms should agree: {} neq {}'.format(tsmean, xmean)))
        self.assertEqual(wsum, 200, msg = 'Expect wsum = 200')

# --------------------------------------------
class InitializeCovarianceMeanSum(unittest.TestCase):
    def test_initialize_covariance_mean_sum(self):
        x = np.zeros([100,2])
        x[:,0] = np.linspace(1,100,100)
        x[:,1] = np.linspace(1,100,100)
        w = np.ones([100,])
        xcov, xmean, wsum = initialize_covariance_mean_sum(x = x, w = w)
        tsmtx = np.array([[8.4167e+02, 8.4167e+02], [8.4167e+02,   8.4167e+02]])
        tsmean = np.array([5.0500e+01, 5.0500e+01])
        self.assertTrue(isinstance(xcov, np.ndarray), msg = 'Expect numpy array')
        self.assertEqual(xcov.shape, (2,2), msg = 'Expect shape = (2,2)')
        self.assertTrue(np.isclose(tsmtx, xcov).all(), msg = str('Mean algorithms should agree: {} neq {}'.format(tsmtx, xcov)))
        self.assertTrue(np.isclose(tsmean, xmean).all(), msg = str('Mean algorithms should agree: {} neq {}'.format(tsmean, xmean)))
        
        self.assertTrue(np.isclose(xmean, np.mean(x, axis = 0)).all(), msg = str('Mean algorithms should agree: {} neq {}'.format(xmean, np.mean(x, axis = 0))))
        self.assertEqual(wsum, 100, msg = 'Expect wsum = 100')

# --------------------------------------------
class Initialization(unittest.TestCase):
    def test_init_adapt(self):
        AD = Adaptation()
        ADD = AD.__dict__
        check_fields = ['qcov', 'qcov_scale', 'R', 'qcov_original', 'invR', 'iacce', 'covchain', 'meanchain']
        for ii, cf in enumerate(check_fields):
            self.assertTrue(ADD[cf] is None, msg = str('Initialize {} to None'.format(cf)))
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
        model, options, parameters, data = gf.setup_mcmc()
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
        Rout = below_burnin_threshold(rejected = rejected, iiadapt = iiadapt, R = R, burnin_scale = burninscale, verbosity = verbosity)
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
        Rout = below_burnin_threshold(rejected = rejected, iiadapt = iiadapt, R = R, burnin_scale = burninscale, verbosity = verbosity)
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
        
# --------------------------------------------
class CheckForSingularCovMatrix(unittest.TestCase):
    @patch('pymcmcstat.samplers.Adaptation.is_semi_pos_def_chol', return_value = (True, np.array([[0.1, 0.],[0., 0.25]])))
    def test_singular_cov_mat_scale_cholesky(self, mock_is):
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
        Rout = check_for_singular_cov_matrix(upcov = upcov, R = R, npar = npar, qcov_adjust = qcov_adjust, qcov_scale = qcov_scale, rejected = rejected, iiadapt = iiadapt, verbosity = verbosity)
        sys.stdout = sys.__stdout__
        self.assertTrue(np.array_equal(Rout, R*qcov_scale), msg = 'Expect arrays to match')
        
    def test_check_for_singular_cov_mat_successfully_adjusted(self):
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
        Rout = check_for_singular_cov_matrix(upcov = upcov, R = R, npar = npar, qcov_adjust = qcov_adjust, qcov_scale = qcov_scale, rejected = rejected, iiadapt = iiadapt, verbosity = verbosity)
        sys.stdout = sys.__stdout__
        self.assertEqual(capturedOutput.getvalue(), 'adjusted covariance matrix\n', msg = 'Expected string {}'.format('adjusted covariance matrix\n'))
        tmp = upcov + np.eye(npar)*qcov_adjust
        pos_def_adjust, pRa = is_semi_pos_def_chol(tmp)
        self.assertTrue(pos_def_adjust, msg = 'Expect True')
        self.assertTrue(np.array_equal(pRa*qcov_scale, Rout), msg = str('Expect arrays to match: {} neq {}'.format(pRa*qcov_scale, Rout)))
        
    @patch('pymcmcstat.samplers.Adaptation.is_semi_pos_def_chol', return_value = (False, None))
    def test_check_for_singular_cov_mat_not_successfully_adjusted(self, mock_is):
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
        Rout = check_for_singular_cov_matrix(upcov = upcov, R = R, npar = npar, qcov_adjust = qcov_adjust, qcov_scale = qcov_scale, rejected = rejected, iiadapt = iiadapt, verbosity = verbosity)
        sys.stdout = sys.__stdout__
        self.assertEqual(capturedOutput.getvalue(), 'covariance matrix singular, no adaptation 96.0\n', msg = 'Expected string {}'.format('covariance matrix singular, no adaptation 96.0\n'))
        self.assertTrue(np.array_equal(R, Rout), msg = str('Expect arrays to match: {} neq {}'.format(R, Rout)))
        
# --------------------------------------------
class UpdateCovFromCovchain(unittest.TestCase):
    def test_update_cov_from_chain_all_adapt(self):
        covchain = np.random.random_sample(size = (3,3))
        qcov = np.random.random_sample(size = (3,3))
        no_adapt_index = np.array([False, False, False], dtype = bool)
        upcov = update_cov_from_covchain(covchain = covchain, qcov = qcov, no_adapt_index = no_adapt_index)
        
        self.assertTrue(np.array_equal(upcov, covchain), msg = str('Expect arrays to match: {} neq {}'.format(covchain, upcov)))
        
    def test_update_cov_from_chain_middle_no_adapt(self):
        covchain = np.random.random_sample(size = (3,3))
        qcov = np.random.random_sample(size = (3,3))
        no_adapt_index = np.array([False, True, False], dtype = bool)
        upcov = update_cov_from_covchain(covchain = covchain, qcov = qcov, no_adapt_index = no_adapt_index)
        covchain[no_adapt_index, :] = qcov[no_adapt_index,:]
        covchain[:,no_adapt_index] = qcov[:,no_adapt_index]
        self.assertTrue(np.array_equal(upcov, covchain), msg = str('Expect arrays to match: {} neq {}'.format(covchain, upcov)))
        
    def test_update_cov_from_chain_ends_no_adapt(self):
        covchain = np.random.random_sample(size = (3,3))
        qcov = np.random.random_sample(size = (3,3))
        no_adapt_index = np.array([True, False, True], dtype = bool)
        upcov = update_cov_from_covchain(covchain = covchain, qcov = qcov, no_adapt_index = no_adapt_index)
        covchain[no_adapt_index, :] = qcov[no_adapt_index,:]
        covchain[:,no_adapt_index] = qcov[:,no_adapt_index]
        self.assertTrue(np.array_equal(upcov, covchain), msg = str('Expect arrays to match: {} neq {}'.format(covchain, upcov)))