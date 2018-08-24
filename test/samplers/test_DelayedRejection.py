#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 08:48:00 2018

@author: prmiles
"""

from pymcmcstat.samplers.DelayedRejection import update_set_based_on_acceptance, extract_state_elements
from pymcmcstat.samplers.DelayedRejection import log_posterior_ratio, nth_stage_log_proposal_ratio
from pymcmcstat.samplers.DelayedRejection import DelayedRejection
from pymcmcstat.structures.ParameterSet import ParameterSet
from pymcmcstat.procedures.SumOfSquares import SumOfSquares
from pymcmcstat.procedures.PriorFunction import PriorFunction
import test.general_functions as gf

import unittest
from mock import patch
import numpy as np

# --------------------------
class InitializeDRMetrics(unittest.TestCase):
    def test_dr_metrics(self):
        DR = DelayedRejection()
        __, options, __, data, covariance, rejected, chain, s2chain, sschain = gf.setup_mcmc_case_dr()
        DR._initialize_dr_metrics(options = options)
        self.assertTrue(np.array_equal(DR.iacce, np.zeros(options.ntry, dtype = int)), msg = 'Arrays should match')
        self.assertEqual(DR.dr_step_counter, 0, msg = 'Counter initialized to zero')

# --------------------------
class InitializeNextMetropolisStep(unittest.TestCase):
    @patch('numpy.random.randn', return_value = np.array([0.2, 0.5]))
    def test_next_set(self, mock_1):
        DR = DelayedRejection()
        npar = 2
        old_theta = np.array([0.1, 0.2])
        RDR = np.array([[0.4, 0.2],[0, 0.3]])
        sigma2 = 0.24
        next_set = DR.initialize_next_metropolis_step(npar = npar, old_theta = old_theta, sigma2 = sigma2, RDR = RDR)
        self.assertEqual(next_set.sigma2, sigma2, msg = 'sigma2 should be 0.24')
        self.assertTrue(np.array_equal(next_set.theta, (old_theta + np.dot(np.array([0.2,0.5]),RDR)).reshape(npar)), msg = 'Arrays should match')
        
# -------------------------------------------
class UpdateSetBasedOnAcceptance(unittest.TestCase):
    def test_set_based_on_accept_false(self):
        next_set = ParameterSet(theta = np.random.random_sample(size = (2,1)), ss = 0.4)
        old_set = ParameterSet(theta = np.random.random_sample(size = (2,1)), ss = 0.6)
        out_set = update_set_based_on_acceptance(accept = False, old_set = old_set, next_set = next_set)
        self.assertEqual(out_set, old_set)
        
    def test_set_based_on_accept_true(self):
        next_set = ParameterSet(theta = np.random.random_sample(size = (2,1)), ss = 0.4)
        old_set = ParameterSet(theta = np.random.random_sample(size = (2,1)), ss = 0.6)
        out_set = update_set_based_on_acceptance(accept = True, old_set = old_set, next_set = next_set)
        self.assertEqual(out_set, next_set)
        
# -------------------------------------------
class ExtractStateElements(unittest.TestCase):
    def setup_trypath(self, iq = 0):
        trypath = []
        trypath.append(ParameterSet(theta = 0.1))
        trypath.append(ParameterSet(theta = 0.2))
        trypath.append(ParameterSet(theta = 0.3))
        trypath.append(ParameterSet(theta = 0.4))
        stage = len(trypath) - 2
        y1, y2, y3, y4 = extract_state_elements(iq = iq, stage = stage, trypath = trypath)
        return trypath, y1, y2, y3, y4
    def test_extract_state_elements_iq_eq_0(self):
        trypath, y1, y2, y3, y4 = self.setup_trypath(iq = 0)
        self.assertEqual(y1, trypath[0].theta, msg = 'Expect [0]')
        self.assertEqual(y2, trypath[1].theta, msg = 'Expect [1]')
        self.assertEqual(y3, trypath[3].theta, msg = 'Expect [3]')
        self.assertEqual(y4, trypath[2].theta, msg = 'Expect [2]')
        
    def test_extract_state_elements_iq_eq_1(self):
        trypath, y1, y2, y3, y4 = self.setup_trypath(iq = 1)
        self.assertEqual(y1, trypath[0].theta, msg = 'Expect [0]')
        self.assertEqual(y2, trypath[2].theta, msg = 'Expect [2]')
        self.assertEqual(y3, trypath[3].theta, msg = 'Expect [3]')
        self.assertEqual(y4, trypath[1].theta, msg = 'Expect [1]')
        
    def test_extract_state_elements_iq_eq_2(self):
        trypath, y1, y2, y3, y4 = self.setup_trypath(iq = 2)
        self.assertEqual(y1, trypath[0].theta, msg = 'Expect [0]')
        self.assertEqual(y2, trypath[3].theta, msg = 'Expect [3]')
        self.assertEqual(y3, trypath[3].theta, msg = 'Expect [3]')
        self.assertEqual(y4, trypath[0].theta, msg = 'Expect [0]')
        
# -------------------------------------------
class LogPosteriorRatio(unittest.TestCase):
    def test_logposteriorratio(self):
        trypath = []
        trypath.append(ParameterSet(theta = 0.1, ss = np.array([10.2]), sigma2 = np.array([0.5]), prior = np.array([0.5])))
        trypath.append(ParameterSet(theta = 0.2, ss = np.array([8.2]), sigma2 = np.array([0.5]), prior = np.array([0.5])))
        zq = log_posterior_ratio(x1 = trypath[0], x2 = trypath[-1])
        self.assertTrue(isinstance(zq, float), msg = 'Expect float return')
        self.assertTrue(zq > 0., msg = 'Expect positve return')
        
    def test_logposteriorratio_with_2d(self):
        trypath = []
        trypath.append(ParameterSet(theta = 0.1, ss = np.array([10.2, 5.1]), sigma2 = np.array([0.5, 0.6]), prior = np.array([0.5, 0.75])))
        trypath.append(ParameterSet(theta = 0.2, ss = np.array([8.2, 7.5]), sigma2 = np.array([0.5, 1.2]), prior = np.array([0.5, 0.25])))
        zq = log_posterior_ratio(x1 = trypath[0], x2 = trypath[-1])
        self.assertTrue(isinstance(zq, float), msg = 'Expect float return')
        
# -------------------------------------------
class NthStateLogProposalRatio(unittest.TestCase):
    def test_nth_stage_logpropratio(self):
        iq = 0
        trypath = []
        trypath.append(ParameterSet(theta = 0.1, ss = np.array([10.2]), sigma2 = np.array([0.5]), prior = np.array([0.5])))
        trypath.append(ParameterSet(theta = 0.2, ss = np.array([8.2]), sigma2 = np.array([0.5]), prior = np.array([0.5])))
        zq = nth_stage_log_proposal_ratio(iq = iq, trypath = trypath, invR = None)
        self.assertTrue(np.array_equal(zq, np.zeros([1])), msg = 'Expect arrays to match')
        
    def test_nth_stage_logpropratio_invR(self):
        iq = 0
        invR = [np.array([[0.4, 0.1],[0., 0.2]])]
        trypath = []
        trypath.append(ParameterSet(theta = 0.1, ss = np.array([10.2]), sigma2 = np.array([0.5]), prior = np.array([0.5])))
        trypath.append(ParameterSet(theta = 0.2, ss = np.array([8.2]), sigma2 = np.array([0.5]), prior = np.array([0.5])))
        trypath.append(ParameterSet(theta = 0.2, ss = np.array([8.2]), sigma2 = np.array([0.5]), prior = np.array([0.5])))
        zq = nth_stage_log_proposal_ratio(iq = iq, trypath = trypath, invR = invR)
        self.assertTrue(isinstance(zq, float), msg = 'Expect float return')
        
# -------------------------------------------
class AlphaFunction(unittest.TestCase):
    def test_alphafun(self):
        invR = []
        invR.append(np.array([[0.4, 0.1],[0., 0.2]]))
        invR.append(np.array([[0.4, 0.1],[0., 0.2]])/4)
        invR.append(np.array([[0.4, 0.1],[0., 0.2]])/5)
        trypath = []
        trypath.append(ParameterSet(theta = 0.1, ss = np.array([10.2]), sigma2 = np.array([0.5]), prior = np.array([0.5])))
        trypath.append(ParameterSet(theta = 0.2, ss = np.array([8.2]), sigma2 = np.array([0.5]), prior = np.array([0.5])))
        trypath.append(ParameterSet(theta = 0.2, ss = np.array([8.2]), sigma2 = np.array([0.5]), prior = np.array([0.5])))
        trypath.append(ParameterSet(theta = 0.2, ss = np.array([8.2]), sigma2 = np.array([0.5]), prior = np.array([0.5])))
        __, options, __, __ = gf.setup_mcmc()
        DR = DelayedRejection()
        DR._initialize_dr_metrics(options = options)
        alpha = DR.alphafun(trypath = trypath, invR = invR)
        self.assertIsInstance(alpha, np.ndarray, msg = 'Expect numpy array return')
        self.assertEqual(alpha.size, 1, msg = 'Expect single element array')
        
# -------------------------------------------
class RunDelayedRejection(unittest.TestCase):
    def setup_dr(self):
        model, options, parameters, data, covariance, __, __, __, __ = gf.setup_mcmc_case_dr()
        RDR = covariance._RDR
        invR = covariance._invR
        old_set = ParameterSet(theta = np.random.rand(2), ss = np.array([10.2]), sigma2 = np.array([0.5]), prior = np.array([0.5]))
        new_set = ParameterSet(theta = np.random.rand(2), ss = np.array([8.2]), sigma2 = np.array([0.5]), prior = np.array([0.5]))
        priorobj = PriorFunction(priorfun = model.prior_function, mu = parameters._thetamu[parameters._parind[:]], sigma = parameters._thetasigma[parameters._parind[:]])
        sosobj = SumOfSquares(model, data, parameters)
        DR = DelayedRejection()
        DR._initialize_dr_metrics(options = options)
        accept, out_set, outbound = DR.run_delayed_rejection(old_set = old_set, new_set = new_set, RDR = RDR, ntry = 2, parameters = parameters, invR = invR, sosobj = sosobj, priorobj = priorobj)
        return accept, out_set, outbound
    @patch('pymcmcstat.samplers.DelayedRejection.acceptance_test', return_value = True)
    def test_run_dr(self, mock_accept):
        accept, __, __ = self.setup_dr()
        self.assertTrue(accept, msg = 'Expect return True')
        
    @patch('pymcmcstat.samplers.DelayedRejection.is_sample_outside_bounds', return_value = True)
    def test_run_dr_outside(self, mock_outside):
        accept, out_set, outbound = self.setup_dr()
        self.assertFalse(accept, msg = 'Expect return False')
        self.assertTrue(outbound, msg = 'Expect return True')
        self.assertIsInstance(out_set, ParameterSet, msg = 'Expect structure')