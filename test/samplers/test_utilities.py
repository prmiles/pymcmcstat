#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 12:08:17 2018

@author: prmiles
"""
from pymcmcstat.samplers.utilities import sample_candidate_from_gaussian_proposal
from pymcmcstat.samplers.utilities import is_sample_outside_bounds
from pymcmcstat.samplers.utilities import posterior_ratio_acceptance_test
from pymcmcstat.samplers.utilities import acceptance_test
from pymcmcstat.samplers.utilities import set_outside_bounds
from pymcmcstat.samplers.utilities import log_posterior_ratio_acceptance_test as lprat
from pymcmcstat.samplers.utilities import calculate_log_posterior_ratio
from pymcmcstat.structures.ParameterSet import ParameterSet
import unittest
from mock import patch
import numpy as np


# --------------------------
class SampleCandidate(unittest.TestCase):
    @patch('numpy.random.randn')
    def test_sample_candidate(self, mock_simple_func):
        mock_simple_func.return_value = np.array([0.1, 0.2])
        oldpar = np.array([0.1, 0.4])
        R = np.array([[0.4, 0.2],[0, 0.3]])
        newpar, npar_sample_from_normal = sample_candidate_from_gaussian_proposal(npar = 2, oldpar = oldpar, R = R)
        self.assertEqual(newpar.size, 2, msg='Size of parameter array is 2')
        self.assertEqual(npar_sample_from_normal.size, 2, msg='Size of sample is 2')
        self.assertTrue(np.array_equal(newpar, (oldpar + np.dot(np.array([0.1, 0.2]), R)).reshape(2)), msg='Arrays should match')

# --------------------------
class OutsideBounds(unittest.TestCase):
    def test_outsidebounds_p1_below(self):
        self.assertTrue(is_sample_outside_bounds(theta = np.array([-1.0, 1.5]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])), msg='p1 below')
        self.assertTrue(is_sample_outside_bounds(theta = np.array([1.0, 0.5]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])), msg='p2 below')
        self.assertTrue(is_sample_outside_bounds(theta = np.array([1.5, 1.5]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])), msg='p1 above')
        self.assertTrue(is_sample_outside_bounds(theta = np.array([1.0, 2.5]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])), msg='p2 above')
        self.assertFalse(is_sample_outside_bounds(theta = np.array([0, 1.5]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])), msg='p1 on low lim')
        self.assertFalse(is_sample_outside_bounds(theta = np.array([0.5, 1]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])), msg='p2 on low lim')
        self.assertFalse(is_sample_outside_bounds(theta = np.array([1, 1.5]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])), msg='p1 on upp lim')
        self.assertFalse(is_sample_outside_bounds(theta = np.array([0.5, 2]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])), msg='p2 on upp lim')
        self.assertFalse(is_sample_outside_bounds(theta = np.array([0.5, 1.5]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])), msg='all outside')


# --------------------------
class Acceptance(unittest.TestCase):
    
    def test_accept_alpha_gt_1(self):
        self.assertEqual(acceptance_test(alpha = 1.1), 1,
                         msg='accept alpha >= 1')
        self.assertEqual(acceptance_test(alpha = 1), 1,
                         msg='accept alpha >= 1')
        self.assertEqual(acceptance_test(alpha = -0.1), 0,
                         msg='Reject alpha <= 0')
        self.assertEqual(acceptance_test(alpha = 0), 0,
                         msg='Reject alpha <= 0')
        np.random.seed(0)
        self.assertEqual(acceptance_test(alpha = 0.4), 0,
                         msg='Reject alpha < u (0.4 < 0.5488135)')
        np.random.seed(0)
        self.assertEqual(acceptance_test(alpha = 0.6), 1,
                         msg='Accept alpha > u (0.6 > 0.5488135)')


# --------------------------
class PosteriorRatioAcceptance(unittest.TestCase):
    
    def test_accept_alpha_gt_1(self):
        self.assertEqual(posterior_ratio_acceptance_test(alpha = 1.1), 1,
                         msg='accept alpha >= 1')
        self.assertEqual(posterior_ratio_acceptance_test(alpha = 1), 1,
                         msg='accept alpha >= 1')
        self.assertEqual(posterior_ratio_acceptance_test(alpha = -0.1), 0,
                         msg='Reject alpha <= 0')
        self.assertEqual(posterior_ratio_acceptance_test(alpha = 0), 0,
                         msg='Reject alpha <= 0')
        np.random.seed(0)
        self.assertEqual(posterior_ratio_acceptance_test(alpha = 0.4), 0,
                         msg='Reject alpha < u (0.4 < 0.5488135)')
        np.random.seed(0)
        self.assertEqual(posterior_ratio_acceptance_test(alpha = 0.6), 1,
                         msg='Accept alpha > u (0.6 > 0.5488135)')
        
# --------------------------
class SetOutsideBounds(unittest.TestCase):
    def test_set_outsidebounds(self):
        next_set = ParameterSet()
        next_set, outbound = set_outside_bounds(next_set = next_set)
        self.assertEqual(next_set.alpha, 0, msg='next_set.alpha should be 0')
        self.assertEqual(next_set.prior, 0, msg='next_set.prior should be 0')
        self.assertEqual(next_set.ss, np.inf, msg='next_set.ss should be np.inf')
        self.assertEqual(outbound, 1, msg='outbound should be 1')


# --------------------------
class LogPosteriorRatioAcceptanceTest(unittest.TestCase):
    
    def test_accept_alpha(self):
        self.assertEqual(lprat(alpha=0.1), 1,
                         msg='accept alpha >= 0')
        self.assertEqual(lprat(alpha=0), 1,
                         msg='accept alpha >= 0')
        np.random.seed(0)
        self.assertEqual(lprat(alpha=np.log(0.4)), 0,
                         msg='Reject alpha < log(u) (-0.916 < -0.600)')
        np.random.seed(0)
        self.assertEqual(lprat(alpha=np.log(0.6)), 1,
                         msg='Accept alpha > log(u) (-0.511 > -0.600)')


# --------------------------
class LogPosteriorRatio(unittest.TestCase):

    def test_io(self):
        alpha = calculate_log_posterior_ratio(
                loglikestar=-0.5,
                loglike=-0.4,
                logpriorstar=-0.5,
                logprior=-0.5)
        self.assertAlmostEqual(alpha, -0.1, msg='Expect approx. -0.1')

    def test_gt_zero(self):
        alpha = calculate_log_posterior_ratio(
                loglikestar=-0.5,
                loglike=-1,
                logpriorstar=0,
                logprior=0)
        self.assertTrue(alpha > 0, msg='Expect > 0')

    def test_lt_zero(self):
        alpha = calculate_log_posterior_ratio(
                loglikestar=-1,
                loglike=-0.5,
                logpriorstar=0,
                logprior=0)
        self.assertTrue(alpha < 0, msg='Expect < 0')


