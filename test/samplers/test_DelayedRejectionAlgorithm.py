#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 08:48:00 2018

@author: prmiles
"""

from pymcmcstat.samplers.DelayedRejection import DelayedRejection

import unittest
import numpy as np

# --------------------------
class OutsideBounds(unittest.TestCase):
    def test_outsidebounds_p1_below(self):
        DR = DelayedRejection()
        self.assertTrue(DR._is_sample_outside_bounds(theta = np.array([-1.0, 1.5]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])), msg = 'p1 below lower limit')
        
    def test_outsidebounds_p2_below(self):
        DR = DelayedRejection()
        self.assertTrue(DR._is_sample_outside_bounds(theta = np.array([1.0, 0.5]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])), msg = 'p2 below lower limit')
        
    def test_outsidebounds_p1_above(self):
        DR = DelayedRejection()
        self.assertTrue(DR._is_sample_outside_bounds(theta = np.array([1.5, 1.5]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])), msg = 'p1 above upper limit')
        
    def test_outsidebounds_p2_above(self):
        DR = DelayedRejection()
        self.assertTrue(DR._is_sample_outside_bounds(theta = np.array([1.0, 2.5]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])), msg = 'p2 above upper limit')
        
    def test_not_outsidebounds_p1_on_lowlim(self):
        DR = DelayedRejection()
        self.assertFalse(DR._is_sample_outside_bounds(theta = np.array([0, 1.5]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])), msg = 'p1 on lower limit')
        
    def test_not_outsidebounds_p2_on_lowlim(self):
        DR = DelayedRejection()
        self.assertFalse(DR._is_sample_outside_bounds(theta = np.array([0.5, 1]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])), msg = 'p2 on lower limit')
        
    def test_not_outsidebounds_p1_on_upplim(self):
        DR = DelayedRejection()
        self.assertFalse(DR._is_sample_outside_bounds(theta = np.array([1, 1.5]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])), msg = 'p1 on upper limit')
        
    def test_not_outsidebounds_p2_on_upplim(self):
        DR = DelayedRejection()
        self.assertFalse(DR._is_sample_outside_bounds(theta = np.array([0.5, 2]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])), msg = 'p2 on upper limit')
        
    def test_not_outsidebounds_all(self):
        DR = DelayedRejection()
        self.assertFalse(DR._is_sample_outside_bounds(theta = np.array([0.5, 1.5]), lower_limits = np.array([0,1]), upper_limits=np.array([1,2])), msg = 'within limits')
        
        
class Acceptance(unittest.TestCase):
    def test_accept_alpha_gt_1(self):
        DR = DelayedRejection()
        DR.iacce = np.zeros([1])
        self.assertEqual(DR.acceptance_test(alpha = 1.1, old_set = 1, next_set = 1, itry = 0)[0], 1, msg = 'accept alpha >= 1')
        
    def test_accept_alpha_eq_1(self):
        DR = DelayedRejection()
        DR.iacce = np.zeros([1])
        self.assertEqual(DR.acceptance_test(alpha = 1, old_set = 1, next_set = 1, itry = 0)[0], 1, msg = 'accept alpha >= 1')
        
    def test_not_accept_alpha_lt_0(self):
        DR = DelayedRejection()
        self.assertEqual(DR.acceptance_test(alpha = -0.1, old_set = 1, next_set = 1, itry = 0)[0], 0, msg = 'Reject alpha <= 0')
        
    def test_not_accept_alpha_eq_0(self):
        DR = DelayedRejection()
        self.assertEqual(DR.acceptance_test(alpha = 0, old_set = 1, next_set = 1, itry = 0)[0], 0, msg = 'Reject alpha <= 0')
        
    def test_not_accept_alpha_based_on_rand(self):
        DR = DelayedRejection()
        np.random.seed(0)
        self.assertEqual(DR.acceptance_test(alpha = 0.4, old_set = 1, next_set = 1, itry = 0)[0], 0, msg = 'Reject alpha < u (0.4 < 0.5488135)')
        
    def test_accept_alpha_based_on_rand(self):
        DR = DelayedRejection()
        DR.iacce = np.zeros([1])
        np.random.seed(0)
        self.assertEqual(DR.acceptance_test(alpha = 0.6, old_set = 1, next_set = 1, itry = 0)[0], 1, msg = 'Accept alpha > u (0.6 > 0.5488135)')
        
    def test_set_assignment_not_accept(self):
        DR = DelayedRejection()
        old_set = {'testvar1':'abc', 'testvar2':100.1}
        self.assertDictEqual(DR.acceptance_test(alpha = 0, old_set = old_set, next_set = 1, itry = 0)[1], old_set, msg = 'Reject alpha -> new_set = old_set')
        
    def test_set_assignment_accept(self):
        DR = DelayedRejection()
        DR.iacce = np.zeros([1])
        new_set = {'testvar1':'abc', 'testvar2':100.1}
        self.assertDictEqual(DR.acceptance_test(alpha = 1, old_set = 1, next_set = new_set, itry = 0)[1], new_set, msg = 'Accept alpha -> new_set = next_set')