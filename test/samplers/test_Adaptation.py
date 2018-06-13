#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 08:26:50 2018

@author: prmiles
"""

from pymcmcstat.samplers.Adaptation import Adaptation
import unittest
import numpy as np
import math

class Initialization(unittest.TestCase):
    def test_init_adapt(self):
        AD = Adaptation()
        ADD = AD.__dict__
        check_fields = ['qcov', 'qcov_scale', 'R', 'qcov_original', 'invR', 'iacce', 'covchain', 'meanchain']
        for ii, cf in enumerate(check_fields):
            self.assertTrue(ADD[cf] is None, msg = str('Initialize {} to None'.format(cf)))
            
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
        
class IsSemiPositiveDefinite(unittest.TestCase):
    def test_semipositivedef(self):
        AD = Adaptation()
        mtx = np.diag([2, 2])
        flag, c = AD.is_semi_pos_def_chol(x = mtx)
        self.assertTrue(flag, msg = 'Expect true')
        self.assertTrue(isinstance(c, np.ndarray), msg = 'Expect numpy array')
        self.assertEqual(c.shape, (2,2), msg = 'Expect 2x2 array')
        
    def test_notsemipositivedef(self):
        AD = Adaptation()
        mtx = np.array([[2, -1],[0, 0]])
        flag, c = AD.is_semi_pos_def_chol(x = mtx)
        self.assertFalse(flag, msg = 'Expect false')