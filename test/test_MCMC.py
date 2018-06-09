#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 08:33:47 2018

@author: prmiles
"""

from pymcmcstat.MCMC import MCMC
import unittest

class MCMCInitialization(unittest.TestCase):
    
    def test_initialization(self):
        MC = MCMC()
        
        check_these = ['data', 'model_settings', 'simulation_options', 'parameters',
                       '_error_variance', '_covariance', '_sampling_methods', '_mcmc_status']
        for ct in check_these:
            self.assertTrue(hasattr(MC, ct), msg = str('Object missing attribute: {}'.format(ct)))
        
        self.assertFalse(MC._mcmc_status, msg = 'Status is False')