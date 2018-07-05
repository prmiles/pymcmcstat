#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 09:47:02 2018

@author: prmiles
"""
from pymcmcstat.samplers.SamplingMethods import SamplingMethods
from pymcmcstat.samplers.Metropolis import Metropolis
from pymcmcstat.samplers.DelayedRejection import DelayedRejection
from pymcmcstat.samplers.Adaptation import Adaptation

import unittest

# -------------------
class CreateSamplingMethods(unittest.TestCase):
    def test_create_sampling_methods(self):
        SM = SamplingMethods()
        self.assertTrue(isinstance(SM.metropolis, Metropolis), msg = 'Metropolis Class')
        self.assertTrue(isinstance(SM.delayed_rejection, DelayedRejection), msg = 'Delayed Rejection Class')
        self.assertTrue(isinstance(SM.adaptation, Adaptation), msg = 'Adaptation Class')