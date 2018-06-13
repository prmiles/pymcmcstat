#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 08:26:50 2018

@author: prmiles
"""

from pymcmcstat.samplers.Adaptation import Adaptation
import unittest

class Initialization(unittest.TestCase):
    def test_init_adapt(self):
        AD = Adaptation()
        ADD = AD.__dict__
        check_fields = ['qcov', 'qcov_scale', 'R', 'qcov_original', 'invR', 'iacce', 'covchain', 'meanchain']
        for ii, cf in enumerate(check_fields):
            self.assertTrue(ADD[cf] is None, msg = str('Initialize {} to None'.format(cf)))