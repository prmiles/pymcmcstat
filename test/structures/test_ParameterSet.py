# -*- coding: utf-8 -*-

"""
Created on Mon Jan. 15, 2018

Description: This file contains a series of utilities designed to test the
features in the 'ParameterSet.py" class of the pymcmcstat module.

@author: prmiles
"""
from pymcmcstat.structures.ParameterSet import ParameterSet
import test.general_functions as gf
import unittest

# --------------------------
# ParameterSet
# --------------------------
class Initialize_Parameter_Set(unittest.TestCase):

    def test_PS_default_match(self):
        PS = ParameterSet()
        PSD = PS.__dict__
        for (k,v) in PSD.items():
            self.assertEqual(v, None, msg = str('Default {} is None'.format(k)))

    def test_PS_set_theta(self):
        x = 1.2
        key = 'theta'
        PS = ParameterSet(theta = x)
        PSD = PS.__dict__
        self.assertEqual(PSD[key], x, msg = str('Expected {} = {}'.format(key, x)))
        PSD = gf.removekey(PSD, key)
        for (k,v) in PSD.items():
            self.assertEqual(v, None, msg = str('Default {} is None'.format(k)))
        
    def test_PS_set_ss(self):
        x = 1.2
        key = 'ss'
        PS = ParameterSet(ss = x)
        PSD = PS.__dict__
        self.assertEqual(PSD[key], x, msg = str('Expected {} = {}'.format(key, x)))
        PSD = gf.removekey(PSD, key)
        for (k,v) in PSD.items():
            self.assertEqual(v, None, msg = str('Default {} is None'.format(k)))
        
    def test_PS_set_prior(self):
        x = 1.2
        key = 'prior'
        PS = ParameterSet(prior = x)
        PSD = PS.__dict__
        self.assertEqual(PSD[key], x, msg = str('Expected {} = {}'.format(key, x)))
        PSD = gf.removekey(PSD, key)
        for (k,v) in PSD.items():
            self.assertEqual(v, None, msg = str('Default {} is None'.format(k)))
            
    def test_PS_set_sigma2(self):
        x = 1.2
        key = 'sigma2'
        PS = ParameterSet(sigma2 = x)
        PSD = PS.__dict__
        self.assertEqual(PSD[key], x, msg = str('Expected {} = {}'.format(key, x)))
        PSD = gf.removekey(PSD, key)
        for (k,v) in PSD.items():
            self.assertEqual(v, None, msg = str('Default {} is None'.format(k)))
            
    def test_PS_set_alpha(self):
        x = 1.2
        key = 'alpha'
        PS = ParameterSet(alpha = x)
        PSD = PS.__dict__
        self.assertEqual(PSD[key], x, msg = str('Expected {} = {}'.format(key, x)))
        PSD = gf.removekey(PSD, key)
        for (k,v) in PSD.items():
            self.assertEqual(v, None, msg = str('Default {} is None'.format(k)))
            
    def test_PS_set_theta_and_prior(self):
        x = 1.2
        keys = ['theta', 'prior']
        PS = ParameterSet(theta = x, prior = x)
        PSD = PS.__dict__
        for key in keys:
            self.assertEqual(PSD[key], x, msg = str('Expected {} = {}'.format(key, x)))
        for key in keys:
            PSD = gf.removekey(PSD, key)
        for (k,v) in PSD.items():
            self.assertEqual(v, None, msg = str('Default {} is None'.format(k)))