#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 07:34:40 2018

@author: prmiles
"""
from pymcmcstat.utilities.NumpyEncoder import NumpyEncoder
import unittest
import numpy as np

NE = NumpyEncoder()

# --------------------------
class NumpyEncoderTesting(unittest.TestCase):

    def test_NE_initialization_attributes(self):
        self.assertTrue(hasattr(NE, 'default'))

    def test_NE_return_ndarray(self):
        self.assertTrue(isinstance(NE.default(obj = np.ones([10,1])), list), msg = 'Numpy array converted to list')

    def test_NE_return_json(self):
        with self.assertRaises(TypeError, msg = 'dict not JSON serializable'):
            NE.default(obj = {'test_object': [10,1]})