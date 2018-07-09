#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 10:22:54 2018

@author: prmiles
"""

from pymcmcstat.chain import ChainStatistics
import unittest
import numpy as np

CS = ChainStatistics
chain = np.random.random_sample(size = (1000,2))
chain[:,1] = 1e6*chain[:,1]

# --------------------------
# chainstats
# --------------------------
class Chainstats_Eval(unittest.TestCase):
    
    def test_cs_eval_with_return(self):
        stats = CS.chainstats(chain = chain, returnstats = True)
        self.assertTrue(isinstance(stats,dict))
        
    def test_cs_eval_with_no_return(self):
        stats = CS.chainstats(chain = chain, returnstats = False)
        self.assertEqual(stats, None)
        
    def test_cs_eval_with_no_chain(self):
        stats = CS.chainstats(chain = None, returnstats = True)
        self.assertTrue(isinstance(stats, str))
        
class BatchMeanSTD(unittest.TestCase):
    
    def test_len_s(self):
        s = CS.batch_mean_standard_deviation(chain, b = None)
        self.assertEqual(len(s), 2)
        
    def test_too_few_batches(self):
        with self.assertRaises(SystemExit, msg = 'too few batches'):
            CS.batch_mean_standard_deviation(chain, b = chain.shape[0])
            
class PowerSpectralDensity(unittest.TestCase):
    
    def test_nfft_none_size(self):
        x = chain[:,0]
        y = CS.power_spectral_density_using_hanning_window(x = x)
        nfft = min(len(x),256)
        n2 = int(np.floor(nfft/2))
        self.assertEqual(n2, len(y))
        
    def test_nfft_not_none_size(self):
        x = chain[:,0]
        nfft = 100
        y = CS.power_spectral_density_using_hanning_window(x = x, nfft = nfft)
        n2 = int(np.floor(nfft/2))
        self.assertEqual(n2, len(y))
        
    def test_nw_not_none(self):
        x = chain[:,0]
        y = CS.power_spectral_density_using_hanning_window(x = x, nw = len(x))
        nfft = min(len(x),256)
        n2 = int(np.floor(nfft/2))
        self.assertEqual(n2, len(y))