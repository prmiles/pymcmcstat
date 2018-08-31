#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 10:22:54 2018

@author: prmiles
"""

from pymcmcstat.chain import ChainStatistics
import unittest
import numpy as np
import io
import sys

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
# --------------------------
class BatchMeanSTD(unittest.TestCase):
    
    def test_len_s(self):
        s = CS.batch_mean_standard_deviation(chain, b = None)
        self.assertEqual(len(s), 2)
        
    def test_too_few_batches(self):
        with self.assertRaises(SystemExit, msg = 'too few batches'):
            CS.batch_mean_standard_deviation(chain, b = chain.shape[0])
# --------------------------
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
        
# --------------------------
def setup_chains():
    chains = []
    for ii in range(4):
        chains.append(np.concatenate((ii*np.linspace(0, 1, 1000).reshape(1000,1), ii*np.linspace(2.5, 3.3, 1000).reshape(1000,1)), axis = 1))
    return chains

class GelmanRubin(unittest.TestCase):
    def standard_check(self, psrf):
        self.assertTrue(isinstance(psrf, dict), msg = 'Expect dictionary output')
        check_these = ['B', 'W', 'V', 'R', 'neff']
        for _, ps in enumerate(psrf):
            for ct in check_these:
                self.assertTrue(ct in psrf[ps], msg = str('{} not in {}'.format(ct, ps)))
    
    def test_gelman_rubin(self):
        chains = setup_chains()
        capturedOutput = io.StringIO()                  # Create StringIO object
        sys.stdout = capturedOutput                     #  and redirect stdout.
        psrf = CS.gelman_rubin(chains = chains, display = True)
        sys.stdout = sys.__stdout__                     # Reset redirect.
        self.assertTrue(isinstance(capturedOutput.getvalue(), str), msg = 'Caputured string')
        self.standard_check(psrf)
    
    def test_gelman_rubin_no_display(self):
        chains = setup_chains()
        capturedOutput = io.StringIO()                  # Create StringIO object
        sys.stdout = capturedOutput                     #  and redirect stdout.
        psrf = CS.gelman_rubin(chains = chains, display = False)
        sys.stdout = sys.__stdout__                     # Reset redirect.
        self.assertTrue(isinstance(capturedOutput.getvalue(), str), msg = 'Caputured string')
        self.assertEqual(capturedOutput.getvalue(), '', msg = 'Caputured string')
        self.standard_check(psrf)
        
    def test_gelman_rubin_with_pres(self):
        chains = setup_chains()
        pres = []
        for _, chain in enumerate(chains):
            pres.append(dict(chain = chain, nsimu = chain.shape[0]))
            
        psrf = CS.gelman_rubin(chains = pres)
        self.standard_check(psrf)
                
    def test_gelman_rubin_with_names(self):
        chains = setup_chains()
        psrf = CS.gelman_rubin(chains = chains, names = ['a', 'b'])
        self.standard_check(psrf)
        
    def test_gelman_rubin_raise_error(self):
        chains = setup_chains()
        for _ in range(len(chains)-1):
            chains.pop(-1)
        with self.assertRaises(ValueError, msg = 'Must have multiple chains'):
            CS.gelman_rubin(chains = chains)
            
# --------------------------
class PSRF(unittest.TestCase):
    def test_calc_psrf(self):
        x = np.concatenate((np.linspace(0, 1, 1000).reshape(1000,1), np.linspace(2.5, 3.3, 1000).reshape(1000,1)), axis = 1)
        psrf = CS.calculate_psrf(x, nsimu = 1000, nchains = 2)
        self.assertTrue(isinstance(psrf, dict), msg = 'Expect dictionary output')
        self.assertAlmostEqual(psrf['R'], 8.001818935964492, places = 6, msg = str('R: {} neq {}'.format(psrf['R'], 8.001818935964492)))
        self.assertAlmostEqual(psrf['B'], 2879.9999999999964, places = 6, msg = str('R: {} neq {}'.format(psrf['B'], 2879.9999999999964)))
        self.assertAlmostEqual(psrf['W'], 0.06853867547894903, places = 6, msg = str('R: {} neq {}'.format(psrf['W'], 0.06853867547894903)))
        self.assertAlmostEqual(psrf['V'], 4.388470136803464, places = 6, msg = str('R: {} neq {}'.format(psrf['V'], 4.388470136803464)))
        self.assertAlmostEqual(psrf['neff'], 3.047548706113521, places = 6, msg = str('R: {} neq {}'.format(psrf['neff'], 3.047548706113521)))