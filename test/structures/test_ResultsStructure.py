#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 08:02:45 2018

@author: prmiles
"""

from pymcmcstat.structures.ResultsStructure import ResultsStructure
from pymcmcstat.settings.SimulationOptions import SimulationOptions
from pymcmcstat.settings.ModelSettings import ModelSettings
import unittest
import numpy as np
import os

# -------------------
class SaveLoadJSONObject(unittest.TestCase):
    
    def generate_temp_file(self):
        tmpfile = 'temp0.json'
        count = 0
        flag = True
        while flag is True:
            if os.path.isfile(str('{}'.format(tmpfile))):
                count += 1
                tmpfile = str('{}{}.json'.format('temp',count))
            else:
                flag = False
        return tmpfile
    
    def test_dump_to_json_file(self):
        RS = ResultsStructure()
        tmpfile = self.generate_temp_file()
        results = {'model': 4, 'data': np.random.random_sample(size = (1000,2))}
        RS.save_json_object(results = results, filename = tmpfile)
        retres = RS.load_json_object(filename = tmpfile)
        self.assertEqual(retres['model'], results['model'], msg = 'Model set to 4')
        self.assertTrue(np.array_equal(retres['data'], results['data']), msg = 'Arrays should be equal')
        os.remove(tmpfile)

# -------------------
class AddUpdateSigma(unittest.TestCase):
    
    def test_updatesigma_true(self):
        updatesigma = True
        sigma2 = np.ones([2,1])
        S20 = np.ones([2,1])
        N0 = np.ones([2,1])
        RS = ResultsStructure()
        RS.add_updatesigma(updatesigma = updatesigma, sigma2 = sigma2, S20 = S20, N0 = N0)
        self.assertTrue(np.isnan(RS.results['sigma2']), msg = 'sigma2 -> np.nan')
        self.assertTrue(np.array_equal(RS.results['S20'], S20), msg = str('S20 -> {}'.format(S20)))
        self.assertTrue(np.array_equal(RS.results['N0'], N0), msg = str('N0 -> {}'.format(N0)))
        
        
    def test_updatesigma_false(self):
        updatesigma = False
        sigma2 = np.ones([2,1])
        S20 = np.ones([2,1])
        N0 = np.ones([2,1])
        RS = ResultsStructure()
        RS.add_updatesigma(updatesigma = updatesigma, sigma2 = sigma2, S20 = S20, N0 = N0)
        self.assertTrue(np.array_equal(RS.results['sigma2'], sigma2), msg = str('sigma2 -> {}'.format(sigma2)))
        self.assertTrue(np.isnan(RS.results['S20']), msg = 'S20 -> np.nan')
        self.assertTrue(np.isnan(RS.results['N0']), msg = 'N0 -> np.nan')
        
# -------------------
class AddArrays(unittest.TestCase):
    
    def test_add_rndnumseq(self):
        rnd = np.random.random_sample(size = (3,2))
        key = 'rndseq'
        RS = ResultsStructure()
        RS.add_random_number_sequence(rndseq = rnd)
        self.assertTrue(np.array_equal(RS.results[key], rnd), 
                        msg = str('Expect equal arrays: {} != {}'.format(RS.results[key], rnd)))
        
    def test_add_chain(self):
        rnd = np.random.random_sample(size = (3,2))
        key = 'chain'
        RS = ResultsStructure()
        RS.add_chain(chain = rnd)
        self.assertTrue(np.array_equal(RS.results[key], rnd), 
                        msg = str('Expect equal arrays: {} != {}'.format(RS.results[key], rnd)))
        
    def test_add_sschain(self):
        rnd = np.random.random_sample(size = (3,2))
        key = 'sschain'
        RS = ResultsStructure()
        RS.add_sschain(sschain = rnd)
        self.assertTrue(np.array_equal(RS.results[key], rnd), 
                        msg = str('Expect equal arrays: {} != {}'.format(RS.results[key], rnd)))
        
    def test_add_s2chain(self):
        rnd = np.random.random_sample(size = (3,2))
        key = 's2chain'
        RS = ResultsStructure()
        RS.add_s2chain(s2chain = rnd)
        self.assertTrue(np.array_equal(RS.results[key], rnd), 
                        msg = str('Expect equal arrays: {} != {}'.format(RS.results[key], rnd)))
    
# -------------------
class AddOptions(unittest.TestCase):
    
    def test_key_removal(self):
        options = SimulationOptions()
        options.define_simulation_options(doram = 3, nsimu = 500)
        RS = ResultsStructure()
        RS.add_options(options = options)
        self.assertEqual(RS.results['simulation_options']['nsimu'], options.nsimu, msg = str('nsimu = {}'.format(options.nsimu)))
        self.assertFalse('doram' in RS.results['simulation_options'].items(), msg = 'doram option should not be saved')
        
# -------------------
class AddModel(unittest.TestCase):
    
    def test_key_removal(self):
        model = ModelSettings()
        model.define_model_settings(N = 100, sos_function = 'hello world')
        RS = ResultsStructure()
        RS.add_model(model = model)
        self.assertEqual(RS.results['model_settings']['N'], model.N, msg = str('N = {}'.format(model.N)))
        self.assertFalse('sos_function' in RS.results['model_settings'].items(), msg = 'sos_function setting should not be saved')
        
# -------------------
class AddTimeStats(unittest.TestCase):
    
    def test_add_time_stats(self):
        key = 'time [mh, dr, am]'
        mtime = 0.01
        drtime = 0.10
        adtime = 0.05
        RS = ResultsStructure()
        RS.add_time_stats(mtime = mtime, drtime = drtime, adtime = adtime)
        self.assertEqual(RS.results[key], [mtime, drtime, adtime], msg = 'Lists should be equal.')
        
# -------------------
class AddPrior(unittest.TestCase):
    
    def test_add_prior(self):
        RS = ResultsStructure()
        mu = 0.0
        sigma = 1.0
        priorfun = 'priorfun'
        priortype = 1
        priorpars = 3
        RS.add_prior(mu = mu, sig = sigma, priorfun = priorfun, priorpars = priorpars, priortype = priortype)
        self.assertEqual(RS.results['prior'], [mu, sigma], msg = str('Expected [{},{}]'.format(mu, sigma)))
        self.assertEqual(RS.results['priorfun'], priorfun, msg = str('Expected {}'.format(priorfun)))
        self.assertEqual(RS.results['priortype'], priortype, msg = str('Expected {}'.format(priortype)))
        self.assertEqual(RS.results['priorpars'], priorpars, msg = str('Expected {}'.format(priorpars)))