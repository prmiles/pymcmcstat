#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 08:02:45 2018

@author: prmiles
"""

from pymcmcstat.structures.ResultsStructure import ResultsStructure
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
        
    