#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 11:52:58 2018

@author: prmiles
"""

from pymcmcstat.settings.SimulationOptions import SimulationOptions
from pymcmcstat.settings.ModelSettings import ModelSettings
import unittest

def setup_options(**kwargs):
    SO = SimulationOptions()
    SO.define_simulation_options(**kwargs)
    return SO

MS = ModelSettings()
MS.define_model_settings()

# --------------------------
class DisplaySimulationOptions(unittest.TestCase):

    def test_print_these_none(self):
        SO = setup_options()
        print_these = SO.display_simulation_options(print_these = None)
        self.assertEqual(print_these, ['nsimu', 'adaptint', 'ntry', 'method', 'printint', 'lastadapt', 'drscale', 'qcov'], msg = 'Default print keys')
        
    def test_print_these_not_none(self):
        SO = setup_options()
        print_these = SO.display_simulation_options(print_these = ['nsimu'])
        self.assertEqual(print_these, ['nsimu'], msg = 'Specified print keys')
        
# --------------------------
class CheckDependentOptions(unittest.TestCase):
    
    def test_dodram(self):
        SO = setup_options(ntry = 1)
        SO._check_dependent_simulation_options(model = MS)
        self.assertEqual(SO.dodram, 0, msg = 'DRAM turned off because ntry <= 1')
        SO = setup_options(ntry = 2)
        SO._check_dependent_simulation_options(model = MS)
        self.assertEqual(SO.dodram, 1, msg = 'DRAM turned on because ntry > 1')
        
    def test_lastadapt(self):
        SO = setup_options(lastadapt = 0)
        SO._check_dependent_simulation_options(model = MS)
        self.assertEqual(SO.lastadapt, SO.nsimu, msg = 'lastadapt set to nsimu')
        SO.define_simulation_options(lastadapt = 10)
        SO._check_dependent_simulation_options(model = MS)
        self.assertEqual(SO.lastadapt, 10, msg = 'lastadapt unchanged')
        
    def test_printint(self):
        SO = setup_options(printint = None)
        SO._check_dependent_simulation_options(model = MS)
        self.assertEqual(SO.printint, max(100,min(1000,SO.adaptint)), msg = 'printint was updated')
        
    def test_updatesigma(self):
        SO = setup_options(updatesigma = False)
        MS.N0 = [1.]
        SO._check_dependent_simulation_options(model = MS)
        self.assertTrue(SO.updatesigma, msg = 'updatesigma turned on because N0 not empty')
        
# --------------------------
class DefineOptions(unittest.TestCase):
    
    def test_define_options_method_assignment(self):
        methods = ['mh', 'dr', 'am', 'dram']
        for method in methods:
            SO = setup_options(method = method)
            self.assertEqual(SO.method, method, msg = str('Method set to {}'.format(method)))
            
    def test_ntry_assigment(self):
        SO = setup_options(ntry = 3)
        self.assertEqual(SO.ntry, 3, msg = 'ntry set to 3')
        
    def test_adascale_assigment(self):
        SO = setup_options(adascale = 3)
        self.assertEqual(SO.adascale, 3, msg = 'adascale set to 3')
        
    def test_label_assigment(self):
        SO = setup_options(label = '3')
        self.assertEqual(SO.label, '3', msg = 'label set to 3')
        
    def test_savedir_assigment(self):
        SO = setup_options(savedir = '3')
        self.assertEqual(SO.savedir, '3', msg = 'savedir set to 3')