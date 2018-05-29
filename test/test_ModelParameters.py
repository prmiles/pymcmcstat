# -*- coding: utf-8 -*-

"""
Created on Mon Jan. 15, 2018

Description: This file contains a series of utilities designed to test the
features in the 'PredictionIntervals.py" package of the pymcmcstat module.  The 
functions tested include:
    - empirical_quantiles(x, p = np.array([0.25, 0.5, 0.75])):

@author: prmiles
"""
from pymcmcstat.settings.ModelParameters import ModelParameters
from pymcmcstat.settings.SimulationOptions import SimulationOptions
import unittest
import numpy as np

# --------------------------
class Add_Model_Parameter_Test(unittest.TestCase):

    def test_does_parameter_assignment_match(self):
        MP = ModelParameters()
        theta0 = 0
        MP.add_model_parameter('aa', theta0)
        self.assertEqual(MP.parameters[0]['theta0'], theta0)
        self.assertEqual(MP.parameters[0]['name'],'aa')
        self.assertEqual(MP.parameters[0]['minimum'], -np.inf)
        self.assertEqual(MP.parameters[0]['maximum'], np.inf)
        self.assertEqual(MP.parameters[0]['prior_mu'],np.zeros([1]))
        self.assertEqual(MP.parameters[0]['prior_sigma'],np.inf)
        self.assertEqual(MP.parameters[0]['sample'],1)
        self.assertEqual(MP.parameters[0]['local'],0)
        
    def test_results_to_params(self):
        MP = ModelParameters()
        MP.add_model_parameter('aa', 0)
        MP._openparameterstructure(nbatch=1)
#        print('parind = {}'.format(MP._parind))
#        print('local = {}'.format(MP._local))
         # define minimal results dictionary
        results = {'parind': MP._parind, 'names': MP._names, 'local': MP._local, 'theta': [1.2]}
        # initialize default options
        SO = SimulationOptions()
        SO.define_simulation_options()
        MP.display_parameter_settings(verbosity = SO.verbosity, noadaptind = SO.noadaptind)
        MP._results_to_params(results, 1)
        MP._openparameterstructure(nbatch=1)
        MP.display_parameter_settings(verbosity = SO.verbosity, noadaptind = SO.noadaptind)
        self.assertEqual(MP.parameters[0]['theta0'], results['theta'][0])
        
# --------------------------
MP = ModelParameters()
class Message(unittest.TestCase):
    
    def test_verbosity_0_level_0(self):
        self.assertTrue(MP.message(verbosity = 0, level = 0, printthis = 'hello world'))
        
    def test_verbosity_1_level_0(self):
        self.assertTrue(MP.message(verbosity = 1, level = 0, printthis = 'hello world'))
        
    def test_verbosity_0_level_1(self):
        self.assertFalse(MP.message(verbosity = 0, level = 1, printthis = 'hello world'))
        
# --------------------------
MP = ModelParameters()
class LessThanOrEqualToZero(unittest.TestCase):
    
    def test_x_lt_0(self):
        self.assertTrue(MP.less_than_or_equal_to_zero(x = -1))
        
    def test_x_eq_0(self):
        self.assertTrue(MP.less_than_or_equal_to_zero(x = 0))
        
    def test_x_gt_0(self):
        self.assertFalse(MP.less_than_or_equal_to_zero(x = 1))

# --------------------------
MP = ModelParameters()
class ReplaceListElements(unittest.TestCase):
    
    def test_list_of_0s(self):
        x = [0, 0, 0]
        self.assertEqual(MP.replace_list_elements(x = x, testfunction = MP.less_than_or_equal_to_zero, value = 10), [10, 10, 10])
        
    def test_list_of_neg(self):
        x = [-1, -1, -1]
        self.assertEqual(MP.replace_list_elements(x = x, testfunction = MP.less_than_or_equal_to_zero, value = 10), [10, 10, 10])
        
    def test_list_of_pos(self):
        x = [1, 1, 1]
        self.assertEqual(MP.replace_list_elements(x = x, testfunction = MP.less_than_or_equal_to_zero, value = 10), x)
        
    def test_list_of_mixed(self):
        x = [1, 0, -1]
        self.assertEqual(MP.replace_list_elements(x = x, testfunction = MP.less_than_or_equal_to_zero, value = 10), [1, 10, 10])
        
# --------------------------
MP = ModelParameters()
class DisplayParametersSettings(unittest.TestCase):
    
    def test_parameter_display_set_1(self):
        MP.add_model_parameter(name = 'm', theta0 = 2., minimum = -10, maximum = np.inf, sample = 1)
        MP.add_model_parameter(name = 'b', theta0 = -5., minimum = -10, maximum = 100, sample = 0)
        MP.add_model_parameter(name = 'b2', theta0 = -5.3e6, minimum = -1e7, maximum = 1e6, sample = 1)
        MP._openparameterstructure(nbatch = 1)
        
        MP.display_parameter_settings(verbosity = 1, noadaptind = None)