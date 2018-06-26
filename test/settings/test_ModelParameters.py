# -*- coding: utf-8 -*-

"""
Created on Mon Jan. 15, 2018

Description: This file contains a series of utilities designed to test the
features in the 'PredictionIntervals.py" package of the pymcmcstat module. The
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

    def test_does_parameter_assignment_match_with_no_name_or_initial_value(self):
        MP = ModelParameters()
        MP.add_model_parameter(name = None, theta0 = None)
        self.assertEqual(MP.parameters[0]['theta0'], 1.0)
        self.assertEqual(MP.parameters[0]['name'],'$p_{0}$')
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
        SO.define_simulation_options(verbosity=0)
        MP.display_parameter_settings(verbosity = SO.verbosity, noadaptind = SO.noadaptind)
        MP._results_to_params(results, 1)
        MP._openparameterstructure(nbatch=1)
        MP.display_parameter_settings(verbosity = SO.verbosity, noadaptind = SO.noadaptind)
        self.assertEqual(MP.parameters[0]['theta0'], results['theta'][0])
                
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
class DisplayParametersSettings(unittest.TestCase):
    
    def test_parameter_display_set_1(self):
        MP.add_model_parameter(name = 'm', theta0 = 2., minimum = -10, maximum = np.inf, sample = 1)
        MP.add_model_parameter(name = 'b', theta0 = -5., minimum = -10, maximum = 100, sample = 0)
        MP.add_model_parameter(name = 'b2', theta0 = -5.3e6, minimum = -1e7, maximum = 1e6, sample = 1)
        MP._openparameterstructure(nbatch = 1)
        
        MP.display_parameter_settings(verbosity = 1, noadaptind = None)
        
    def test_parameter_display_set_2(self):
        MP.add_model_parameter(name = 'm', theta0 = 2., minimum = -10, maximum = np.inf, sample = 1)
        MP.add_model_parameter(name = 'b', theta0 = -5., minimum = -10, maximum = 100, sample = 0)
        MP.add_model_parameter(name = 'b2', theta0 = -5.3e6, minimum = -1e7, maximum = 1e6, sample = 1)
        MP._openparameterstructure(nbatch = 1)
        
        MP.display_parameter_settings(verbosity = None, noadaptind = None)

# --------------------------
class NoadaptindDisplaySetting(unittest.TestCase):
    
    def test_noadaptind_is_empty(self):
        self.assertEqual(MP.noadapt_display_setting(ii = 1, noadaptind = []), '', msg = 'Default is blank string')
        self.assertEqual(MP.noadapt_display_setting(ii = 2, noadaptind = []), '', msg = 'Default is blank string')
        
    def test_noadaptind_is_not_empty(self):
        self.assertEqual(MP.noadapt_display_setting(ii = 1, noadaptind = [2]), '', msg = 'Default is blank string')
        self.assertEqual(MP.noadapt_display_setting(ii = 2, noadaptind = [2]), ' (*)', msg = 'Default is blank string')

# --------------------------        
class PriorDisplaySetting(unittest.TestCase):
    
    def test_x_is_inf(self):
        self.assertEqual(MP.prior_display_setting(x = np.inf), '', msg = 'Blank string if infinity.')
        
    def test_x_is_not_inf(self):
        self.assertEqual(MP.prior_display_setting(x = 2.0), '^2', msg = 'Raised to the second power if not infinity.')
        
# --------------------------        
class CheckNoadaptind(unittest.TestCase):
    
    def test_noadaptind_is_none(self):
        self.assertEqual(MP.check_noadaptind(noadaptind = None), [], msg = 'Returns empty list.')
        
    def test_noadaptind_is_not_none(self):
        self.assertEqual(MP.check_noadaptind(noadaptind = [1]), [1], msg = 'Returns input.')
        
# --------------------------        
class Verbosity(unittest.TestCase):
    
    def test_verbosity_is_none(self):
        self.assertEqual(MP.check_verbosity(verbosity = None), 0, msg = 'Returns 0.')
        
    def test_verbosity_is_not_none(self):
        self.assertEqual(MP.check_verbosity(verbosity = 1), 1, msg = 'Returns input.')
        
# --------------------------        
class ParameterLimits(unittest.TestCase):
    
    def test_initial_parameter_outside(self):
        MP = ModelParameters()
        MP.add_model_parameter('aa', 0, minimum = -10., maximum = 10)
        MP.add_model_parameter('bb', 11, minimum = -10., maximum = 10)
        MP._openparameterstructure(nbatch=1)
        with self.assertRaises(SystemExit, msg = 'Initial value outside of bounds.'):
            MP._check_initial_values_wrt_parameter_limits()
            
    def test_initial_parameter_okay(self):
        MP = ModelParameters()
        MP.add_model_parameter('aa', 0, minimum = -10., maximum = 10)
        MP.add_model_parameter('bb', 9, minimum = -10., maximum = 10)
        MP._openparameterstructure(nbatch=1)
        self.assertTrue(MP._check_initial_values_wrt_parameter_limits())
        
# --------------------------        
class GenerateDefaultName(unittest.TestCase):
    
    def test_default_name_generation(self):
        self.assertEqual(MP.generate_default_name(nparam = 0), '$p_{0}$', msg = '0 based naming convention.')
        self.assertEqual(MP.generate_default_name(nparam = 3), '$p_{3}$', msg = '0 based naming convention.')
        self.assertEqual(MP.generate_default_name(nparam = 2), '$p_{2}$', msg = '0 based naming convention.')
        self.assertEqual(MP.generate_default_name(nparam = 10), '$p_{10}$', msg = '0 based naming convention.')