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
# Model Parameters 
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
        
#    def test_does_non_square_matrix_return_error(self):
#        cmat = np.zeros([3,2])
#        mu = np.zeros([2,1])
#        with self.assertRaises(SystemExit):
#            EC.generate_ellipse(mu, cmat)
#            
#    def test_does_non_symmetric_matrix_return_error(self):
#        cmat = np.array([[3,2],[1,3]])
#        mu = np.zeros([2,1])
#        with self.assertRaises(SystemExit):
#            EC.generate_ellipse(mu, cmat)
#            
#    def test_does_non_positive_definite_matrix_return_error(self):
#        cmat = np.zeros([2,2])
#        mu = np.zeros([2,1])
#        with self.assertRaises(SystemExit):
#            EC.generate_ellipse(mu, cmat)
#            
#    def test_does_good_matrix_return_equal_sized_xy_arrays(self):
#        cmat = np.eye(2)
#        mu = np.zeros([2,1])
#        x,y = EC.generate_ellipse(mu, cmat)
#        self.assertEqual(x.shape,y.shape)
#        
#    def test_does_good_matrix_return_correct_size_array(self):
#        cmat = np.eye(2)
#        mu = np.zeros([2,1])
#        ndp = 50 # number of oints to generate ellipse shape
#        x,y = EC.generate_ellipse(mu, cmat, ndp)
#        self.assertEqual(x.size,ndp)