# -*- coding: utf-8 -*-

"""
Created on Mon Jan. 15, 2018

Description: This file contains a series of utilities designed to test the
features in the 'EllipseContour.py" package of the pymcmcstat module.  

@author: prmiles
"""
from pymcmcstat.EllipseContour import EllipseContour
import unittest
import numpy as np

EC = EllipseContour()

# --------------------------
# empirical_quantiles 
# --------------------------
class Generate_Ellipse_Test(unittest.TestCase):

    def test_does_non_square_matrix_return_error(self):
        cmat = np.zeros([3,2])
        mu = np.zeros([2,1])
        with self.assertRaises(SystemExit):
            EC.generate_ellipse(mu, cmat)
            
    def test_does_non_symmetric_matrix_return_error(self):
        cmat = np.array([[3,2],[1,3]])
        mu = np.zeros([2,1])
        with self.assertRaises(SystemExit):
            EC.generate_ellipse(mu, cmat)
            
    def test_does_non_positive_definite_matrix_return_error(self):
        cmat = np.zeros([2,2])
        mu = np.zeros([2,1])
        with self.assertRaises(SystemExit):
            EC.generate_ellipse(mu, cmat)
            
    def test_does_good_matrix_return_equal_sized_xy_arrays(self):
        cmat = np.eye(2)
        mu = np.zeros([2,1])
        x,y = EC.generate_ellipse(mu, cmat)
        self.assertEqual(x.shape,y.shape)
        
    def test_does_good_matrix_return_correct_size_array(self):
        cmat = np.eye(2)
        mu = np.zeros([2,1])
        ndp = 50 # number of oints to generate ellipse shape
        x,y = EC.generate_ellipse(mu, cmat, ndp)
        self.assertEqual(x.size,ndp)