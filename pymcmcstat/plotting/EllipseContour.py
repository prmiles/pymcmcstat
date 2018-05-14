#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 13:43:17 2018

@author: prmiles
"""

import numpy as np
from scipy import pi,sin,cos
import sys

class EllipseContour:
    def __init__(self):
        self.description = 'Generates a probability contour ellipse'

    def generate_ellipse(self, mu, cmat, ndp = 100):
        '''
        % Generates points for a probability contour ellipse
        
        % Marko Laine <Marko.Laine@Helsinki.FI>
        % $Revision: 1.6 $  $Date: 2007/08/10 08:50:40 $
        
        Modified for Python by Paul Miles
        '''
        
        # check shape of covariance matrix
        if cmat.shape != (2,2):
            sys.exit('covariance matrix must be 2x2')
        
        if self.__check_symmetric(cmat) is not True:
            sys.exit('covariance matrix must be symmetric')
            
        
        # define t space
        t = np.linspace(0, 2*pi, ndp)
        
        pdflag, R = self.__is_semi_pos_def_chol(cmat)
        if pdflag is False:
            sys.exit('covariance matrix must be positive definite')
        
        x = mu[0] + R[0,0]*cos(t)
        y = mu[1] + R[0,1]*cos(t) + R[1,1]*sin(t)
        
        return x, y
    
    def __check_symmetric(self, a, tol=1e-8):
        return np.allclose(a, a.T, atol = tol)
    
    def __is_semi_pos_def_chol(self, x):
        c = None
        try:
            c = np.linalg.cholesky(x)
            return True, c.transpose()
        except np.linalg.linalg.LinAlgError:
            return False, c