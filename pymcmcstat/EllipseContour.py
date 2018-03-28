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

    def generate_ellipse(self, mu, cmat):
        '''
        %ELLIPSE Draws a probability contour ellipse
        % ellipse(mu,cmat) draws an ellipse centered at mu and with
        % covariance matrix cmat.
        % h = ellipse(...) plots and returns handle to the ellipse line
        % [x,y] = ellipse(...) returns ellipse points (no plot)
        % use chiqf(0.9,2)*cmat to get 90% Gaussian probability region
        % example:
        % plot(x(:,1),x(:,2),'.'); % plot data points
        % hold on; ellipse(xmean, chiqf(0.9,2)*xcov); hold off
        
        % Marko Laine <Marko.Laine@Helsinki.FI>
        % $Revision: 1.6 $  $Date: 2007/08/10 08:50:40 $
        
        Modified for Python by Paul Miles
        '''
        
        # check shape of covariance matrix
        if cmat.shape != (2,2):
            sys.exit('covmat must be 2x2')
        
        if self.__check_symmetric(cmat) is not True:
            sys.exit('covmat must be symmetric')
            
        
        # define t space
        t = np.linspace(0, 2*pi, 100)
        R = np.linalg.cholesky(cmat).T
        
        x = mu[0] + R[0,0]*cos(t)
        y = mu[1] + R[0,1]*cos(t) + R[1,1]*sin(t)
        
        return x, y
    
    def __check_symmetric(self, a, tol=1e-8):
        return np.allclose(a, a.T, atol = tol)