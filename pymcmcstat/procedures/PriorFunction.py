#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 09:10:21 2018

Description: Prior function

@author: prmiles
"""
# Import required packages
import numpy as np

class PriorFunction:
    def __init__(self, priorfun = None, mu = None, sigma = None):

        self.mu = mu
        self.sigma = sigma
        
        # Setup prior function and evaluate
        if priorfun is None:
            priorfun = self.default_priorfun
        
        self.priorfun = priorfun # function handle
            
    def default_priorfun(self, theta, mu, sigma):
        # consider converting everything to numpy array - should allow for optimized performance
        n = len(theta)
        pf = np.zeros(1)
        for ii in range(n):
            pf = pf + ((theta[ii]-mu[ii])*(sigma[ii]**(-1)))**2
        
#        pf = np.sum(((theta - mu)*(sigma**(-1)))**(2))
#        print('pf = {}, pftest = {}'.format(pf, pftest))
#        sys.exit()
        return pf
        
    def evaluate_prior(self, theta):
        return self.priorfun(theta, self.mu, self.sigma)