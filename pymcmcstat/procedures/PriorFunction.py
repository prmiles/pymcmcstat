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
    '''
    Prior distribution functions.

    Attributes:
        * :meth:`default_priorfun`
        * :meth:`evaluate_prior`
    '''
    def __init__(self, priorfun = None, mu = None, sigma = None):

        self.mu = mu
        self.sigma = sigma

        # Setup prior function and evaluate
        if priorfun is None:
            priorfun = self.default_priorfun

        self.priorfun = priorfun # function handle
        
    @classmethod
    def default_priorfun(cls, theta, mu, sigma):
        '''
        Default prior function.

        .. math::

            \\pi_0(q) = \sum_{i=1}^N \\Big(\\frac{\\theta_i - \\mu_i}{\\sigma_i^2}\\Big)^2

        Args:
            * **theta** (:class:`~numpy.ndarray`): Current parameter values.
            * **mu** (:class:`~numpy.ndarray`): Prior mean.
            * **sigma** (:class:`~numpy.ndarray`): Prior standard deviation.
        '''
        # consider converting everything to numpy array - should allow for optimized performance
        n = len(theta)
        
        if mu is None:
            mu = np.zeros([n,1])
        
        if sigma is None:
            sigma = np.ones([n,1])
        
        pf = np.zeros(1)
        for ii in range(n):
            pf = pf + ((theta[ii]-mu[ii])*(sigma[ii]**(-1)))**2

        return pf
        
    def evaluate_prior(self, theta):
        '''
        Evaluate the prior function.

        Args:
            * **theta** (:class:`~numpy.ndarray`): Current parameter values.
        '''
        return self.priorfun(theta, self.mu, self.sigma)