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
    def __init__(self, priorfun=None, mu=np.array([0]), sigma=np.array([np.inf])):

        self.mu = mu
        self.sigma = sigma

        # Setup prior function and evaluate
        if priorfun is None:
            priorfun = self.default_priorfun

        self.priorfun = priorfun  # function handle

    @classmethod
    def default_priorfun(cls, theta, mu, sigma):
        '''
        Default prior function - Gaussian.

        .. math::

            \\pi_0(q) = \\frac{1}{\\sigma\\sqrt{2\\pi}}\\exp\\Big[\\Big(\\frac{\\theta - \\mu}{\\sigma^2}\\Big)^2\\Big]

        Args:
            * **theta** (:class:`~numpy.ndarray`): Current parameter values.
            * **mu** (:class:`~numpy.ndarray`): Prior mean.
            * **sigma** (:class:`~numpy.ndarray`): Prior standard deviation.
        '''
        # proposed numpy implementation
        res = (mu - theta)/sigma
        pf = np.dot(res.reshape(1, res.size), res.reshape(res.size, 1))
        return pf

    def evaluate_prior(self, theta):
        '''
        Evaluate the prior function.

        Args:
            * **theta** (:class:`~numpy.ndarray`): Current parameter values.
        '''
        return self.priorfun(theta, self.mu, self.sigma)
