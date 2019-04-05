#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:12:50 2018

@author: prmiles
"""
# import required packages
import numpy as np


class ErrorVarianceEstimator:
    '''
    Estimate observation errors.

    Attributes:
        * :meth:`~update_error_variance`
        * :meth:`~gammar`
        * :meth:`~gammar_mt`
    '''
    def __init__(self):
        self.description = 'Estimate error variance.'

    def update_error_variance(self, sos, model):
        '''
        Update observation error variance.

        **Strategy:** Treat error variance :math:`\\sigma^2` as parameter to be sampled.

        **Definition:** The property that the prior and posterior distributions have the
        same parametric form is termed conjugacy.

        Starting from the likelihood function, it can be shown

        .. math::

            \\sigma^2|(\\nu, q) \\sim \\text{Inv-Gamma}\\Big(\\frac{N_s + N}{2}, \
            \\frac{N_s\\sigma_{s}^2+ SS_q}{2}\\Big)

        where :math:`N_s` and :math:`\\sigma_{s}^2` are shape and scaling parameters,
        :math:`N` is the number of observations, and :math:`SS_q` is the sum-of-squares error.  For more details
        regarding the interpretation of :math:`N_s` and :math:`\\sigma_{s}^2`, please refer to
        :cite:`smith2013uncertainty` page 163.

        .. note::

            The variables :math:`N_s` and :math:`\\sigma_{s}^2` correspond
            to :code:`N0` and :code:`S20` in the :class:`~.ModelSettings` class, respectively.

        Args:
            * **sos** (:class:`~numpy.ndarray`): Return argument from evaluation of sum-of-squares function.
            * **model** (:class:`~.ModelSettings`): MCMC model settings.
        '''
        N0 = model.N0
        S20 = model.S20
        N = model.N
        sigma2 = model.sigma2  # initializes it as array
        nsos = len(sos)

        for jj in range(0, nsos):
            sigma2[jj] = (self.gammar(1, 1, 0.5*(N0[jj]+N[jj]),
                          2*((N0[jj]*S20[jj]+sos[jj])**(-1))))**(-1)
        return sigma2

    def gammar(self, m, n, a, b=1):
        '''
        Random deviates from gamma distribution.

        Returns a m x n matrix of random deviates from the Gamma
          distribution with shape parameter A and scale parameter B:

        .. math::

            p(x|A,B) = \\frac{B^{-A}}{\\Gamma(A)}*x^{A-1}*\\exp(-x/B)

        Args:
            * **m** (:py:class:`int`): Number of rows in return
            * **n** (:py:class:`int`): Number of columns in return
            * **a** (:py:class:`float`): Shape parameter
            * **b** (:py:class:`float`): Scaling parameter
        '''
        if a <= 0:  # special case
            y = np.zeros([m, n])
            return y

        y = self.gammar_mt(m, n, a, b)
        return y

    def gammar_mt(self, m, n, a, b=1):
        '''
        Wrapper routine for calculating random deviates from gamma distribution
        using method of Marsaglia and Tsang (2000) :cite:`marsaglia2000simple`.

        Args:
            * **m** (:py:class:`int`): Number of rows in return
            * **n** (:py:class:`int`): Number of columns in return
            * **a** (:py:class:`float`): Shape parameter
            * **b** (:py:class:`float`): Scaling parameter
        '''
        y = np.zeros([m, n])
        for jj in range(0, n):
            for ii in range(0, m):
                y[ii, jj] = self._gammar_mt1(a=a, b=b)
        return y

    def _gammar_mt1(self, a, b=1):
        '''
        Calculates random deviate from gamma distribution using method of
        Marsaglia and Tsang (2000).

        Args:
            * **a** (:py:class:`float`): Shape parameter
            * **b** (:py:class:`float`): Scaling parameter
        '''
        if a < 1:
            y = self._gammar_mt1(1+a, b)*np.random.rand(1)**(a**(-1))
            return y
        else:
            d = a - 3**(-1)
            c = (9*d)**(-0.5)
            while 1:
                while 1:
                    x = np.random.randn(1)
                    v = 1.0 + c*x
                    if v > 0:
                        break
                v = v**(3)
                u = np.random.rand(1)
                if u < 1.0 - 0.0331*x**(4):
                    break
                if np.log(u) < 0.5*x**2 + d*(1.0 - v + np.log(v)):
                    break
            y = b*d*v
            return y
