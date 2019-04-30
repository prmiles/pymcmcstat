#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 05:31:57 2019

@author: prmiles
"""
import numpy as np
import sys


class LikelihoodFunction:
    '''
    Likelihood class

    Attributes:
        * :meth:`default_priorfun`
        * :meth:`evaluate_prior`
    '''
    def __init__(self, model, data, parameters):
        # check if user defined likelihood
        if (hasattr(model, 'likelihood') is False) or (model.likelihood is None):
            self.likelihood = self.default_gaussian_likelihood
#            self._sos_object = SumOfSquares(model, data, parameters)
            self.sos_function = model.sos_function
            self.model_function = model.model_function
            self.type = 'default'
        else:
            self.likelihood = model.likelihood
            self.type = 'custom'
        # copy reference arrays
        self.parind = parameters._parind.copy()
        self.value = parameters._initial_value.copy()
        self.local = parameters._local.copy()
        self.data = data

    def evaluate_likelihood(self, q, custom=None):
        # Copy sampled parameter values to full parameter array
        self.value[self.parind] = q
        try:
            like = self.likelihood(self.value, self.data, custom=custom)
        except TypeError:
            like = self.likelihood(self.value, self.data)
        return like

    def default_gaussian_likelihood(self, q, data, custom=None):
        '''
        Gaussian likelihood function

        .. math::

            \\mathcal{L}(\\nu_{obs}|q, \\sigma) = \
            \\exp\\Big(-\\frac{SS_q}{2\\sigma}\\Big)
        '''
        if custom is not None:
            sigma = custom[0]
        else:
            sigma = 1.0
        rawssq = self.evaluate_sos_function(q, data, custom=custom)
        # check if SOS is array and sum
        ssq = self._check_sos(rawssq)
        return np.exp(-1./2 * ssq.sum()/sigma**2)

    def evaluate_sos_function(self, q, data=None, custom=None):
        if data is None:
            data = self.data
        try:
            rawssq = self.sos_function(q, data, custom=custom)
        except TypeError:
            rawssq = self.sos_function(q, data)
        rawssq = self._check_sos(rawssq)
        return rawssq

    @classmethod
    def _check_sos(cls, ssq):
        if isinstance(ssq, float):
            return np.array([ssq])
        elif isinstance(ssq, np.ndarray):
            if np.greater(ssq.shape, 1).sum() <= 1:
                return ssq
            else:
                sys.exit(str('Expect numpy array of shape (n,) '
                             + 'or (n,1) -> {}'.format(ssq.shape)))
        elif isinstance(ssq, list):
            return np.array(ssq)
        else:
            valid = ['float', 'list', 'numpy array: (n,)', 'numpy array: (n, 1)']
            sys.exit(str('Return sum-of-squares as' + ''.join('\n\t{}'.format(v) for v in valid)))
