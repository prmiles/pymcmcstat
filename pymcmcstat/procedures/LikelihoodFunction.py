#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 05:31:57 2019

@author: prmiles
"""
import numpy as np
from .SumOfSquares import SumOfSquares


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
            self.__sos_object = SumOfSquares(model, data, parameters)
        else:
            self.likelihood = model.likelihood
        # copy reference arrays
        self.parind = parameters._parind.copy()
        self.value = parameters._initial_value.copy()
        self.local = parameters._local.copy()

    def default_gaussian_likelihood(self, q, sigma=1.0):
        '''
        Gaussian likelihood function

        .. math::

            \\mathcal{L}(\\nu_{obs}|q, \\sigma) = \
            \\exp\\Big(-\\frac{SS_q}{2\\sigma}\\Big)
        '''
        SSq = self.__sos_object.evaluate_sos_function(q)
        return np.exp(-1./2 * SSq/sigma**2)

    def evaluate_likelihood(self, q, custom=None):
        # Copy sampled parameter values to full parameter array
        self.value[self.parind] = q
        try:
            like = self.likelihood(self.value, self.data, custom=custom)
        except TypeError:
            like = self.likelihood(self.value, self.data)
        return like
