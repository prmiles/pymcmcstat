#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:15:37 2018

@author: prmiles
"""


class ParameterSet:
    '''
    Basic MCMC parameter set.

    **Description:** Storage device for passing parameter sets back and forth between sampling methods.

    Args:
        * **theta** (:class:`~numpy.ndarray`): Sampled values.
        * **ss** (:class:`~numpy.ndarray`): Sum-of-squares error(s).
        * **prior** (:class:`~numpy.ndarray`): Result from prior function.
        * **sigma2** (:class:`~numpy.ndarray`): Observation errors.
        * **alpha** (:py:class:`float`): Result from evaluating likelihood function.
    '''
    def __init__(self, theta=None, ss=None, prior=None, sigma2=None, alpha=None):
        self.theta = theta
        self.ss = ss
        self.prior = prior
        self.sigma2 = sigma2
        self.alpha = alpha
