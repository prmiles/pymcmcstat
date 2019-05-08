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
    def __init__(self, theta=None, alpha=None, like=None, prior=None):
        self.theta = theta
        self.alpha = alpha
        self.like = like
        self.prior = prior
