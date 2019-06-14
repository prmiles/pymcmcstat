#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 12:07:11 2018

Utility functions used by different samplers

@author: prmiles
"""

import numpy as np


# --------------------------------------------------------
def sample_candidate_from_gaussian_proposal(npar, oldpar, R):
    '''
    Sample candidate from Gaussian proposal distribution

    Args:
        * **npar** (:py:class:`int`): Number of parameters being samples
        * **oldpar** (:class:`~numpy.ndarray`): :math:`q^{k-1}` Old parameter set.
        * **R** (:class:`~numpy.ndarray`): Cholesky decomposition of parameter covariance matrix.

    Returns:
        * **newpar** (:class:`~numpy.ndarray`): :math:`q^*` - candidate
        * **npar_sample_from_sample** (:class:`~numpy.ndarray`): \
        Sampled values from normal distibution: :math:`N(0,1)`.
    '''
    npar_sample_from_normal = np.random.randn(1, npar)
    newpar = oldpar + np.dot(npar_sample_from_normal, R)
    newpar = newpar.reshape(npar)
    return newpar, npar_sample_from_normal


# --------------------------------------------------------
def is_sample_outside_bounds(theta, lower_limits, upper_limits):
    '''
    Check whether proposal value is outside parameter limits

    Args:
        * **theta** (:class:`~numpy.ndarray`): Value of sampled model parameters
        * **lower_limits** (:class:`~numpy.ndarray`): Lower limits
        * **upper_limits** (:class:`~numpy.ndarray`): Upper limits

    Returns:
        * **outsidebounds** (:py:class:`bool`): True -> Outside of parameter limits
    '''
    if (theta < lower_limits).any() or (theta > upper_limits).any():
        outsidebounds = True
    else:
        outsidebounds = False
    return outsidebounds


# --------------------------------------------------------
def acceptance_test(alpha):
    '''
    Run standard acceptance test

    .. math::

        & \\text{If}~u_{\\alpha} <~\\alpha, \\

        & \\quad \\text{Set}~q^k = q^*,~SS_{q^k} = SS_{q^*} \\

        & \\text{Else} \\

        & \\quad \\text{Set}~q^k = q^{k-1},~SS_{q^k} = SS_{q^{k-1}}

    Args:
        * **alpha** (:py:class:`float`): Result of likelihood function

    Returns:
        * **accept** (:py:class:`bool`): False - reject, True - accept
    '''
    print('This function was deprecated in v1.8.0')
    if alpha >= 1 or alpha > np.random.rand(1, 1):
        accept = True
    else:
        accept = False
    return accept


# -------------------------------------------
def set_outside_bounds(next_set):
    '''
    Assign set features based on being outside bounds

    Args:
        * **next_set** (:class:`~.ParameterSet`): :math:`q^*`

    Returns:
        * **next_set** (:class:`~.ParameterSet`): :math:`q^*` with updated features
        * **outbound** (:class:`bool`): True
    '''
    next_set.alpha = 0
    next_set.prior = 0
    next_set.ss = np.inf
    outbound = True

    return next_set, outbound


# --------------------------------------------------------
def posterior_ratio_acceptance_test(alpha):
    '''
    Run posterior ratio acceptance test

    Args:
        * **alpha** (:py:class:`float`): Posterior ratio
    Returns:
        * **accept** (:py:class:`bool`): False - reject, True - accept
    '''
    if alpha >= 1.0 or alpha > np.random.rand(1, 1):
        return True
    else:
        return False


# --------------------------------------------------------
def log_posterior_ratio_acceptance_test(alpha):
    '''
    Run log posterior ratio acceptance test

    Args:
        * **alpha** (:py:class:`float`): Log posterior ratio
    Returns:
        * **accept** (:py:class:`bool`): False - reject, True - accept
    '''
    if alpha > np.log(np.random.rand(1, 1)):
        return True
    else:
        return False


# -------------------------------------------
def calculate_log_posterior_ratio(loglikestar, loglike,
                                  logpriorstar, logprior):
    '''
    Calculate log posterior ratio:

    .. math::

        \\log(\\alpha) = \\min\\Big[0, \\log(\\mathcal{L}(\\nu_{obs}|q^*)) \
        + \\log(\\pi_0(q^*)) - \\log(\\mathcal{L}(\\nu_{obs}|q^{k-1})) \
        - \\log(\\pi_0(q^{k-1}))\\Big]

    For more details regarding the prior and likelihood functions,
    please refer to the :class:`~.PriorFunction` and
    :class:`~.LikelihoodFunction` class, respectively.

    .. note::
        The default behavior of the package is to use Gaussian
        likelihood and prior functions (as of v1.8.0).  Future releases
        will expand the functionality to allow for alternative likelihood
        and prior definitions.

    Args:
        * **likestar** (:py:class:`float`): \
        Likelihood from proposed candidate, :math:`q^*`
        * **like** (:py:class:`float`): \
        Likelihood from previous sample point, :math:`q^{k-1}`
        * **priorstar** (:py:class:`float`): Prior for proposal candidate
        * **prior** (:py:class:`float`): Prior for previous sample

    Returns:
        * **alpha** (:py:class:`float`): Acceptance ratio
    '''
    logalpha = loglikestar + logpriorstar - (loglike + logprior)
    return logalpha
