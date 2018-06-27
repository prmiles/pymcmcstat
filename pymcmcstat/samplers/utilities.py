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
    npar_sample_from_normal = np.random.randn(1, npar)
    newpar = oldpar + np.dot(npar_sample_from_normal, R)
    newpar = newpar.reshape(npar)
    return newpar, npar_sample_from_normal

# --------------------------------------------------------
def is_sample_outside_bounds(theta, lower_limits, upper_limits):
    '''
    Check whether proposal value is outside parameter limits
    
    :Args:
        * **theta** (:class:`~numpy.ndarray`): Value of sampled model parameters
        * **lower_limits** (:class:`~numpy.ndarray`): Lower limits
        * **upper_limits** (:class:`~numpy.ndarray`): Upper limits
        
    \\
    
    :Returns:
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
        
    :Args:
        * **alpha** (:py:class:`float`): Result of likelihood function
    
    \\
    
    :Returns:
        * **accept** (:py:class:`int`): 0 - reject, 1 - accept
    '''
    if alpha >= 1 or alpha > np.random.rand(1,1):
        accept = 1
    else:
        accept = 0
        
    return accept