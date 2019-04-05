#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:30:29 2018

@author: prmiles
"""
# import required packages
import numpy as np
from ..structures.ParameterSet import ParameterSet
from .utilities import sample_candidate_from_gaussian_proposal
from .utilities import is_sample_outside_bounds, set_outside_bounds
from .utilities import acceptance_test


class Metropolis:
    '''
    .. |br| raw:: html

        <br>

    Pseudo-Algorithm:

        #. Sample :math:`z_k \\sim N(0,1)`
        #. Construct candidate :math:`q^* = q^{k-1} + Rz_k`
        #. Compute |br| :math:`\\quad SS_{q^*} = \\sum_{i=1}^N[v_i-f_i(q^*)]^2`
        #. Compute |br| :math:`\\quad \\alpha = \\min\\Big(1, e^{[SS_{q^*} - SS_{q^{k-1}}]/(2\\sigma^2_{k-1})}\\Big)`
        #. If :math:`u_{\\alpha} <~\\alpha,` |br|
            Set :math:`q^k = q^*,~SS_{q^k} = SS_{q^*}`
           Else
            Set :math:`q^k = q^{k-1},~SS_{q^k} = SS_{q^{k-1}}`

    Attributes:
        * :meth:`~acceptance_test`
        * :meth:`~run_metropolis_step`
        * :meth:`~unpack_set`
    '''
    # --------------------------------------------------------
    def run_metropolis_step(self, old_set, parameters, R, prior_object, sos_object, custom=None):
        '''
        Run Metropolis step.

        Args:
            * **old_set** (:class:`~.ParameterSet`): Features of :math:`q^{k-1}`
            * **parameters** (:class:`~.ModelParameters`): Model parameters
            * **R** (:class:`~numpy.ndarray`): Cholesky decomposition of parameter covariance matrix
            * **priorobj** (:class:`~.PriorFunction`): Prior function
            * **sosobj** (:class:`~.SumOfSquares`): Sum-of-Squares function

        Returns:
            * **accept** (:py:class:`int`): 0 - reject, 1 - accept
            * **newset** (:class:`~.ParameterSet`): Features of :math:`q^*`
            * **outbound** (:py:class:`int`): 1 - rejected due to sampling outside of parameter bounds
            * **npar_sample_from_normal** (:class:`~numpy.ndarray`): Latet random sample points
        '''
        # unpack oldset
        oldpar, ss, oldprior, sigma2 = self.unpack_set(old_set)

        # Sample new candidate from Gaussian proposal
        newpar, npar_sample_from_normal = sample_candidate_from_gaussian_proposal(
                npar=parameters.npar, oldpar=oldpar, R=R)
        # Reject points outside boundaries
        outsidebounds = is_sample_outside_bounds(newpar, parameters._lower_limits[parameters._parind[:]],
                                                 parameters._upper_limits[parameters._parind[:]])
        if outsidebounds is True:
            # proposed value outside parameter limits
            newset = ParameterSet(theta=newpar, sigma2=sigma2)
            newset, outbound = set_outside_bounds(next_set=newset)
            accept = False
        else:
            outbound = 0
            # prior SS for the new theta
            newprior = prior_object.evaluate_prior(newpar)
            # calculate sum-of-squares
            ss2 = ss  # old ss
            ss1 = sos_object.evaluate_sos_function(newpar, custom=custom)
            # evaluate likelihood
            alpha = self.evaluate_likelihood_function(ss1, ss2, sigma2, newprior, oldprior)
            # make acceptance decision
            accept = acceptance_test(alpha)
            # store parameter sets in objects
            newset = ParameterSet(theta=newpar, ss=ss1, prior=newprior, sigma2=sigma2, alpha=alpha)
        return accept, newset, outbound, npar_sample_from_normal

    # --------------------------------------------------------
    @classmethod
    def unpack_set(cls, parset):
        '''
        Unpack parameter set

        Args:
            * **parset** (:class:`~.ParameterSet`): Parameter set to unpack

        Returns:
            * **theta** (:class:`~numpy.ndarray`): Value of sampled model parameters
            * **ss** (:class:`~numpy.ndarray`): Sum-of-squares error using sampled value
            * **prior** (:class:`~numpy.ndarray`): Value of prior
            * **sigma2** (:class:`~numpy.ndarray`): Error variance
        '''
        theta = parset.theta
        ss = parset.ss
        prior = parset.prior
        sigma2 = parset.sigma2
        return theta, ss, prior, sigma2

    # --------------------------------------------------------
    @classmethod
    def evaluate_likelihood_function(cls, ss1, ss2, sigma2, newprior, oldprior):
        '''
        Evaluate likelihood function:

        .. math::

            \\alpha = \\exp\\Big[-0.5\\Big(\\sum\\Big(\\frac{ SS_{q^*} \
            - SS_{q^{k-1}} }{ \\sigma_{k-1}^2 }\\Big) + p_1 - p_2\\Big)\\Big]

        Args:
            * **ss1** (:class:`~numpy.ndarray`): SS error from proposed candidate, :math:`q^*`
            * **ss2** (:class:`~numpy.ndarray`): SS error from previous sample point, :math:`q^{k-1}`
            * **sigma2** (:class:`~numpy.ndarray`): Error variance estimate \
            from previous sample point, :math:`\\sigma_{k-1}^2`
            * **newprior** (:class:`~numpy.ndarray`): Prior for proposal candidate
            * **oldprior** (:class:`~numpy.ndarray`): Prior for previous sample

        Returns:
            * **alpha** (:py:class:`float`): Result of likelihood function
        '''
        alpha = np.exp(-0.5*(sum((ss1 - ss2)*(sigma2**(-1))) + newprior - oldprior))
        return sum(alpha)
