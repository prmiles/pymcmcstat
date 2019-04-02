#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:42:07 2018

@author: prmiles
"""
# import required packages
import numpy as np
from ..structures.ParameterSet import ParameterSet
from .utilities import sample_candidate_from_gaussian_proposal
from .utilities import is_sample_outside_bounds, set_outside_bounds
from .utilities import acceptance_test


class DelayedRejection:
    '''
    Delayed Rejection (DR) algorithm based on :cite:`haario2006dram`.

    Attributes:
        * :meth:`~run_delayed_rejection`
        * :meth:`~initialize_next_metropolis_step`
        * :meth:`~alphafun`
    '''
    # -------------------------------------------
    def run_delayed_rejection(self, old_set, new_set, RDR, ntry, parameters, invR, sosobj, priorobj, custom=None):
        '''
        Perform delayed rejection step - occurs in standard metropolis is not accepted.

        Args:
            * **old_set** (:class:`~.ParameterSet`): Features of :math:`q^{k-1}`
            * **new_set** (:class:`~.ParameterSet`): Features of :math:`q^*`
            * **RDR** (:class:`~numpy.ndarray`): Cholesky decomposition of parameter covariance matrix for DR steps
            * **ntry** (:py:class:`int`): Number of DR steps to perform until rejection
            * **parameters** (:class:`~.ModelParameters`): Model parameters
            * **invR** (:class:`~numpy.ndarray`): Inverse Cholesky decomposition matrix
            * **sosobj** (:class:`~.SumOfSquares`): Sum-of-Squares function
            * **priorobj** (:class:`~.PriorFunction`): Prior function

        Returns:
            * **accept** (:py:class:`int`): 0 - reject, 1 - accept
            * **out_set** (:class:`~.ParameterSet`): If accept == 1, then latest DR set; Else, :math:`q^k=q^{k-1}`
            * **outbound** (:py:class:`int`): 1 - rejected due to sampling outside of parameter bounds
        '''
        # create trypath
        trypath = [old_set, new_set]
        itry = 1  # dr step index
        accept = False  # initialize acceptance criteria
        while accept is False and itry < ntry:
            itry += 1  # update dr step index
            # initialize next step parameter set
            next_set = self.initialize_next_metropolis_step(
                    npar=parameters.npar, old_theta=old_set.theta, sigma2=new_set.sigma2, RDR=RDR[itry-1])

            # Reject points outside boundaries
            outsidebounds = is_sample_outside_bounds(next_set.theta,
                                                     parameters._lower_limits[parameters._parind[:]],
                                                     parameters._upper_limits[parameters._parind[:]])
            if outsidebounds is True:
                next_set, outbound = set_outside_bounds(next_set=next_set)
                trypath.append(next_set)
                out_set = old_set
                continue  # return to beginning of while loop

            # Evaluate new proposals
            outbound = 0
            next_set.ss = sosobj.evaluate_sos_function(next_set.theta, custom=custom)
            next_set.prior = priorobj.evaluate_prior(theta=next_set.theta)
            trypath.append(next_set)  # add set to trypath
            alpha = self.alphafun(trypath, invR)
            trypath[-1].alpha = alpha  # add propability ratio
            # check results of delayed rejection
            accept = acceptance_test(alpha=alpha)
            out_set = update_set_based_on_acceptance(accept, old_set=old_set, next_set=next_set)
            self.iacce[itry - 1] += accept  # if accepted, adds 1, if not, adds 0
        return accept, out_set, outbound

    # -------------------------------------------
    @classmethod
    def initialize_next_metropolis_step(cls, npar, old_theta, sigma2, RDR):
        '''
        Take metropolis step according to DR

        Args:
            * **npar** (:py:class:`int`): Number of parameters
            * **old_theta** (:class:`~numpy.ndarray`): `q^{k-1}`
            * **sigma2** (:py:class:`float`): Observation error variance
            * **RDR** (:class:`~numpy.ndarray`): Cholesky decomposition of parameter covariance matrix for DR steps
            * **itry** (:py:class:`int`): DR step counter

        Returns:
            * **next_set** (:class:`~.ParameterSet`): New proposal set
            * **u** (:class:`numpy.ndarray`): Numbers sampled from standard normal \
            distributions (:code:`u.shape = (1,npar)`)
        '''
        next_set = ParameterSet()
        next_set.theta, u = sample_candidate_from_gaussian_proposal(npar=npar, oldpar=old_theta, R=RDR)
        next_set.sigma2 = sigma2
        return next_set

    # -------------------------------------------
    def _initialize_dr_metrics(self, options):
        '''
        Initialize counting metrics for delayed rejection algorithm.

        Args:
            * **options** (:class:`~.SimulationOptions`): MCMC simulation options
        '''
        self.iacce = np.zeros(options.ntry, dtype=int)
        self.dr_step_counter = 0

    # -------------------------------------------
    def alphafun(self, trypath, invR):
        '''
        Calculate likelihood according to DR

        Args:
            * **trypath** (:py:class:`list`): Sequence of DR steps
            * **invR** (:class:`~numpy.ndarray`): Inverse Cholesky decomposition matrix

        Returns:
            * **alpha** (:py:class:`float`): Result of likelihood function according to delayed rejection
        '''
        self.dr_step_counter += 1
        stage = len(trypath) - 1  # The stage we're in, elements in trypath - 1
        # recursively compute past alphas
        a1 = 1.0  # initialize
        a2 = 1.0  # initialize
        for kk in range(0, stage-1):
            tmp1 = self.alphafun(trypath[0:(kk+2)], invR)
            a1 = a1*(1 - tmp1)
            tmp2 = self.alphafun(trypath[stage:stage-kk-2:-1], invR)
            a2 = a2*(1 - tmp2)
            if a2 == 0:  # we will come back with prob 1
                alpha = np.zeros(1)
                return alpha
        y = log_posterior_ratio(trypath[0], trypath[-1])
        for kk in range(stage):
            y = y + nth_stage_log_proposal_ratio(kk, trypath, invR)
        alpha = min(np.ones(1), np.exp(y)*a2*(a1**(-1)))
        return alpha


# -------------------------------------------
def nth_stage_log_proposal_ratio(iq, trypath, invR):
    '''
    Gaussian nth stage log proposal ratio.

    Logarithm of :math:`q_i(y_n,...,y_{n-j})/q_i(x,y_1,...,y_j)`

    Args:
        * **iq** (:py:class:`int`): Stage number.
        * **trypath** (:py:class:`list`): Sequence of DR steps
        * **invR** (:class:`~numpy.ndarray`): Inverse Cholesky decomposition matrix

    Returns:
        * **zq** (:py:class:`float`): Logarithm of Gaussian nth stage proposal ratio.
    '''
    stage = len(trypath) - 1 - 1  # - 1, iq;
    if stage == iq:  # shift index due to 0-indexing
        zq = np.zeros(1)  # we are symmetric
    else:
        iR = invR[iq]  # proposal^(-1/2)
        y1, y2, y3, y4 = extract_state_elements(iq=iq, stage=stage, trypath=trypath)
        zq = -0.5*((np.linalg.norm(np.dot(y4-y3, iR)))**2 - (np.linalg.norm(np.dot(y2-y1, iR)))**2)
    return zq


# -------------------------------------------
def extract_state_elements(iq, stage, trypath):
    '''
    Extract elements from tried paths.

    Args:
        * **iq** (:py:class:`int`): Stage number.
        * **stage** (:py:class:`int`): Number of stages - 2
        * **trypath** (:py:class:`list`): Sequence of DR steps
    '''
    y1 = trypath[0].theta
    y2 = trypath[iq + 1].theta  # check index
    y3 = trypath[stage + 1].theta
    y4 = trypath[stage - iq].theta
    return y1, y2, y3, y4


# -------------------------------------------
def log_posterior_ratio(x1, x2):
    '''
    Calculate the logarithm of the posterior ratio.

    Args:
        * **x1** (:class:`~.ParameterSet`): Old set - :math:`q^{k-1}`
        * **x2** (:class:`~.ParameterSet`): New set - :math:`q^*`

    Returns:
        * **zq** (:py:class:`float`): Logarithm of posterior ratio.
    '''
    zq = -0.5*(sum((x2.ss*(x2.sigma2**(-1.0)) - x1.ss*(x1.sigma2**(-1.0)))) + x2.prior - x1.prior)
    return sum(zq)


# -------------------------------------------
def update_set_based_on_acceptance(accept, old_set, next_set):
    '''
    Define output set based on acceptance

    .. math::

        & \\text{If}~u_{\\alpha} <~\\alpha, \\

        & \\quad \\text{Set}~q^k = q^*,~SS_{q^k} = SS_{q^*} \\

        & \\text{Else} \\

        & \\quad \\text{Set}~q^k = q^{k-1},~SS_{q^k} = SS_{q^{k-1}}

    Args:
        * **accept** (:py:class:`int`): 0 - reject, 1 - accept
        * **old_set** (:class:`~.ParameterSet`): Features of :math:`q^{k-1}`
        * **next_set** (:class:`~.ParameterSet`): New proposal set

    Returns:
        * **out_set** (:class:`~.ParameterSet`): If accept == 1, then latest DR set; Else, :math:`q^k=q^{k-1}`
    '''
    if accept is True:
        out_set = next_set
    else:
        out_set = old_set
    return out_set
