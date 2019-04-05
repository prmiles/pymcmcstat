#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 11:14:11 2018

@author: prmiles
"""
# import required packages
import numpy as np
import math
from ..utilities.general import message


# --------------------------------------------
class Adaptation:
    '''
    Adaptive Metropolis (AM) algorithm based on :cite:`haario2001adaptive`.

    Attributes:
        * :meth:`~run_adaptation`
    '''
    def __init__(self):
        self.label = 'Covariance Variables and Methods'
        self.qcov = None
        self.qcov_scale = None
        self.R = None
        self.qcov_original = None
        self.invR = None
        self.iacce = None
        self.covchain = None
        self.meanchain = None
        self.last_index_since_adaptation = 0

    # -------------------------------------------
    def run_adaptation(self, covariance, options, isimu, iiadapt, rejected, chain, chainind, u, npar, alpha):
        '''
        Run adaptation step

        Args:
            * **covariance** (:class:`~.CovarianceProcedures`): Covariance methods and variables
            * **options** (:class:`~.SimulationOptions`): Options for MCMC simulation
            * **isimu** (:py:class:`int`): Simulation counter
            * **iiadapt** (:py:class:`int`): Adaptation counter
            * **rejected** (:py:class:`dict`): Rejection counter
            * **chain** (:class:`~numpy.ndarray`): Sampling chain
            * **chainind** (:py:class:`ind`): Relative point in chain
            * **u** (:class:`~numpy.ndarray`): Latest random sample points
            * **npar** (:py:class:`int`): Number of parameters being sampled
            * **alpha** (:py:class:`float`): Latest Likelihood evaluation

        Returns:
            * **covariance** (:class:`~.CovarianceProcedures`): Updated covariance object
        '''
        # unpack options
        (burnintime, burnin_scale, ntry, drscale, alphatarget,
         etaparam, qcov_adjust, doram, verbosity) = unpack_simulation_options(options=options)

        # unpack covariance
        (last_index_since_adaptation, R, oldcovchain, oldmeanchain, oldwsum,
         no_adapt_index, qcov_scale, qcov) = unpack_covariance_settings(covariance=covariance)

        if isimu < burnintime:
            R = below_burnin_threshold(rejected=rejected, iiadapt=iiadapt, R=R,
                                       burnin_scale=burnin_scale, verbosity=verbosity)
            covchain = oldcovchain
            meanchain = oldmeanchain
            wsum = oldwsum
            RDR = None
            invR = None
        else:
            message(verbosity, 3, str('i:{} adapting ({}, {}, {})'.format(
                    isimu, rejected['total']*(isimu**(-1))*100, rejected['in_adaptation_interval']*(iiadapt**(-1))*100,
                    rejected['outside_bounds']*(isimu**(-1))*100)))
            # UPDATE COVARIANCE MATRIX - CHOLESKY, MEAN, SUM
            covchain, meanchain, wsum = update_covariance_mean_sum(
                    chain[last_index_since_adaptation:chainind, :], np.ones(1), oldcovchain, oldmeanchain, oldwsum)
            last_index_since_adaptation = isimu
            # ram
            if doram:
                upcov = update_cov_via_ram(u=u, isimu=isimu, etaparam=etaparam,
                                           npar=npar, alphatarget=alphatarget, alpha=alpha, R=R)
            else:
                upcov = update_cov_from_covchain(covchain=covchain, qcov=qcov, no_adapt_index=no_adapt_index)

            # check if singular covariance matrix
            R = check_for_singular_cov_matrix(upcov=upcov, R=R, npar=npar,
                                              qcov_adjust=qcov_adjust, qcov_scale=qcov_scale,
                                              rejected=rejected, iiadapt=iiadapt, verbosity=verbosity)
            # update dram covariance matrix
            RDR, invR = update_delayed_rejection(R=R, npar=npar, ntry=ntry, drscale=drscale)

        covariance._update_covariance_from_adaptation(R, covchain, meanchain, wsum,
                                                      last_index_since_adaptation, iiadapt)
        covariance._update_covariance_for_delayed_rejection_from_adaptation(RDR=RDR, invR=invR)
        return covariance


# --------------------------------------------
def unpack_simulation_options(options):
    '''
    Unpack simulation options

    Args:
        * **options** (:class:`~.SimulationOptions`): Options for MCMC simulation

    Returns:
        * **burnintime** (:py:class:`int`):
        * **burnin_scale** (:py:class:`float`): Scale for burnin.
        * **ntry** (:py:class:`int`): Number of tries to take before rejection. Default is method dependent.
        * **drscale** (:class:`~numpy.ndarray`): Reduced scale for sampling in DR algorithm. Default is [5,4,3].
        * **alphatarget** (:py:class:`float`): Acceptance ratio target.
        * **etaparam** (:py:class:`float`):
        * **qcov_adjust** (:py:class:`float`): Adjustment scale for covariance matrix.
        * **doram** (:py:class:`int`): Flag to perform :code:`'ram'` algorithm (Obsolete).
        * **verbosity** (:py:class:`int`): Verbosity of display output.
    '''
    burnintime = options.burnintime
    burnin_scale = options.burnin_scale
    ntry = options.ntry
    drscale = options.drscale
    alphatarget = options.alphatarget
    etaparam = options.etaparam
    qcov_adjust = options.qcov_adjust
    doram = options.doram
    verbosity = options.verbosity
    return burnintime, burnin_scale, ntry, drscale, alphatarget, etaparam, qcov_adjust, doram, verbosity


# --------------------------------------------
def unpack_covariance_settings(covariance):
    '''
    Unpack covariance settings

    Args:
        * **covariance** (:class:`~.CovarianceProcedures`): Covariance methods and variables

    Returns:
        * **last_index_since_adaptation** (:py:class:`int`): Last index since adaptation occured.
        * **R** (:class:`~numpy.ndarray`): Cholesky decomposition of covariance matrix.
        * **oldcovchain** (:class:`~numpy.ndarray`): Covariance matrix history.
        * **oldmeanchain** (:class:`~numpy.ndarray`): Current mean chain values.
        * **oldwsum** (:class:`~numpy.ndarray`): Weights
        * **no_adapt_index** (:class:`numpy.ndarray`): Indices of parameters not being adapted.
        * **qcov_scale** (:py:class:`float`): Scale parameter
        * **qcov** (:class:`~numpy.ndarray`): Covariance matrix
    '''
    # unpack covariance
    last_index_since_adaptation = covariance._last_index_since_adaptation
    R = covariance._R
    oldcovchain = covariance._covchain
    oldmeanchain = covariance._meanchain
    oldwsum = covariance._wsum
    no_adapt_index = covariance._no_adapt_index
    qcov_scale = covariance._qcov_scale
    qcov = covariance._qcov
    return last_index_since_adaptation, R, oldcovchain, oldmeanchain, oldwsum, no_adapt_index, qcov_scale, qcov


# --------------------------------------------
def below_burnin_threshold(rejected, iiadapt, R, burnin_scale, verbosity):
    '''
    Update Cholesky Matrix using below burnin thershold

    Args:
        * **rejected** (:py:class:`dict`): Rejection counters.
        * **iiadapt** (:py:class:`int`): Adaptation counter
        * **R** (:class:`~numpy.ndarray`): Cholesky decomposition of covariance matrix.
        * **burnin_scale** (:py:class:`float`): Scale for burnin.
        * **verbosity** (:py:class:`int`): Verbosity of display output.

    Returns:
        * **R** (:class:`~numpy.ndarray`): Scaled Cholesky matrix.
    '''
    # during burnin no adaptation, just scaling down
    if rejected['in_adaptation_interval']*(iiadapt**(-1)) > 0.95:
        message(verbosity, 2, str(' (burnin/down) {}'.format(rejected['in_adaptation_interval']*(iiadapt**(-1))*100)))
        R = R*(burnin_scale**(-1))
    elif rejected['in_adaptation_interval']*(iiadapt**(-1)) < 0.05:
        message(verbosity, 2, str(' (burnin/up) {}'.format(
                rejected['in_adaptation_interval']*(iiadapt**(-1))*100)))
        R = R*burnin_scale
    return R


# --------------------------------------------
def update_covariance_mean_sum(x, w, oldcov, oldmean, oldwsum, oldR=None):
    '''
    Update covariance chain, local mean, local sum

    Args:
        * **x** (:class:`~numpy.ndarray`): Chain segment
        * **w** (:class:`~numpy.ndarray`): Weights
        * **oldcov** (:class:`~numpy.ndarray` or `None`): Previous covariance matrix
        * **oldmean** (:class:`~numpy.ndarray`): Previous mean chain values
        * **oldwsum** (:class:`~numpy.ndarray`): Previous weighted sum
        * **oldR** (:class:`~numpy.ndarray`): Previous Cholesky decomposition matrix

    Returns:
        * **xcov** (:class:`~numpy.ndarray`): Updated covariance matrix
        * **xmean** (:class:`~numpy.ndarray`): Updated mean chain values
        * **wsum** (:class:`~numpy.ndarray`): Updated weighted sum
    '''
    n, p = x.shape
    if n == 0 or p == 0:  # nothing to update with
        return oldcov, oldmean, oldwsum
    w, R = setup_w_R(w=w, oldR=oldR, n=n)
    if oldcov is None:
        xcov, xmean, wsum = initialize_covariance_mean_sum(x, w)
    else:
        for ii in range(0, n):
            xi = x[ii, :]
            wsum = w[ii]
            xmean = oldmean + wsum*((wsum+oldwsum)**(-1.0))*(xi - oldmean)
            if R is not None:
                Rin, xin = setup_cholupdate(R=R, oldwsum=oldwsum,
                                            wsum=wsum, oldmean=oldmean, xi=xi)
                R = cholupdate(Rin, xin)
            xcov = (((oldwsum-1)*((wsum + oldwsum - 1)**(-1)))*oldcov + (wsum*oldwsum*((wsum
                    + oldwsum-1)**(-1))) * ((wsum + oldwsum)**(-1)) * (
                np.dot((xi-oldmean).reshape(p, 1), (xi-oldmean).reshape(1, p))))
            wsum = wsum + oldwsum
            oldcov = xcov
            oldmean = xmean
            oldwsum = wsum
    return xcov, xmean, wsum


# --------------------------------------------
def setup_w_R(w, oldR, n):
    '''
    Setup weights and Cholesky matrix

    Args:
        * **x** (:class:`~numpy.ndarray`): Chain segment
        * **w** (:class:`~numpy.ndarray`): Weights
        * **oldcov** (:class:`~numpy.ndarray` or `None`): Previous covariance matrix
        * **oldmean** (:class:`~numpy.ndarray`): Previous mean chain values
        * **oldwsum** (:class:`~numpy.ndarray`): Previous weighted sum
        * **oldR** (:class:`~numpy.ndarray`): Previous Cholesky decomposition matrix

    Returns:
        * **w** (:class:`~numpy.ndarray`): Weights
        * **R** (:class:`~numpy.ndarray`): Previous Cholesky decomposition matrix
    '''
    if w is None:
        w = np.ones(1)
    if len(w) == 1:
        w = np.ones(n)*w
    if oldR is None:
        R = None
    else:
        R = oldR
    return w, R


# --------------------------------------------
def initialize_covariance_mean_sum(x, w):
    '''
    Initialize covariance chain, local mean, local sum

    Args:
        * **x** (:class:`~numpy.ndarray`): Chain segment
        * **w** (:class:`~numpy.ndarray`): Weights

    Returns:
        * **xcov** (:class:`~numpy.ndarray`): Initial covariance matrix
        * **xmean** (:class:`~numpy.ndarray`): Initial mean chain values
        * **wsum** (:class:`~numpy.ndarray`): Initial weighted sum
    '''
    npar = x.shape[1]
    wsum = sum(w)
    xmean = np.zeros(npar)
    xcov = np.zeros([npar, npar])
    for ii in range(npar):
        xmean[ii] = sum(x[:, ii]*w)*(wsum**(-1))
    if wsum > 1:
        for ii in range(0, npar):
            for jj in range(0, ii+1):
                term1 = x[:, ii] - xmean[ii]
                term2 = x[:, jj] - xmean[jj]
                xcov[ii, jj] = np.dot(term1.transpose(), ((term2)*w)*((wsum-1)**(-1)))
                if ii != jj:
                    xcov[jj, ii] = xcov[ii, jj]
    return xcov, xmean, wsum


# --------------------------------------------
def setup_cholupdate(R, oldwsum, wsum, oldmean, xi):
    '''
    Setup input arguments to the Cholesky update function

    Args:
        * **R** (:class:`~numpy.ndarray`): Previous Cholesky decomposition matrix
        * **oldwsum** (:class:`~numpy.ndarray`): Previous weighted sum
        * **w** (:class:`~numpy.ndarray`): Current Weights
        * **oldmean** (:class:`~numpy.ndarray`): Previous mean chain values
        * **xi** (:class:`~numpy.ndarray`): Row of chain segment

    Returns:
        * **Rin** (:class:`~numpy.ndarray`): Scaled Cholesky decomposition matrix
        * **xin** (:class:`~numpy.ndarray`): Updated mean chain values for Cholesky function
    '''
    Rin = np.sqrt((oldwsum-1)*((wsum+oldwsum-1)**(-1)))*R
    xin = (xi - oldmean).transpose()*np.sqrt(((wsum*oldwsum)*((wsum+oldwsum-1)**(-1))*((wsum+oldwsum)**(-1))))
    return Rin, xin


# --------------------------------------------
# Cholesky Update
def cholupdate(R, x):
    '''
    Update Cholesky decomposition

    Args:
        * **R** (:class:`~numpy.ndarray`): Weighted Cholesky decomposition
        * **x** (:class:`~numpy.ndarray`): Weighted sum based on local chain update

    Returns:
        * **R1** (:class:`~numpy.ndarray`): Updated Cholesky decomposition
    '''
    n = len(x)
    R1 = R.copy()
    x1 = x.copy()
    for ii in range(n):
        r = math.sqrt(R1[ii, ii]**2 + x1[ii]**2)
        c = r*(R1[ii, ii]**(-1))
        s = x1[ii]*(R1[ii, ii]**(-1))
        R1[ii, ii] = r
        if ii+1 < n:
            R1[ii, ii+1:n] = (R1[ii, ii+1:n] + s*x1[ii+1:n])*(c**(-1))
            x1[ii+1:n] = c*x1[ii+1:n] - s*R1[ii, ii+1:n]

    return R1


# --------------------------------------------
def update_cov_via_ram(u, isimu, etaparam, npar, alphatarget, alpha, R):
    '''
    Update covariance matrix via RAM

    Args:
        * **u** (:class:`~numpy.ndarray`): Latest random sample points
        * **isimu** (:py:class:`int`): Simulation counter
        * **alphatarget** (:py:class:`float`): Acceptance ratio target.
        * **npar** (:py:class:`int`): Number of parameters.
        * **etaparam** (:py:class:`float`):
        * **alpha** (:py:class:`float`): Latest Likelihood evaluation
        * **R** (:class:`~numpy.ndarray`): Cholesky decomposition of covariance matrix.

    Returns:
        * **upcov** (:class:`~numpy.ndarray`): Updated parameter covariance matrix.
    '''
    uu = u*(np.linalg.norm(u)**(-1))
    eta = (isimu**(etaparam))**(-1)
    ram = np.eye(npar) + eta*(min(1.0, alpha) - alphatarget)*(np.dot(uu.transpose(), uu))
    upcov = np.dot(np.dot(R.transpose(), ram), R)
    return upcov


# --------------------------------------------
def update_cov_from_covchain(covchain, qcov, no_adapt_index):
    '''
    Update covariance matrix from covariance matrix chain

    Args:
        * **covchain** (:class:`~numpy.ndarray`): Covariance matrix history.
        * **qcov** (:class:`~numpy.ndarray`): Parameter covariance matrix
        * **no_adapt_index** (:class:`numpy.ndarray`): Indices of parameters not being adapted.

    Returns:
        * **upcov** (:class:`~numpy.ndarray`): Updated covariance matrix
    '''
    upcov = covchain.copy()
    upcov[no_adapt_index, :] = qcov[no_adapt_index, :]
    upcov[:, no_adapt_index] = qcov[:, no_adapt_index]
    return upcov


# --------------------------------------------
def check_for_singular_cov_matrix(upcov, R, npar, qcov_adjust, qcov_scale, rejected, iiadapt, verbosity):
    '''
    Check if singular covariance matrix

    Args:
        * **upcov** (:class:`~numpy.ndarray`): Parameter covariance matrix.
        * **R** (:class:`~numpy.ndarray`): Cholesky decomposition of covariance matrix.
        * **npar** (:py:class:`int`): Number of parameters.
        * **qcov_adjust** (:py:class:`float`): Covariance adjustment factor.
        * **qcov_scale** (:py:class:`float`): Scale parameter
        * **rejected** (:py:class:`dict`): Rejection counters.
        * **iiadapt** (:py:class:`int`): Adaptation counter.
        * **verbosity** (:py:class:`int`): Verbosity of display output.

    Returns:
        * **R** (:class:`~numpy.ndarray`): Adjusted Cholesky decomposition of covariance matrix.
    '''
    pos_def, pRa = is_semi_pos_def_chol(upcov)
    if pos_def == 1:  # not singular!
        return scale_cholesky_decomposition(Ra=pRa, qcov_scale=qcov_scale)
    else:  # singular covariance matrix
        return adjust_cov_matrix(upcov=upcov, R=R, npar=npar, qcov_adjust=qcov_adjust,
                                 qcov_scale=qcov_scale, rejected=rejected, iiadapt=iiadapt,
                                 verbosity=verbosity)


# --------------------------------------------
def is_semi_pos_def_chol(x):
    '''
    Check if matrix is semi-positive definite using Cholesky Decomposition

    Args:
        * **x** (:class:`~numpy.ndarray`): Covariance matrix

    Returns:
        * `Boolean`
        * **c** (:class:`~numpy.ndarray`): Cholesky decomposition (upper triangular form) or `None`
    '''
    c = None
    try:
        c = np.linalg.cholesky(x)
        return True, c.transpose()
    except np.linalg.linalg.LinAlgError:
        return False, c


# --------------------------------------------
def scale_cholesky_decomposition(Ra, qcov_scale):
    '''
    Scale Cholesky decomposition

    Args:
        * **R** (:class:`~numpy.ndarray`): Cholesky decomposition of covariance matrix.
        * **qcov_scale** (:py:class:`float`): Scale factor

    Returns:
        * **R** (:class:`~numpy.ndarray`): Scaled Cholesky decomposition of covariance matrix.
    '''
    return Ra*qcov_scale


# --------------------------------------------
def adjust_cov_matrix(upcov, R, npar, qcov_adjust, qcov_scale, rejected, iiadapt, verbosity):
    '''
    Adjust covariance matrix if found to be singular.

    Args:
        * **upcov** (:class:`~numpy.ndarray`): Parameter covariance matrix.
        * **R** (:class:`~numpy.ndarray`): Cholesky decomposition of covariance matrix.
        * **npar** (:py:class:`int`): Number of parameters.
        * **qcov_adjust** (:py:class:`float`): Covariance adjustment factor.
        * **qcov_scale** (:py:class:`float`): Scale parameter
        * **rejected** (:py:class:`dict`): Rejection counters.
        * **iiadapt** (:py:class:`int`): Adaptation counter.
        * **verbosity** (:py:class:`int`): Verbosity of display output.

    Returns:
        * **R** (:class:`~numpy.ndarray`): Cholesky decomposition of covariance matrix.
    '''
    # try to blow it up
    tmp = upcov + np.eye(npar)*qcov_adjust
    pos_def_adjust, pRa = is_semi_pos_def_chol(tmp)
    if pos_def_adjust == 1:  # not singular!
        message(verbosity, 1, 'adjusted covariance matrix')
        # scale R
        R = scale_cholesky_decomposition(Ra=pRa, qcov_scale=qcov_scale)
        return R
    else:  # still singular...
        errstr = str('covariance matrix singular, no adaptation')
        message(verbosity, 0, '{} {}'.format(errstr, rejected['in_adaptation_interval']*(iiadapt**(-1))*100))
        return R


# --------------------------------------------
def update_delayed_rejection(R, npar, ntry, drscale):
    '''
    Update Cholesky/Inverse matrix for Delayed Rejection

    Args:
        * **R** (:class:`~numpy.ndarray`): Cholesky decomposition of covariance matrix.
        * **npar** (:py:class:`int`): Number of parameters.
        * **ntry** (:py:class:`int`): Number of tries to take before rejection. Default is method dependent.
        * **drscale** (:class:`~numpy.ndarray`): Reduced scale for sampling in DR algorithm. Default is [5,4,3].

    Returns:
        * **RDR** (:py:class:`list`): List of Cholesky decomposition of covariance matrices for each stage of DR.
        * **InvR** (:py:class:`list`): List of Inverse Cholesky decomposition of \
        covariance matrices for each stage of DR.
    '''
    RDR = None
    invR = None
    if ntry > 1:  # delayed rejection
        RDR = []
        invR = []
        RDR.append(R)
        invR.append(np.linalg.solve(RDR[0], np.eye(npar)))
        for ii in range(1, ntry):
            RDR.append(RDR[ii-1]*((drscale[min(ii, len(drscale)) - 1])**(-1)))
            invR.append(invR[ii-1]*(drscale[min(ii, len(drscale)) - 1]))
    return RDR, invR
