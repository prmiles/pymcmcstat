#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 10:23:51 2018

@author: prmiles
"""

# import required packages
import numpy as np
import numpy.matlib as npm
import sys
import scipy
import scipy.stats
from scipy.fftpack import fft
from ..plotting.utilities import generate_default_names, extend_names_to_match_nparam

# display chain statistics
def chainstats(chain = None, results = None, returnstats = False):
    '''
    Calculate chain statistics.

    Args:
        * **chain** (:class:`~numpy.ndarray`): Sampling chain.
        * **results** (:py:class:`dict`): Results from MCMC simulation.
        * **returnstats** (:py:class:`bool`): Flag to return statistics.

    Returns:
        * **stats** (:py:class:`dict`): Statistical measures of chain convergence.
    '''
    if chain is None:
        prst = str('No chain reported - run simulation first.')
        print(prst)
        return prst
    else:
        nsimu, npar = chain.shape
        names = get_parameter_names(npar, results)

        # calculate mean and standard deviation of entire chain
        meanii = []
        stdii = []
        for ii in range(npar):
            meanii.append(np.mean(chain[:,ii]))
            stdii.append(np.std(chain[:,ii]))
            
        # calculate batch mean standard deviation
        bmstd = batch_mean_standard_deviation(chain)
        mcerr = bmstd/np.sqrt(nsimu)
            
        # calculate geweke's MCMC convergence diagnostic
        z,p = geweke(chain)
        
        # estimate integrated autocorrelation time
        tau, m = integrated_autocorrelation_time(chain)
        
        # print statistics
        print_chain_statistics(names, meanii, stdii, mcerr, tau, p)
        
        # assign stats to dictionary
        stats = {'mean': list(meanii),
                'std': list(stdii),
                'bmstd': list(bmstd),
                'mcerr': list(mcerr),
                'geweke': {'z': list(z), 'p': list(p)},
                'iact': {'tau': list(tau), 'm': list(m)}
                }
        
        if returnstats is True:
            return stats
          
def print_chain_statistics(names, meanii, stdii, mcerr, tau, p):
    '''
    Print chain statistics to terminal window.

    Args:
        * **names** (:py:class:`list`): List of parameter names.
        * **meanii** (:py:class:`list`): Parameter mean values.
        * **stdii** (:py:class:`list`): Parameter standard deviation.
        * **mcerr** (:class:`~numpy.ndarray`): Normalized batch mean standard deviation.
        * **tau** (:class:`~numpy.ndarray`): Integrated autocorrelation time.
        * **p** (:class:`~numpy.ndarray`): Geweke's convergence diagnostic.

    Example display:

    ::

        ---------------------
        name      :       mean        std     MC_err        tau     geweke
        $p_{0}$   :     1.9680     0.0319     0.0013    36.3279     0.9979
        $p_{1}$   :     3.0818     0.0803     0.0035    37.1669     0.9961
        ---------------------
    '''
    npar = len(names)
    # print statistics
    print('\n---------------------')
    print('{:10s}: {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}'.format('name','mean','std', 'MC_err', 'tau', 'geweke'))
    for ii in range(npar):
        if meanii[ii] > 1e4:
            print('{:10s}: {:10.4g} {:10.4g} {:10.4f} {:10.4f} {:10.4f}'.format(names[ii],meanii[ii],stdii[ii],mcerr[ii],tau[ii],p[ii]))
        else:
            print('{:10s}: {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f}'.format(names[ii],meanii[ii],stdii[ii],mcerr[ii],tau[ii],p[ii]))
    print('---------------------')
        
# ----------------------------------------------------
def batch_mean_standard_deviation(chain, b = None):
    '''
    Standard deviation calculated from batch means

    Args:
        * **chain** (:class:`~numpy.ndarray`): Sampling chain.
        * **b** (:py:class:`int`): Step size.

    Returns:
        * **s** (:class:`~numpy.ndarray`): Batch mean standard deviation.
    '''
    
    nsimu, npar = chain.shape
    
    # determine step size
    if b is None:
        b = max(10,np.fix(nsimu/20))
        b = int(b)
        
    # define indices
    inds = np.arange(0,nsimu+1,b)
    nb = len(inds)-1
    if nb < 2:
        sys.exit('too few batches')
        
    # initialize batch mean array
    y = np.zeros([nb, npar])
    # calculate mean of each batch
    for ii in range(nb):
        y[ii,:] = np.mean(chain[inds[ii]:inds[ii+1],:],0)
        
    # calculate estimated standard deviation of MC estimate
    s = np.sqrt(sum((y - npm.repmat(np.mean(chain,0),nb,1))**2)/(nb-1)*b)
    
    return s
# ----------------------------------------------------
def geweke(chain, a = 0.1, b = 0.5):
    '''
    Geweke's MCMC convergence diagnostic

    Test for equality of the means of the first a% (default 10%) and
    last b% (50%) of a Markov chain - see :cite:`brooks1998assessing`.

    Args:
        * **chain** (:class:`~numpy.ndarray`): Sampling chain.
        * **a** (:py:class:`float`): First a% of chain.
        * **b** (:py:class:`float`): Last b% of chain.

    Returns:
        * **z** (:class:`~numpy.ndarray`): Convergence diagnostic prior to CDF.
        * **p** (:class:`~numpy.ndarray`): Geweke's MCMC convergence diagnostic.

    .. note::

        The percentage of the chain should be given as a decimal between zero and one.
        So, for the first 10% of the chain, define :code:`a = 0.1`.  Likewise,
        for the last 50% of the chain, define :code:`b = 0.5`.
    '''
    
    nsimu = chain.shape[0]
    
    # determine integer locations based on percentages
    na = int(np.floor(a*nsimu))
    nb = nsimu - int(np.floor(b*nsimu))
    
    # check sizes
    if (na + nb) >= nsimu:
        sys.exit('na + nb = {}, > nsimu = {}'.format(na+nb,nsimu))
    
    # calculate mean chain responses
    m1 = np.mean(chain[0:na,:],0)
    m2 = np.mean(chain[nb:nsimu+1,:],0)
    
    # calculate spectral estimates for variance
    sa = spectral_estimate_for_variance(chain[0:na,:])
    sb = spectral_estimate_for_variance(chain[nb:nsimu+1,:])
    
    z = (m1 - m2)/(np.sqrt(sa/na + sb/(nsimu - nb)))
    p = 2*(1-scipy.stats.norm.cdf(np.abs(z)))
    return z, p
        
        
def spectral_estimate_for_variance(x):
    '''
    Spectral density at frequency zero.

    Args:
        * **x** (:class:`~numpy.ndarray`): Array of points - portion of chain.
    
    Returns:
        * **s** (:class:`~numpy.ndarray`): Spectral estimate for variance.
    '''
    m,n = x.shape
    s = np.zeros([n,])
    for ii in range(n):
        y = power_spectral_density_using_hanning_window(x[:,ii],m)
        s[ii] = y[0]
        
    return s
            
def power_spectral_density_using_hanning_window(x, nfft = None, nw = None):
    '''
    Power spectral density using Hanning window.
    
    Args:
        * **x** (:class:`~numpy.ndarray`): Array of points - portion of chain.
        * **nfft** (:py:class:`int`): Length of Fourier transform.
        * **nw** (:py:class:`int`): Size of window.
    
    Returns:
        * **y** (:class:`~numpy.ndarray`): Power spectral density.
    '''
    if nfft is None:
        nfft = min(len(x),256)
        
    if nw is None:
        nw = int(np.fix(nfft/4))
    
    noverlap = int(np.fix(nw/2))
    
    # define hanning window
    w = 0.5*(1-np.cos(2*np.pi*np.arange(1,nw+1)/(nw+1)))
    
    n = len(x)
    if n < nw:
        x[nw+1] = 0
        n = nw
        
    # number of windows
    k = int(np.fix((n - noverlap)/(nw - noverlap)))
    index = np.arange(nw) # 0,1,...,nw-1
    kmu = k*np.linalg.norm(w)**2 # normalizing scale factor
    
    # initialize array
    y = np.zeros([nfft,1])
    for ii in range(k):
        xw = w*x[index]
        index = index + (nw - noverlap)
        Xx = np.abs(fft(xw,nfft)**2).reshape(nfft,1)
        y = y + Xx
    
    y = y*(1/kmu) # normalize
    n2 = int(np.floor(nfft/2))
    y = y[0:n2]
#    f = 1/n*np.arange(0,n2)
    return y
# ----------------------------------------------------
def integrated_autocorrelation_time(chain):
    '''
    Estimates the integrated autocorrelation time using Sokal's
    adaptive truncated periodogram estimator.

    Args:
        * **chain** (:class:`~numpy.ndarray`): Sampling chain.

    Returns:
        * **tau** (:class:`~numpy.ndarray`): Autocorrelation time.
        * **m** (:class:`~numpy.ndarray`): Counter.
    '''
    # get shape of chain
    npar = chain.shape[1]
    
    # initialize arrays
    tau = np.zeros([npar,])
    m = np.zeros([npar,])
    
    x = fft(chain,axis=0)
    xr = np.real(x)
    xi = np.imag(x)
    xmag = xr**2 + xi**2
    xmag[0,:] = 0.
    xmag = np.real(fft(xmag,axis=0))
    var = xmag[0,:]/len(chain)/(len(chain)-1)
    
    for jj in range(npar):
        if var[jj] == 0:
            continue
        xmag[:,jj] = xmag[:,jj]/xmag[0,jj]
        snum = -1/3
        for ii in range(len(chain)):
            snum = snum + xmag[ii,jj] - 1/6
            if snum < 0:
                tau[jj] = 2*(snum + (ii)/6)
                m[jj] = ii + 1
                break
            
    return tau, m
                
# ----------------------------------------------------
def get_parameter_names(nparam, results):
    '''
    Get parameter names from results dictionary.
    
    If no results found, then default names are generated.  If some results are
    found, then an extended set is generated to complete the list requirement.
    Uses the functions: :func:`~.plotting.utilities.generate_default_names` and
    :func:`~.plotting.utilities.extend_names_to_match_nparam`
    
    Args:
        * **nparam** (:py:class:`int`): Number of parameter names needed
    
    Returns:
        * **names** (:py:class:`list`): List of length `nparam` with strings.
    '''
    if results is None: # results not specified
        names = generate_default_names(nparam)
    else:
        names = results['names']
        names = extend_names_to_match_nparam(names, nparam)
            
    return names