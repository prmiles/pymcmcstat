#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 10:23:51 2018

@author: prmiles
"""

# import required packages
import numpy as np
import sys
import scipy
from scipy.fftpack import fft
#from scipy.stats import norm

class ChainStatistics:
    '''
    Methods for calculating chain statistics.
    
    :Attributes:
        * :meth:`~chainstats`
        * :meth:`~print_chain_statistics`
        * :meth:`~batch_mean_standard_deviation`
        * :meth:`~geweke`
        * :meth:`~integrated_autocorrelation_time`    
    '''
    
#    def __init__(self):
#        self.description = 'Chain Statistics'
        
    # display chain statistics
    def chainstats(self, chain = None, results = None, returnstats = False):
        '''
        Calculate chain statistics.
        
        :Args:
            * **chain** (:class:`~numpy.ndarray`): Sampling chain.
            * **results** (:py:class:`dict`): Results from MCMC simulation.
            * **returnstats** (:py:class:`bool`): Flag to return statistics.
            
        :Returns:
            * **stats** (:py:class:`dict`): Statistical measures of chain convergence.
            
        '''
        # 
        if chain is None:
            print('No chain reported - run simulation first.')
            pass
        else:
            nsimu, npar = chain.shape
            names = self.__get_parameter_names(npar, results)
            
            # calculate mean and standard deviation of entire chain
            meanii = []
            stdii = []
            for ii in range(npar):
                meanii.append(np.mean(chain[:,ii]))
                stdii.append(np.std(chain[:,ii]))
                
            # calculate batch mean standard deviation
            bmstd = self.batch_mean_standard_deviation(chain)
            mcerr = bmstd/np.sqrt(nsimu)
                
            # calculate geweke's MCMC convergence diagnostic
            z,p = self.geweke(chain)
            
            # estimate integrated autocorrelation time
            tau, m = self.integrated_autocorrelation_time(chain)
            
            # print statistics
            self.print_chain_statistics(names, meanii, stdii, mcerr, tau, p)
            
            # assign stats to dictionary
            stats = {'mean': list(meanii),
                    'std': list(stdii),
                    'bmstd': list(bmstd),
                    'geweke': list(p),
                    'tau': list(tau)}
            
            if returnstats is True:
                return stats
          
    def print_chain_statistics(self, names, meanii, stdii, mcerr, tau, p):
        '''
        Print chain statistics to terminal window.
        
        :Args:
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
    def batch_mean_standard_deviation(self, chain, b = None):
        '''
        Standard deviation calculated from batch means
        
        :Args:
            * **chain** (:class:`~numpy.ndarray`): Sampling chain.
            * **b** (:py:class:`int`): Step size.
            
        \\
        
        :Returns:
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
        s = np.sqrt(sum((y - np.matlib.repmat(np.mean(chain,0),nb,1))**2)/(nb-1)*b)
        
        return s
    # ----------------------------------------------------
    def geweke(self, chain, a = 0.1, b = 0.5):
        '''
        Geweke's MCMC convergence diagnostic
        
        Test for equality of the means of the first a% (default 10%) and
        last b% (50%) of a Markov chain - see :cite:`brooks1998assessing`.
        
        :Args:
            * **chain** (:class:`~numpy.ndarray`): Sampling chain.
            * **a** (:py:class:`float`): First a% of chain.
            * **b** (:py:class:`float`): Last b% of chain.
            
        \\
        
        :Returns:
            * **z** (:class:`~numpy.ndarray`): Convergence diagnostic prior to CDF.
            * **p** (:class:`~numpy.ndarray`): Geweke's MCMC convergence diagnostic.

        .. note::
            
            The percentage of the chain should be given as a decimal between zero and one.
            So, for the first 10% of the chain, define :code:`a = 0.1`.  Likewise,
            for the last 50% of the chain, define :code:`b = 0.5`.
            
        '''
        
        nsimu, npar = chain.shape
        
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
        sa = self._spectral_estimate_for_variance(chain[0:na,:])
        sb = self._spectral_estimate_for_variance(chain[nb:nsimu+1,:])
        
        z = (m1 - m2)/(np.sqrt(sa/na + sb/(nsimu - nb)))
        p = 2*(1-scipy.stats.norm.cdf(np.abs(z)))
        return z, p
        
        
    def _spectral_estimate_for_variance(self, x):
        '''
        Spectral density at frequency zero.
        
        :Args:
            x: portion of chain
        '''
        m,n = x.shape
        s = np.zeros([n,])
        for ii in range(n):
            y,f = self.__power_spectral_density_using_hanning_window(x[:,ii],m)
            s[ii] = y[0]
            
        return s
            
    def __power_spectral_density_using_hanning_window(self, x, nfft = None, nw = None):
        '''
        Power spectral density using Hanning window
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
        f = 1/n*np.arange(0,n2)
        return y, f
    # ----------------------------------------------------
    def integrated_autocorrelation_time(self, chain):
        '''
        Estimates the integrated autocorrelation time using Sokal's
        adaptive truncated periodogram estimator.
        
        :Args:
            * **chain** (:class:`~numpy.ndarray`): Sampling chain.
            
        \\
        
        :Returns:
            * **tau** (:class:`~numpy.ndarray`): Autocorrelation time.
            * **m** (:class:`~numpy.ndarray`): Counter.
        '''
        # get shape of chain
        nsimu, npar = chain.shape
        
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
    def __get_parameter_names(self, n, results):
        if results is None: # results not specified
            names = self.__generate_default_names(n)
        else:
            names = results['names']
            names = self.__extend_names_to_match_nparam(names, n)
                
        return names
    
    def __generate_default_names(self, nparam):
        # generate generic parameter name set
        names = []
        for ii in range(nparam):
            names.append(str('$p_{{{}}}$'.format(ii)))
        return names
    
    def __extend_names_to_match_nparam(self, names, nparam):
        # generate generic parameter name set
        n0 = len(names)
        for ii in range(n0,nparam):
            names.append(str('$p_{{{}}}$'.format(ii)))
        return names