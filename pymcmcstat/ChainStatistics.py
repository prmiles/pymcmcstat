#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 10:23:51 2018

@author: prmiles
"""

# import required packages
import numpy as np
import sys
import scipy as scp
#from scipy.fftpack import fft
#from scipy.stats import norm

class ChainStatistics:
    
#    def __init__(self):
#        self.description = 'Chain Statistics'
        
    # display chain statistics
    def chainstats(self, chain = None, results = None):
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
            mcerr = bmstd/sqrt(nsimu)
                
            # calculate geweke's MCMC convergence diagnostic
            z,p = self.geweke(chain)
            
            
            # print statistics
            print('\n---------------------')
            print('{:10s}: {:>10s} {:>10s}'.format('name','mean','std'))
            for ii in range(npar):
                if meanii[ii] > 1e4:
                    print('{:10s}: {:10.4g} {:10.4g}'.format(names[ii],meanii[ii],stdii[ii]))
                else:
                    print('{:10s}: {:10.4f} {:10.4f}'.format(names[ii],meanii[ii],stdii[ii]))
                    
    def batch_mean_standard_deviation(self, chain, b = None):
        """
        %BMSTD standard deviation calculated from batch means
        % s = bmstd(x,b) - x matrix - b length of the batch
        % bmstd(x) gives an estimate of the Monte Carlo std of the 
        % estimates calculated from x
        
        Input:
            chain: [nsimu, npar]
            b: length of batch
        
        Adapted for Python by prmiles
        """
        
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
    
    def geweke(self, chain, a = 0.1, b = 0.5):
        """
        %GEWEKE Geweke's MCMC convergence diagnostic
        % [z,p] = geweke(chain,a,b)
        % Test for equality of the means of the first a% (default 10%) and
        % last b% (50%) of a Markov chain.
        % See:
        % Stephen P. Brooks and Gareth O. Roberts.
        % Assessing convergence of Markov chain Monte Carlo algorithms.
        % Statistics and Computing, 8:319--335, 1998.
        
        Input:
            chain: [nsimu, npar]
            a: first a% of chain
            b: last b% of chain
        """
        
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
        p = 2*(1-scp.stats.norm.cdf(np.abs(z)))
        return z, p
        
        
    def _spectral_estimate_for_variance(self, x):
        '''
        Spectral density at frequency zero
        
        Input:
            x: portion of chain
        '''
        m,n = x.shape
        s = np.zeros([1,n])
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
            Xx = np.abs(scp.fftpack.fft(xw,nfft)**2).reshape(nfft,1)
            y = y + Xx
        
        y = y*(1/kmu) # normalize
        n2 = int(np.floor(nfft/2))
        y = y[0:n2]
        f = 1/n*np.arange(0,n2)
        return y, f
    
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