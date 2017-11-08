#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 12:00:11 2017

function out=mcmcpred(results,chain,s2chain,data,modelfun,nsample,varargin)
%MCMCPRED predictive calculations from the mcmcrun chain
% out = mcmcpred(results,chain,s2chain,data,modelfun,nsample,varargin)
% Calls modelfun(data,theta,varargin{:})
% or modelfun(data{ibatch},theta(local),varargin{:}) in case the
% data has batches. 
% It samples theta from the chain and optionally sigma2 from s2chain.
% If s2chain is not empty, it calculates predictive limits for
% new observations assuming Gaussian error model.
% The output contains information that can be given to mcmcpredplot.

% $Revision: 1.4 $  $Date: 2007/09/11 11:55:59 $

Adapted for Python by Paul miles
@author: prmiles
"""

import numpy as np
import sys
import generalfunctions as genfun

def mcmcpred(results, data, modelfun, sstype = None, nsample = None):
    
    # extract chain & s2chain from results
    chain = results['chain']
    s2chain = results['s2chain']
    
    # define number of simulations by the size of the chain array
    nsimu, npar = chain.shape
    
    # unpack other required components
    parind = results['parind']
    local = results['local']
    nbatch = results['model'].nbatch
    theta = results['theta']
    
    # define interval limits
    if s2chain is None:
        lims = np.array([0.005,0.025,0.05,0.25,0.5,0.75,0.9,0.975,0.995])
    else:
        lims = np.array([0.025, 0.5, 0.975])
    
    if sstype is None:
        if 'sstype' in results:
            sstype = results['sstype']
        else:
            sstype = 0
    else:
        sstype = 0
    
    # check value of nsample
    if nsample is None:
        nsample = nsimu
        
    # define sample points
    if nsample >= nsimu:
        iisample = range(nsimu) # sample all points from chain
    else:
        # randomly sample from chain
        iisample = np.ceil(np.random.rand(nsample,1)*nsimu)
        iisample = iisample.astype(int)
    
    
    # loop through data sets
    ybatchsave = []
    obatchsave = []
    for ii in xrange(nbatch):
        for jj in xrange(len(data[ii].n)):
            dataii = data[ii].xdata[jj]
            ysave = np.zeros([nsample, data[ii].n[jj]])
            osave = np.zeros([nsample, data[ii].n[jj]])
            for kk in xrange(nsample):
#                print('iisample[{}] = {}'.format(kk,iisample[kk]))
                theta[parind[:]] = chain[iisample[kk],:]
                # some parameters may only apply to certain batch sets
                th = theta[local == 0]
                th = th[local == ii] 
                y = modelfun(dataii, th)
                ysave[kk,:] = y # store model output
            
                if s2chain is not None:
                    if sstype == 0:
                        osave[kk,:] = y + np.random.randn(y.size)*np.diag(
                            np.sqrt(s2chain[iisample[kk],:]))
                    elif sstype == 1: # sqrt
                        osave[kk,:] = (np.sqrt(y) + np.random.randn(y.size)*np.diag(
                            np.sqrt(s2chain[iisample[kk],:])))**2
                    elif sstype == 2: # log
                        osave[kk,:] = y*np.exp(np.random.randn(y.size))*np.diag(
                            np.sqrt(s2chain[iisample[kk],:]))
                    else:
                        sys.exit('Unknown sstype')
                        
            ybatchsave.append(ysave)
            obatchsave.append(osave)


    # generate quantiles
    ny = len(ybatchsave)
    plim = []
    olim = []
    for jj in range(ny):
        if 0 and nbatch == 1 and ny == 1:
            plim.append(genfun.empirical_quantiles(ybatchsave[jj], lims))
        elif 0 and nbatch == 1:
            plim.append(genfun.empirical_quantiles(ybatchsave[jj], lims))
        else:
            plim.append(genfun.empirical_quantiles(ybatchsave[jj], lims))
            
        if s2chain is not None:
            olim.append(genfun.empirical_quantiles(obatchsave[jj], lims))
        
    # generate output dictionary
    out = {'predlims': plim, 'obslims': olim, 'data': data}
    
    return out