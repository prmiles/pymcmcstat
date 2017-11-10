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
import classes as mcclass

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
        iisample = np.ceil(np.random.rand(nsample,1)*nsimu) - 1
        iisample = iisample.astype(int)
    
    
    # loop through data sets
    credible_intervals = []
    prediction_intervals = []
    for ii in xrange(nbatch):
        ybatchsave = []
        obatchsave = []
        for jj in xrange(len(data[ii].n)):
            # setup data structure for prediction
            # this is required to allow user to send objects other than xdata to model function
            datapred = mcclass.DataStructure()
            datapred.add_data_set(x = data[ii].xdata[jj], y = data[ii].ydata[jj], udobj = data[ii].udobj[jj])
            
            ysave = np.zeros([nsample, data[ii].n[jj]])
            osave = np.zeros([nsample, data[ii].n[jj]])
#            print('ii = {}, jj = {}'.format(ii,jj))
            for kk in xrange(nsample):
#                print('iisample[{}] = {}'.format(kk,iisample[kk]))
                theta[parind[:]] = chain[iisample[kk],:]
                # some parameters may only apply to certain batch sets
                test1 = local == 0
                test2 = local == ii
                th = theta[test1 + test2]
                y = modelfun(datapred, th)
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
        print('ny = {}'.format(ny))
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
                
        credible_intervals.append(plim)
        prediction_intervals.append(olim)
        
    if s2chain is None:
        prediction_intervals = None
        
    # generate output dictionary
    out = {'credible_intervals': credible_intervals, 
           'prediction_intervals': prediction_intervals, 
           'data': data}
    
    return out