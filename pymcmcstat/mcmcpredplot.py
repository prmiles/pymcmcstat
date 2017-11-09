#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 17:27:12 2017
function h=mcmcpredplot(out,data,adddata)
%MCMCPREDPLOT - predictive plot for mcmc results
% Creates predictive figures for each batch in the data set using
% mcmc chain. Needs input from the function mcmcpred.
% Example:
%  out=mcmcpred(results,chain,s2chain,data,modelfun);
%  mcmcpredplot(out)
%
% If s2chain has been given to mcmcpred, then the plot shows 95%
% probability limits for new observations and for model parameter
% uncertainty. If s2chain is not used then the plot contains 50%,
% 90%, 95%, and 99% predictive probability limits due parameter uncertainty.

% $Revision: 1.4 $  $Date: 2007/08/22 16:10:58 $
Adapted for Python by Paul Miles 2017/11/08
"""

from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt

def mcmcpredplot(out, data = None, adddata = None):
    
    if data is None:
        data = out['data']
        
    if adddata is None:
        adddata = 0
                
    # unpack out dictionary
    credible_intervals = out['credible_intervals']
    prediction_intervals = out['prediction_intervals']
    
    # define number of batches
    nbatch = len(credible_intervals)
    
    # define counting metrics
    nlines = len(credible_intervals[0][0]) # number of lines
    nn = (nlines + 1)/2 # median
    nlines = nn - 1
    
    # initialize figure handle
    hh = []
    tmp = plt.figure()
    hh.append(tmp)
    for ii in range(nbatch):
        if ii > 0:
            tmp = plt.figure() # create new figure
            hh.append(tmp)
        
        credlims = credible_intervals[ii] # should be np lists inside
        ny = len(credlims)
        
        # extract data
        dataii = data[ii]
        
        time = dataii.xdata[0] # need to add functionality for multiple xdata
        
        for jj in range(ny):
            intcol = [0.9, 0.9, 0.9] # dimmest (lightest) color
            fig, ax = plt.subplots(ny,1)
            if ny == 1:
                ax = [ax]
                
            if prediction_intervals is not None:
                ax[jj].fill_between(time, prediction_intervals[ii][jj][0], 
                                prediction_intervals[ii][jj][-1], facecolor = intcol, alpha = 0.5)
                intcol = [0.8, 0.8, 0.8]
            
            ax[jj].fill_between(time, credlims[jj][0], credlims[jj][-1],
                              facecolor = intcol, alpha = 0.5)
            
            for kk in range(1,int(nn)-1):
                tmpintcol = np.array(intcol)*0.9**(kk)
                ax[jj].fill_between(time, credlims[jj][kk], credlims[jj][-kk - 1],
                              facecolor = tmpintcol, alpha = 0.5)
            # add model (median parameter values)
            ax[jj].plot(time, credlims[jj][nn], '-k', linewidth=2)
                
            if adddata:
                plt.plot(dataii.xdata[0], dataii.ydata[0], 'sk', linewidth=2)
            
            if nbatch > 1:
                plt.title(str('Data set {}, y[{}]'.format(ii,jj)))
            elif ny > 1:
                plt.title(str('y[{}]'.format(jj)))

    return hh