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
#import math
import numpy as np
import matplotlib.pyplot as plt

def mcmcpredplot(out, plot_pred_int = 'on', data = None, adddata = None):
    
    if data is None:
        data = out['data']
                
    # unpack out dictionary
    credible_intervals = out['credible_intervals']
    prediction_intervals = out['prediction_intervals']
    
    clabels = ['95% CI']
    plabels = ['95% PI']
    
    # check if prediction intervals exist and if user wants to plot them
    if plot_pred_int is not 'on' or prediction_intervals is None:
        prediction_intervals = None # turn off prediction intervals
        clabels = ['99% CI', '95% CI', '90% CI', '50% CI']
        
    # define number of batches
    nbatch = len(credible_intervals)
    
    # define counting metrics
    nlines = len(credible_intervals[0][0]) # number of lines
    nn = np.int((nlines + 1)/2) # median
    nlines = nn - 1
    
    if prediction_intervals is None and nlines == 1:
        clabels = ['95% CI']
    
    # initialize figure handle
    fighandle = []
    axhandle = []
#    tmp = plt.figure()
#    hh.append(tmp)
    for ii in range(nbatch):
        fighand = str('data set {}'.format(ii))
        htmp = plt.figure(fighand, figsize=(7,5)) # create new figure
        fighandle.append(htmp)
        
        credlims = credible_intervals[ii] # should be np lists inside
        ny = len(credlims)
        
        # extract data
        dataii = data[ii]
        
        for jj in range(ny):
            # define independent data
            time = dataii.xdata[jj]
            
            intcol = [0.9, 0.9, 0.9] # dimmest (lightest) color
            plt.figure(fighand)
            ax = plt.subplot(ny,1,jj+1)
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.5)
            axhandle.append(ax)
            
            # add prediction intervals - if applicable
            if prediction_intervals is not None:
                ax.fill_between(time, prediction_intervals[ii][jj][0], 
                                prediction_intervals[ii][jj][-1], 
                                facecolor = intcol, alpha = 0.5,
                                label = plabels[0])
                intcol = [0.8, 0.8, 0.8]
            
            # add first credible interval
            ax.fill_between(time, credlims[jj][0], credlims[jj][-1],
                              facecolor = intcol, alpha = 0.5, label = clabels[0])
            
            # add range of credible intervals - if applicable
            for kk in range(1,int(nn)-1):
                tmpintcol = np.array(intcol)*0.9**(kk)
                ax.fill_between(time, credlims[jj][kk], credlims[jj][-kk - 1],
                              facecolor = tmpintcol, alpha = 0.5,
                              label = clabels[kk])
                
            # add model (median parameter values)
            ax.plot(time, credlims[jj][nn], '-k', linewidth=2, label = 'model')
            
            # add data to plot
            if adddata is not None:
                plt.plot(dataii.xdata[jj], dataii.ydata[jj], '.b', linewidth=1,
                         label = 'data')
                
            # add title
            if nbatch > 1:
                plt.title(str('Data set {}, y[{}]'.format(ii,jj)))
            elif ny > 1:
                plt.title(str('y[{}]'.format(jj)))
                
            # add legend
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='upper left')

    return fighandle, axhandle