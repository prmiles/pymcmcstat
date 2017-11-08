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

def mcmcpredplot(out, data = None, adddata = None):
    
    if data is None:
        data = out['data']
        
    if adddata is None:
        adddata = 0

