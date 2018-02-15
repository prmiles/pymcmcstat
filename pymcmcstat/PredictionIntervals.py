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
from scipy.interpolate import interp1d
from pymcmcstat.DataStructure import DataStructure
import matplotlib.pyplot as plt

class PredictionIntervals:
    
    def setup_prediction_interval_calculation(self, results, data, modelfunction):
        
        # Analyze data structure
        dshapes = data.shape
        self.__ndatabatches = len(dshapes)
        nrows = []
        ncols = []
        for ii in xrange(self.__ndatabatches):
            nrows.append(dshapes[ii][0])
            if len(dshapes[0]) != 1:
                ncols.append(dshapes[ii][1])
            else:
                ncols.append(1)
                
        self.datapred = []
        for ii in xrange(self.__ndatabatches):
            # setup data structure for prediction
            # this is required to allow user to send objects other than xdata to model function
            self.datapred.append(DataStructure())
            self.datapred[ii].add_data_set(x = data.xdata[ii], y = data.ydata[ii], 
                                  user_defined_object = data.user_defined_object[ii])
            
        # assign model function
        self.modelfunction = modelfunction
        
        # assign required features from the results structure
        self.__chain = results['chain']
        self.__s2chain = results['s2chain']
        self.__parind = results['parind']
        self.__local = results['local']
        self.__nbatch = results['model'].nbatch
        self.__theta = results['theta']
        
        if 'sstype' in results:
            self.__sstype = results['sstype']
        else:
            self.__sstype = 0
        
        # evaluate model function to determine shape of response
        self.__nrow = []
        self.__ncol = []
        for ii in xrange(self.__ndatabatches):
            y = self.modelfunction(self.datapred[ii], self.__theta)
            sh = y.shape
            if len(sh) == 1:
                self.__nrow.append(sh[0])
                self.__ncol.append(1)
            else:
                self.__nrow.append(sh[0])
                self.__ncol.append(sh[1])
            
    def generate_prediction_intervals(self, sstype = None, nsample = 500, calc_pred_int = 'on'):
        
        # extract chain & s2chain from results
        chain = self.__chain
        if calc_pred_int is not 'on':
            s2chain = None
        else:
            s2chain = self.__s2chain
        
        # define number of simulations by the size of the chain array
        nsimu, npar = chain.shape
        
        # define interval limits
        if s2chain is None:
            lims = np.array([0.005,0.025,0.05,0.25,0.5,0.75,0.9,0.975,0.995])
        else:
            lims = np.array([0.025, 0.5, 0.975])
        
        if sstype is None:
            sstype = self.__sstype
        else:
            sstype = 0
        
        # check value of nsample
        if nsample is None:
            nsample = nsimu
            
        # define sample points
        if nsample >= nsimu:
            iisample = range(nsimu) # sample all points from chain
            nsample = nsimu
        else:
            # randomly sample from chain
            iisample = np.ceil(np.random.rand(nsample,1)*nsimu) - 1
            iisample = iisample.astype(int)
        
        # loop through data sets
        theta = self.__theta
        credible_intervals = []
        prediction_intervals = []
        for ii in xrange(len(self.datapred)):
            datapredii = self.datapred[ii]
            
            ysave = np.zeros([nsample, self.__nrow[ii], self.__ncol[ii]])
            osave = np.zeros([nsample, self.__nrow[ii], self.__ncol[ii]])
            
            for kk in xrange(nsample):
                theta[self.__parind[:]] = chain[iisample[kk],:]
                # some parameters may only apply to certain batch sets
                test1 = self.__local == 0
                test2 = self.__local == ii
                th = theta[test1 + test2]
                ypred = self.modelfunction(datapredii, th)
                ypred = ypred.reshape(self.__nrow[ii], self.__ncol[ii])

                if s2chain is not None:
                    s2elem = s2chain[iisample[kk],ii].reshape(1,1)
                    if sstype == 0:
                        opred = ypred + np.random.standard_normal(ypred.shape)*np.diag(
                            np.sqrt(s2elem))
                    elif sstype == 1: # sqrt
                        opred = (np.sqrt(ypred) + np.random.standard_normal(ypred.shape)*np.diag(
                            np.sqrt(s2elem)))**2
                    elif sstype == 2: # log
                        opred = ypred*np.exp(np.random.standard_normal(ypred.shape))*np.diag(
                            np.sqrt(s2elem))
                    else:
                        sys.exit('Unknown sstype')
                   
                # store model prediction
                ysave[kk,:,:] = ypred # store model output
                osave[kk,:,:] = opred # store model output with observation errors

            # generate quantiles
            ny = len(ysave)
            plim = []
            olim = []
            for jj in xrange(self.__ncol[ii]):
                if 0 and self.__nbatch == 1 and ny == 1:
                    plim.append(self.__empirical_quantiles(ysave[:,:,jj], lims))
                elif 0 and self.__nbatch == 1:
                    plim.append(self.__empirical_quantiles(ysave[:,:,jj], lims))
                else:
                    plim.append(self.__empirical_quantiles(ysave[:,:,jj], lims))
                
                if s2chain is not None:
                    olim.append(self.__empirical_quantiles(osave[:,:,jj], lims))
                    
            credible_intervals.append(plim)
            prediction_intervals.append(olim)
            
        if s2chain is None:
            prediction_intervals = None
            
        # generate output dictionary
        self.intervals = {'credible_intervals': credible_intervals, 
               'prediction_intervals': prediction_intervals}
        
    def __empirical_quantiles(self, x, p = np.array([0.25, 0.5, 0.75])):
        """
        function y=plims(x,p)
        %PLIMS Empirical quantiles
        % plims(x,p)  calculates p quantiles from columns of x
        % Marko Laine <Marko.Laine@Helsinki.FI>
        % $Revision: 1.4 $  $Date: 2007/05/21 11:19:12 $
        Adapted for Python by Paul Miles on 2017/11/08
        """
    
        # extract number of rows/cols from np.array
        n, m = x.shape 
    #    print('n = {}, m = {}'.format(n,m))
        # define vector valued interpolation function
        xpoints = range(n)
        interpfun = interp1d(xpoints, np.sort(x, 0), axis = 0)
        
        # evaluation points
        itpoints = (n-1)*p   
    #    print('xpoints = {}'.format(xpoints))
    #    print('itpoints = {}'.format(itpoints))
        
        return interpfun(itpoints)
    
    def plot_prediction_intervals(self, plot_pred_int = 'on', adddata = False):
        
        # unpack out dictionary
        credible_intervals = self.intervals['credible_intervals']
        prediction_intervals = self.intervals['prediction_intervals']
        
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
        for ii in range(self.__ndatabatches):
            
            credlims = credible_intervals[ii] # should be np lists inside
            ny = len(credlims)
            
            # extract data
            dataii = self.datapred[ii]
            
            # define independent data
            time = dataii.xdata[0].reshape(dataii.xdata[0].shape[0],)
            
            for jj in range(ny):
                fighand = str('data set {}'.format(jj))
                htmp = plt.figure(fighand, figsize=(7,5)) # create new figure
                fighandle.append(htmp)
                
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
                if adddata is True:
                    plt.plot(dataii.xdata[0], dataii.ydata[0][:,jj], '.b', linewidth=1,
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