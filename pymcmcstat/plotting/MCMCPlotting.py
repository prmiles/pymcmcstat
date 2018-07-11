#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 12:54:16 2018

@author: prmiles
"""

# import required packages
from __future__ import division
import math
import matplotlib.pyplot as plt
from pylab import hist
from .utilities import generate_names, setup_plot_features, make_x_grid

import warnings

try:
    from statsmodels.nonparametric.kernel_density import KDEMultivariate
except ImportError as e:
    warnings.warn(str("Exception raised importing statsmodels.nonparametric.kernel_density - plot_density_panel will not work. {}".format(e)))

# --------------------------------------------
def plot_density_panel(chains, names = None, hist_on = False, figsizeinches = None):
    '''
    Plot marginal posterior densities

    Args:
        * **chains** (:class:`~numpy.ndarray`): Sampling chain for each parameter
        * **names** (:py:class:`list`): List of strings - name of each parameter
        * **hist_on** (:py:class:`bool`): Flag to include histogram on density plot
        * **figsizeinches** (:py:class:`list`): Specify figure size in inches [Width, Height]
    '''
    nsimu, nparam = chains.shape # number of rows, number of columns
    ns1, ns2, names, figsizeinches = setup_plot_features(nparam = nparam, names = names, figsizeinches = figsizeinches)

    f = plt.figure(dpi=100, figsize=(figsizeinches)) # initialize figure
    for ii in range(nparam):
        # define chain
        chain = chains[:,ii].reshape(nsimu,1) # check indexing
        
        # define x grid
        chain_grid = make_x_grid(chain)
        
        # Compute kernel density estimate
        kde = KDEMultivariate(chain, bw = 'normal_reference', var_type = 'c')

        # plot density on subplot
        plt.subplot(ns1,ns2,ii+1)
             
        if hist_on == True: # include histograms
            hist(chain, normed=True)
            
        plt.plot(chain_grid, kde.pdf(chain_grid), 'k')
        # format figure
        plt.xlabel(names[ii])
        plt.ylabel(str('$\pi$({}$|M^{}$)'.format(names[ii], '{data}')))
        plt.tight_layout(rect=[0, 0.03, 1, 0.95],h_pad=1.0) # adjust spacing

    return f

# --------------------------------------------
def plot_histogram_panel(chains, names = None, figsizeinches = None):
    """
    Plot histogram from each parameter's sampling history

    Args:
        * **chains** (:class:`~numpy.ndarray`): Sampling chain for each parameter
        * **names** (:py:class:`list`): List of strings - name of each parameter
        * **hist_on** (:py:class:`bool`): Flag to include histogram on density plot
        * **figsizeinches** (:py:class:`list`): Specify figure size in inches [Width, Height]
    """
    nsimu, nparam = chains.shape # number of rows, number of columns
    ns1, ns2, names, figsizeinches = setup_plot_features(nparam = nparam, names = names, figsizeinches = figsizeinches)
        
    f = plt.figure(dpi=100, figsize=(figsizeinches)) # initialize figure
    for ii in range(nparam):
        # define chain
        chain = chains[:,ii].reshape(nsimu,1) # check indexing
        
        # plot density on subplot
        ax = plt.subplot(ns1,ns2,ii+1)
        hist(chain, normed=True)
        # format figure
        plt.xlabel(names[ii])
        ax.set_yticklabels([])
        plt.tight_layout(rect=[0, 0.03, 1, 0.95],h_pad=1.0) # adjust spacing
        
    return f
        
# --------------------------------------------
def plot_chain_panel(chains, names = None, figsizeinches = None, maxpoints = 500):
    """
    Plot sampling chain for each parameter

    Args:
        * **chains** (:class:`~numpy.ndarray`): Sampling chain for each parameter
        * **names** (:py:class:`list`): List of strings - name of each parameter
        * **figsizeinches** (:py:class:`list`): Specify figure size in inches [Width, Height]
        * **maxpoints** (:py:class:`int`): Max number of display points - keeps scatter plot from becoming overcrowded
    """
    nsimu, nparam = chains.shape # number of rows, number of columns
    ns1, ns2, names, figsizeinches = setup_plot_features(nparam = nparam, names = names, figsizeinches = figsizeinches)
    
    skip = 1
    if nsimu > maxpoints:
        skip = int(math.floor(nsimu/maxpoints))
    
    f = plt.figure(dpi=100, figsize=(figsizeinches)) # initialize figure
    for ii in range(nparam):
        # define chain
        chain = chains[:,ii].reshape(nsimu,1) # check indexing
        
        # plot chain on subplot
        plt.subplot(ns1,ns2,ii+1)
        plt.plot(range(0,nsimu,skip), chain[range(0,nsimu,skip),0], '.b')
        # format figure
        plt.xlabel('Iteration')
        plt.ylabel(str('{}'.format(names[ii])))
        if ii+1 <= ns1*ns2 - ns2:
            plt.xlabel('')
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95],h_pad=1.0) # adjust spacing
        
    return f
        
# --------------------------------------------
def plot_pairwise_correlation_panel(chains, names = None, figsizeinches = None, skip = 1):
    """
    Plot pairwise correlation for each parameter

    Args:
        * **chains** (:class:`~numpy.ndarray`): Sampling chain for each parameter
        * **names** (:py:class:`list`): List of strings - name of each parameter
        * **figsizeinches** (:py:class:`list`): Specify figure size in inches [Width, Height]
        * **skip** (:py:class:`int`): Indicates step size to be used when plotting elements from the chain
    """
    nsimu, nparam = chains.shape # number of rows, number of columns
    
    inds = range(0,nsimu,skip)
    
    names = generate_names(nparam = nparam, names = names)
        
    if figsizeinches is None:
        figsizeinches = [7,5]
        
    f = plt.figure(dpi=100, figsize=(figsizeinches)) # initialize figure
    for jj in range(2,nparam+1):
        for ii in range(1,jj):
            chain1 = chains[inds,ii-1]
            chain1 = chain1.reshape(nsimu,1)
            chain2 = chains[inds,jj-1]
            chain2 = chain2.reshape(nsimu,1)
            
            # plot density on subplot
            ax = plt.subplot(nparam-1,nparam-1,(jj-2)*(nparam-1)+ii)
            plt.plot(chain1, chain2, '.b')
            
            # format figure
            if jj != nparam: # rm xticks
                ax.set_xticklabels([])
            if ii != 1: # rm yticks
                ax.set_yticklabels([])
            if ii == 1: # add ylabels
                plt.ylabel(str('{}'.format(names[jj-1])))
            if ii == jj - 1:
                if nparam == 2: # add xlabels
                    plt.xlabel(str('{}'.format(names[ii-1])))
                else: # add title
                    plt.title(str('{}'.format(names[ii-1])))
         
    # adjust figure margins
    plt.tight_layout(rect=[0, 0.03, 1, 0.95],h_pad=1.0) # adjust spacing
    
    return f
 
# --------------------------------------------
def plot_chain_metrics(chain, name = None, figsizeinches = None):
    '''
    Plot chain metrics for individual chain

    - Scatter plot of chain
    - Histogram of chain
    
    Args:
        * **chains** (:class:`~numpy.ndarray`): Sampling chain for specific parameter
        * **names** (:py:class:`str`): Name of each parameter
        * **figsizeinches** (:py:class:`list`): Specify figure size in inches [Width, Height]
    '''
    name = generate_names(nparam = 1, names = name)
    
    if figsizeinches is None:
        figsizeinches = [7,5]
        
    f = plt.figure(dpi=100, figsize=(figsizeinches)) # initialize figure
    plt.suptitle('Chain metrics for {}'.format(name), fontsize='12')
    plt.subplot(2,1,1)
    plt.plot(range(0,len(chain)),chain, marker='.')
    # format figure
    plt.xlabel('Iterations')
    ystr = str('{}-chain'.format(name))
    plt.ylabel(ystr)
    # Add histogram
    plt.subplot(2,1,2)
    hist(chain)
    # format figure
    plt.xlabel(name)
    plt.ylabel(str('Histogram of {}-chain'.format(name)))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95],h_pad=1.0) # adjust spacing
    return f
    
class Plot:
    '''
    Plotting routines for analyzing sampling chains from MCMC process.

    Attributes:
        - :meth:`~plot_density_panel`
        - :meth:`~plot_chain_panel`
        - :meth:`~plot_pairwise_correlation_panel`
        - :meth:`~plot_histogram_panel`
        - :meth:`~plot_chain_metrics`
    '''
    def __init__(self):
        self.plot_density_panel = plot_density_panel
        self.plot_chain_panel = plot_chain_panel
        self.plot_pairwise_correlation_panel = plot_pairwise_correlation_panel
        self.plot_histogram_panel = plot_histogram_panel
        self.plot_chain_metrics = plot_chain_metrics