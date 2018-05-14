#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 12:54:16 2018

@author: prmiles
"""

# import required packages
from __future__ import division
import math
import matplotlib.pyplot as pyplot
from pylab import hist 
from .utilities import generate_default_names, extend_names_to_match_nparam, make_x_grid

import warnings

try:
    from statsmodels.nonparametric.kernel_density import KDEMultivariate
except ImportError as e:
    warnings.warn("Exception raised importing statsmodels.nonparametric.kernel_density - plot_density_panel will not work.)", ImportWarning)

# --------------------------------------------
def plot_density_panel(chains, names = None, hist_on = False, figsizeinches = None):
    """
    Plot marginal posterior densities
    
    :param chains: Sampling chain for each parameter
    :type chains: :class:`~numpy.ndarray`
    :param names: Name of each parameter (if `None`, default names are generated)
    :type names: :py:class:`list` or `None`
    :param hist_on: Flag to include histogram on density plot
    :type hist_on: :py:class:`bool`
    :param figsizeinches: Specify figure size in inches [Width, Height]
    :type figsizeinches: :py:class:`list`
    """
    nrow, ncol = chains.shape # number of rows, number of columns
    
    nparam = ncol # number of parameter chains
    ns1 = math.ceil(math.sqrt(nparam))
    ns2 = round(math.sqrt(nparam))

    # Check if names defined
    if names == None:
        names = generate_default_names(nparam)
        
    # Check if enough names defined
    if len(names) != nparam:
        names = extend_names_to_match_nparam(names, nparam)
    
    if figsizeinches is None:
        figsizeinches = [5,4]
        
    pyplot.figure(dpi=100, figsize=(figsizeinches)) # initialize figure
    for ii in range(nparam):
        # define chain
        chain = chains[:,ii] # check indexing
        chain = chain.reshape(nrow,1)
        
        # define x grid
        chain_grid = make_x_grid(chain)        

        # Compuate kernel density estimate
        kde = KDEMultivariate(chain, bw = 'normal_reference', var_type = 'c')

        # plot density on subplot
        pyplot.subplot(ns1,ns2,ii+1)
             
        if hist_on == True: # include histograms
            hist(chain, normed=True)
            
        pyplot.plot(chain_grid, kde.pdf(chain_grid), 'k')
        # format figure
        pyplot.xlabel(names[ii])
        pyplot.ylabel(str('$\pi$({}$|M^{}$)'.format(names[ii], '{data}')))
        pyplot.tight_layout(rect=[0, 0.03, 1, 0.95],h_pad=1.0) # adjust spacing

# --------------------------------------------
def plot_histogram_panel(chains, names = None, figsizeinches = None):
    """
    Plot histogram from each parameter's sampling history
    
    :param chains: Sampling chain for each parameter
    :type chains: :class:`~numpy.ndarray`
    :param names: Name of each parameter (if `None`, default names are generated)
    :type names: :py:class:`list` or `None`
    :param figsizeinches: Specify figure size in inches [Width, Height]
    :type figsizeinches: :py:class:`list`
    """
    nrow, ncol = chains.shape # number of rows, number of columns
    
    nparam = ncol # number of parameter chains
    ns1 = math.ceil(math.sqrt(nparam))
    ns2 = round(math.sqrt(nparam))
    
    # Check if names defined
    if names == None:
        names = generate_default_names(nparam)
    
    # Check if enough names defined
    if len(names) != nparam:
        names = extend_names_to_match_nparam(names, nparam)
       
    if figsizeinches is None:
        figsizeinches = [5,4]
        
    f = pyplot.figure(dpi=100, figsize=(figsizeinches)) # initialize figure
    for ii in range(nparam):
        # define chain
        chain = chains[:,ii] # check indexing
        chain = chain.reshape(nrow,1) 
        
        # plot density on subplot
        ax = pyplot.subplot(ns1,ns2,ii+1)
        hist(chain, normed=True)
        # format figure
        pyplot.xlabel(names[ii])
        ax.set_yticklabels([])
        pyplot.tight_layout(rect=[0, 0.03, 1, 0.95],h_pad=1.0) # adjust spacing
        
    return f
        
# --------------------------------------------
def plot_chain_panel(chains, names = None, figsizeinches = None):
    """
    Plot sampling chain for each parameter
    
    :param chains: Sampling chain for each parameter
    :type chains: :class:`~numpy.ndarray`
    :param names: Name of each parameter (if `None`, default names are generated)
    :type names: :py:class:`list` or `None`
    :param figsizeinches: Specify figure size in inches [Width, Height]
    :type figsizeinches: :py:class:`list`
    """
    nsimu, nparam = chains.shape # number of rows, number of columns

    skip = 1
    maxpoints = 500 # max number of display points - keeps scatter plot from becoming overcrowded
    if nsimu > maxpoints:
        skip = int(math.floor(nsimu/maxpoints))
    
    ns1 = math.ceil(math.sqrt(nparam))
    ns2 = round(math.sqrt(nparam))
    
    # Check if names defined
    if names == None:
        names = generate_default_names(nparam)
    
    # Check if enough names defined
    if len(names) != nparam:
        names = extend_names_to_match_nparam(names, nparam)
        
    if figsizeinches is None:
        figsizeinches = [5,4]
        
    f = pyplot.figure(dpi=100, figsize=(figsizeinches)) # initialize figure
    for ii in range(nparam):
        # define chain
        chain = chains[:,ii] # check indexing
        chain = chain.reshape(nsimu,1)
        
        # plot density on subplot
        pyplot.subplot(ns1,ns2,ii+1)
        pyplot.plot(range(0,nsimu,skip), chain[range(0,nsimu,skip),0], '.b')
        # format figure
        pyplot.xlabel('Iteration')
        pyplot.ylabel(str('{}'.format(names[ii])))
        if ii+1 <= ns1*ns2 - ns2:
            pyplot.xlabel('')
            
        pyplot.tight_layout(rect=[0, 0.03, 1, 0.95],h_pad=1.0) # adjust spacing
        
    return f
        
# --------------------------------------------
def plot_pairwise_correlation_panel(chains, names = None, figsizeinches = None, skip=1):
    """
    Plot pairwise correlation for each parameter
    
    :param chains: Sampling chain for each parameter
    :type chains: :class:`~numpy.ndarray`
    :param names: Name of each parameter (if `None`, default names are generated)
    :type names: :py:class:`list` or `None`
    :param figsizeinches: Specify figure size in inches [Width, Height]
    :type figsizeinches: :py:class:`list`
    :param skip: Indicates step size to be used when plotting elements from the chain
    :type skip: :py:class:`int`
    """
    nsimu, nparam = chains.shape # number of rows, number of columns
    
    inds = range(0,nsimu,skip)
    
    # Check if names defined
    if names == None:
        names = generate_default_names(nparam)
        
    # Check if enough names defined
    if len(names) != nparam:
        names = extend_names_to_match_nparam(names, nparam)
        
    if figsizeinches is None:
        figsizeinches = [7,5]
        
    f = pyplot.figure(dpi=100, figsize=(figsizeinches)) # initialize figure
    for jj in range(2,nparam+1):
        for ii in range(1,jj):
            chain1 = chains[inds,ii-1]
            chain1 = chain1.reshape(nsimu,1)
            chain2 = chains[inds,jj-1]
            chain2 = chain2.reshape(nsimu,1)                    
            
            # plot density on subplot
            ax = pyplot.subplot(nparam-1,nparam-1,(jj-2)*(nparam-1)+ii)
            pyplot.plot(chain1, chain2, '.b')
            
            # format figure
            if jj != nparam: # rm xticks
                ax.set_xticklabels([])
            if ii != 1: # rm yticks
                ax.set_yticklabels([])
            if ii == 1: # add ylabels
                pyplot.ylabel(str('{}'.format(names[jj-1])))
            if ii == jj - 1: 
                if nparam == 2: # add xlabels
                    pyplot.xlabel(str('{}'.format(names[ii-1])))
                else: # add title
                    pyplot.title(str('{}'.format(names[ii-1])))
         
    # adjust figure margins
    pyplot.tight_layout(rect=[0, 0.03, 1, 0.95],h_pad=1.0) # adjust spacing
    
    return f
 
# --------------------------------------------
def plot_chain_metrics(chain, name):
    """
    Plot chain metrics for individual chain
    
    - Scatter plot of chain
    - Histogram of chain
    
    :param chain: Sampling chain for specifi parameter
    :type chain: :class:`~numpy.ndarray`
    :param name: Name of parameter
    :type name: :py:class:`str`
    """
    pyplot.figure(dpi=100) # initialize figure
    pyplot.suptitle('Chain metrics for {}'.format(name), fontsize='12')
    pyplot.subplot(2,1,1)
    pyplot.scatter(range(0,len(chain)),chain, marker='.')
    # format figure
    pyplot.xlabel('Iterations')
    ystr = str('{}-chain'.format(name))
    pyplot.ylabel(ystr)
    # Add histogram
    pyplot.subplot(2,1,2)
    hist(chain)
    # format figure
    pyplot.xlabel(name)
    pyplot.ylabel(str('Histogram of {}-chain'.format(name)))
    pyplot.tight_layout(rect=[0, 0.03, 1, 0.95],h_pad=1.0) # adjust spacing
    
class Plot:
    """
    Plotting routines for analyzing sampling chains from MCMC process.
    
    Methods:
        1. :func:`plot_density_panel`
        2. :func:`plot_chain_panel`
        3. :func:`plot_pairwise_correlation_panel`
        4. :func:`plot_histogram_panel`
        5. :func:`plot_chain_metrics`
    """
    def __init__(self):
        self.plot_density_panel = plot_density_panel
        self.plot_chain_panel = plot_chain_panel
        self.plot_pairwise_correlation_panel = plot_pairwise_correlation_panel
        self.plot_histogram_panel = plot_histogram_panel
        self.plot_chain_metrics = plot_chain_metrics