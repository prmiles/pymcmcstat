# import required packages
from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as pyplot
#from sklearn.neighbors import KernelDensity
from pylab import hist 

def make_x_grid(x):
    xmin = min(x)
    xmax = max(x)
    xxrange = xmax-xmin
    if len(x) > 200:
        x_grid=np.linspace(xmin-0.08*xxrange,xmax+0.08*xxrange,100)
    else:
        x_grid=np.linspace(np.mean(x)-4*np.std(x, ddof=1),np.mean(x)+4*np.std(x, ddof=1),100)
    return x_grid.reshape(x_grid.shape[0],1) # returns 1d column vector

#if iqrange(x)<=0
#  s=1.06*std(x)*nx^(-1/5);
#else
#  s=1.06*min(std(x),iqrange(x)/1.34)*nx^(-1/5);
#end
#
    
"""see MASS 2nd ed page 181."""
def iqrange(x):
    nr, nc = x.shape
    if nr == 1: # make sure it is a column vector
        x = x.reshape(nc,nr)
        nr = nc
        nc = 1
    
    # sort
    x.sort()
    
    i1 = math.floor((nr + 1)/4)
    i3 = math.floor(3/4*(nr+1))
    f1 = (nr+1)/4-i1
    f3 = 3/4*(nr+1)-i3
    q1 = (1-f1)*x[int(i1),:] + f1*x[int(i1)+1,:]
    q3 = (1-f3)*x[int(i3),:] + f3*x[int(i3)+1,:]
    return q3-q1
    
def gaussian_density_function(x, mu, sigma2):
    y = 1/math.sqrt(2*math.pi*sigma2)*math.exp(-0.5*(x-mu)**2/sigma2)
    return y

def scale_bandwidth(x):
    n = len(x)
    if iqrange(x) <=0:
        s = 1.06*np.std(x, ddof=1)*n**(-1/5)
    else:
        s = 1.06*min(np.std(x, ddof=1),iqrange(x)/1.34)*n**(-1/5)
    return s
    
        
# plot density panel
# input:
# chains - 2d array - each column is chain of parameter (construct 2d array using numpy)
def plot_density_panel(chains, names, hist_on = 0):
    nrow, ncol = chains.shape # number of rows, number of columns
    
    nparam = ncol # number of parameter chains
    ns1 = math.ceil(math.sqrt(nparam))
    ns2 = round(math.sqrt(nparam))
    
    pyplot.figure(figsize=(5,4)) # initialize figure
    for ii in range(nparam):
        # define chain
        chain = chains[:,ii] # check indexing
        chain = chain.reshape(nrow,1)
        
        # define x grid
        chain_grid = make_x_grid(chain)        
        ngrid = len(chain_grid)
        
        ss = 2 # typically defined this way in matlab version
        s = scale_bandwidth(chain)
        
        if ss > 0:
            s = ss*s
        elif s<0:
            s = abs(ss)
        
        # calculate density
        chain_density = np.zeros((ngrid,1))
        for jj in range(ngrid):
#            chain_density[jj,0] = 1/nrow*sum(gaussian_density_function((chain_grid[jj,0]-chain)/s, 0, 1))/s
            gdf = np.vectorize(gaussian_density_function)
            chain_density[jj,0] = 1/(s*nrow)*sum(gdf((chain_grid[jj,0]-chain), 0, 1))
            
        
        # plot density on subplot
        pyplot.subplot(ns1,ns2,ii+1)
             
        if hist_on == 1: # include histograms
            hist(chain, normed=True)
            
        pyplot.plot(chain_grid, chain_density, 'k')
        # format figure
        pyplot.xlabel(names[ii])
        pyplot.ylabel(str('$\pi$({}$|M^{}$)'.format(names[ii], '{data}')))
        pyplot.tight_layout(rect=[0, 0.03, 1, 0.95],h_pad=1.0) # adjust spacing

# plot histogram panel
# input:
# chains - 2d array - each column is chain of parameter (construct 2d array using numpy)
def plot_histogram_panel(chains, names):
    nrow, ncol = chains.shape # number of rows, number of columns
    
    nparam = ncol # number of parameter chains
    ns1 = math.ceil(math.sqrt(nparam))
    ns2 = round(math.sqrt(nparam))
    
    f = pyplot.figure(dpi=100, figsize=(5,4)) # initialize figure
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
        
# plot chain panel
# input:
# chains - 2d array - each column is chain of parameter (construct 2d array using numpy)
def plot_chain_panel(chains, names):
    nsimu, nparam = chains.shape # number of rows, number of columns

    skip = 1
    maxpoints = 500 # max number of display points - keeps scatter plot from becoming overcrowded
    if nsimu > maxpoints:
        skip = int(math.floor(nsimu/maxpoints))
    
    ns1 = math.ceil(math.sqrt(nparam))
    ns2 = round(math.sqrt(nparam))
    
    f = pyplot.figure(dpi=100, figsize=(5,4)) # initialize figure
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
        pyplot.tight_layout(rect=[0, 0.03, 1, 0.95],h_pad=1.0) # adjust spacing
        
    return f
        
# plot pairwise correlation panel
# input:
# chains - 2d array - each column is chain of parameter (construct 2d array using numpy)
def plot_pairwise_correlation_panel(chains, names, skip=1):
    nsimu, nparam = chains.shape # number of rows, number of columns
    
    inds = range(0,nsimu,skip)
        
    f = pyplot.figure(dpi=100) # initialize figure
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
    
def plot_chain_metrics(chain, name):
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
    hist(samples_x)
    # format figure
    pyplot.xlabel(name)
    pyplot.ylabel(str('Histogram of {}-chain'.format(name)))
    pyplot.tight_layout(rect=[0, 0.03, 1, 0.95],h_pad=1.0) # adjust spacing