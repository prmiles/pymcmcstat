#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 14:48:24 2017

@author: prmiles
"""

import math
import numpy as np
import mcmcfunctions as mcfun
import classes as mcclass
import sys

def alphafun(trypath, A_count, invR):
    A_count = A_count + 1
    # mcfun.alphafun(trypath)
    stage = len(trypath) - 1 # The stage we're in, elements in trypath - 1
#    print('stage = {}'.format(stage))
    # recursively compute past alphas
    a1 = 1 # initialize
    a2 = 1 # initialize
    for k in range(0,stage-1):
#        print('k = {}'.format(k))
#        print('0:(k+1) = {}'.format(range(0,k+2)))
        tmp1, A_count = mcfun.alphafun(trypath[0:(k+2)], A_count, invR)
#        print('tmp1 = {}'.format(tmp1))
        a1 = a1*(1 - tmp1)
#        print('a1 = {}'.format(a1))
#        print('stage:stage-k-2:-1 = {}'.format(range(stage, stage-k-2, -1)))
        tmp2, A_count = mcfun.alphafun(trypath[stage:stage-k-2:-1], A_count, invR)
#        print('tmp2 = {}'.format(tmp2))
        a2 = a2*(1 - tmp2)
#        print('a2 = {}'.format(a2))
        if a2 == 0: # we will come back with prob 1
            alpha = np.zeros(1)
            return alpha, A_count
        
    y = mcfun.logposteriorratio(trypath[0], trypath[-1])
    
    for k in range(stage):
        y = y + mcfun.qfun(k, trypath, invR)
        
    alpha = min(np.ones(1), np.exp(y)*a2*(a1**(-1)))
#    print('A_count = {}, alpha = {}, y = {}, a1 = {}, a2 = {}'.format(A_count, alpha, y, a1, a2))
    
    return alpha, A_count
    
def qfun(iq, trypath, invR):
    # Gaussian nth stage log proposal ratio
    # log of q_i(y_n,...,y_{n-j})/q_i(x,y_1,...,y_j)
        
#    print('IN QFUN')
#    print('iq = {}'.format(iq))
    
    stage = len(trypath) - 1 - 1 # - 1, iq; 
#    print('stage = {}'.format(stage))
    if stage == iq: # shift index due to 0-indexing
        zq = np.zeros(1) # we are symmetric
    else:
        iR = invR[iq] # proposal^(-1/2)
#        print('iR = {}'.format(iR))
        y1 = trypath[0].theta
#        print('iq + 1 = {}'.format(iq + 1))
        y2 = trypath[iq + 1].theta # check index
#        print('stage + 1 = {}'.format(stage + 1))
        y3 = trypath[stage + 1].theta
#        print('stage - iq = {}'.format(stage - iq))
        y4 = trypath[stage - iq].theta
        
#        print('terma = {}'.format((np.linalg.norm((y4-y3)*iR))**2))
#        print('termb = {}'.format((np.linalg.norm((y2-y1)*iR))**2))
        
#        print('y4 - y3 = {}'.format(y4 - y3))
#        print('y2 - y1 = {}'.format(y2 - y1))
#        print('(y4 - y3)*iR = {}'.format((y4-y3)*iR))
#        print('(y4 - y3)*iR = {}'.format(np.dot(y4-y3,iR)))
        zq = -0.5*((np.linalg.norm(np.dot(y4-y3, iR)))**2 - (np.linalg.norm(np.dot(y2-y1, iR)))**2)
        
    return zq 
    
def logposteriorratio(x1, x2):
    # log posterior ratio, log(pi(x2)/pi(x1)*p(x2)/p(x1))
#    print('x1.ss = {}, x1.sigma2 = {}, x1.prior = {}'.format(x1.ss, x1.sigma2, x1.prior))
#    print('x2.ss = {}, x2.sigma2 = {}, x2.prior = {}'.format(x2.ss, x2.sigma2, x2.prior))
    
    
    zq = -0.5*(sum((x2.ss*(x2.sigma2**(-1)) - x1.ss*(x1.sigma2**(-1)))) + x2.prior - x1.prior)
    
    return zq
    
    
def gammar(m,n,a,b = 1):
    #%GAMMAR random deviates from gamma distribution
    #%  GAMMAR(M,N,A,B) returns a M*N matrix of random deviates from the Gamma
    #%  distribution with shape parameter A and scale parameter B:
    #%
    #%  p(x|A,B) = B^-A/gamma(A)*x^(A-1)*exp(-x/B)
    #
    #% Marko Laine <Marko.Laine@Helsinki.FI>
    #% Written for python by PRM
    
    if a <= 0: # special case
        y = np.zeros([m,n])
        return y
    
    y = mcfun.gammar_mt(m, n, a, b)
    return y

def gammar_mt(m, n, a, b = 1):
    #%GAMMAR_MT random deviates from gamma distribution
    #% 
    #%  GAMMAR_MT(M,N,A,B) returns a M*N matrix of random deviates from the Gamma
    #%  distribution with shape parameter A and scale parameter B:
    #%
    #%  p(x|A,B) = B^-A/gamma(A)*x^(A-1)*exp(-x/B)
    #%
    #%  Uses method of Marsaglia and Tsang (2000)
    #
    #% G. Marsaglia and W. W. Tsang:
    #% A Simple Method for Generating Gamma Variables,
    #% ACM Transactions on Mathematical Software, Vol. 26, No. 3,
    #% September 2000, 363-372.
    # Written for python by PRM
    import numpy as np
    y = np.zeros([m,n])
    for jj in range(0,n):
        for ii in range(0,m):
            y[ii,jj] = mcfun.gammar_mt1(a,b)
            
    return y
    
def gammar_mt1(a,b):
    if a < 1:
        y = mcfun.gammar_mt1(1+a,b)*np.random.rand(1)**(a**(-1))
        return y
    else:
        d = a - 3**(-1)
        c = (9*d)**(-0.5)
        while 1:
            while 1:
                x = np.random.randn(1)
                v = 1 + c*x
                if v > 0:
                    break
                
            v = v**(3)
            u = np.random.rand(1)
            if u < 1-0.0331*x**(4):
                break
            if np.log(u) < 0.5*x**2 + d*(1-v+np.log(v)):
                break
        
        y = b*d*v
        return y
        
    
def covupd(x, w, oldcov, oldmean, oldwsum, oldR = None):  
    #function [xcov,xmean,wsum,R]=covupd(x,w,oldcov,oldmean,oldwsum,oldR)
    #%COVUPD covariance update
    #% [xcov,xmean,wsum]=covupd(x,w,oldcov,oldmean,oldwsum)
    #
    #% optionally updates also the Cholesky factor R
    #
    #% Marko Laine <Marko.Laine@Helsinki.FI>
    #% $Revision: 1.3 $  $Date: 2006/09/06 09:15:16 $
    # Written for Python by PRM
    n, p = x.shape
#    print('n = {}, p = {}'.format(n, p))
    
    if n == 0: # nothing to update with
        return oldcov, oldmean, oldwsum
    
    if not w:
        w = np.ones(1)
        
    if len(w) == 1:
        w = np.ones(n)*w
        
    if oldR is None:
        R = None
    else:
        R = oldR
           
    if oldcov is None:
#        print('Input covariance is empty...')
        wsum = sum(w)
        xmean = np.zeros(p)
        xcov = np.zeros([p,p])
#        print('wsum = {}'.format(wsum))
#        print('xmean = {}'.format(xmean))
#        print('xcov = {}'.format(xcov))
#        print('range(p) = {}'.format(range(p)))
        for ii in range(p):
            xmean[ii] = sum(x[:,ii]*w)*(wsum**(-1))
#            print('xmean[{}] = {}'.format(ii,xmean[ii]))
        if wsum > 1:
            for ii in range(0,p):
                for jj in range(0,ii+1):
#                    print('ii = {}, jj = {}'.format(ii, jj))
#                    print('x[:,{}] = {}'.format(ii,x[:,ii]))
#                    print('xmean[{}] = {}'.format(ii,xmean[ii]))
#                    print('x[:,{}] = {}'.format(jj,x[:,jj]))
#                    print('xmean[{}] = {}'.format(jj,xmean[jj]))
#                    print('wsum = {}'.format(wsum))
#                    print('w = {}'.format(w))
                    term1 = x[:,ii] - xmean[ii]
                    term2 = x[:,jj] - xmean[jj]
#                    print('{}, {}'.format(len(term1), len(term2)))
#                    print('x[:,{}]-xmean[{}] = {}'.format(ii, ii, term1.transpose()))
                    xcov[ii,jj] = np.dot(term1.transpose(), ((term2)*w)*((wsum-1)**(-1)))
                    if ii != jj:
                        xcov[jj,ii] = xcov[ii,jj]
                        
    else:
#        print('Covariance is not empty...')
#        prtcount = 0
#        print('n = {}'.format(n))
        for ii in range(0,n):
#            print('-----\nii = {}\n'.format(ii))
            xi = x[ii,:]
            wsum = w[ii]
            xmean = oldmean + wsum*((wsum+oldwsum)**(-1))*(xi - oldmean)
            
#            prtcount += 1
#            if prtcount == 100:
##                print('x[0:9,:] = {}'.format(x[0:9,:]))
#                print('x[{},:] = {}'.format(ii, x[ii,:]))
#                print('wsum = {}'.format(wsum))
#                print('oldwsum = {}'.format(oldwsum))
#                print('oldcov = \n{}'.format(oldcov))
#                print('ii = {}, xmean = {}'.format(ii, xmean))
##                print('xi - oldmean = {}'.format(xi - oldmean))
#                prtcount = 0
                
#            print('R = {}\n'.format(R))
            if R is not None:
                print('R = \n{}\n'.format(R))
                print('np.sqrt((oldwsum-1)*((wsum+oldwsum-1)**(-1))) = {}\n'.format(np.sqrt((oldwsum-1)*((wsum+oldwsum-1)**(-1)))))
            
                R = cholupdate(np.sqrt((oldwsum-1)*((wsum+oldwsum-1)**(-1)))*R, 
                               np.dot((xi - oldmean).transpose(), 
                                      np.sqrt(((wsum*oldwsum)
                                      *((wsum+oldwsum-1)**(-1))
                                      *((wsum+oldwsum)**(-1))))))
        
            
#            print('np.dot((xi - oldmean).transpose(), (xi - oldmean)) = {}'.format(np.dot((xi-oldmean).reshape(1,1),(xi-oldmean).reshape(1,p))))
            
#            print('oldwsum = {}, wsum = {}'.format(oldwsum, wsum))
#            print('oldcov = {}'.format(oldcov))
#            print('xi - oldmean = {}'.format(xi - oldmean))
            xcov = (((oldwsum-1)*((wsum + oldwsum - 1)**(-1)))*oldcov 
                    + (wsum*oldwsum*((wsum+oldwsum-1)**(-1)))*((wsum 
                           + oldwsum)**(-1))*(np.dot((xi-oldmean).reshape(p,1),(xi-oldmean).reshape(1,p))))
        
#            print('xcov = \n{}\n'.format(xcov))
            wsum = wsum + oldwsum
            oldcov = xcov
            oldmean = xmean
            oldwsum = wsum
   
    return xcov, xmean, wsum

# Cholesky Update
def cholupdate(R, x):
    n = len(x)
    R1 = R.copy()
    x1 = x.copy()
    for ii in range(n):
        r = math.sqrt(R1[ii,ii]**2 + x1[ii]**2)
        c = r*(R1[ii,ii]**(-1))
        s = x1[ii]*(R1[ii,ii]**(-1))
        R1[ii,ii] = r
        if ii+1 < n:
            R1[ii,ii+1:n] = (R1[ii,ii+1:n] + s*x1[ii+1:n])*(c**(-1))
            x1[ii+1:n] = c*x1[ii+1:n] - s*R1[ii,ii+1:n]

    return R1

# display chain statistics
def chainstats(chain, results = []):
    # 
    m,n = chain.shape
    
    if results == []: # results is dictionary
        names = []
        for ii in range(n):
            names.append(str('P{}'.format(ii)))
    else:
        names = results['names']
    
    meanii = []
    stdii = []
    for ii in range(n):
        meanii.append(np.mean(chain[:,ii]))
        stdii.append(np.std(chain[:,ii]))
        
    print('\n---------------------')
    print('{:10s}: {:>10s} {:>10s}'.format('name','mean','std'))
    for ii in range(n):
        if meanii[ii] > 1e4:
            print('{:10s}: {:10.4g} {:10.4g}'.format(names[ii],meanii[ii],stdii[ii]))
        else:
            print('{:10s}: {:10.4f} {:10.4f}'.format(names[ii],meanii[ii],stdii[ii]))
    
def batch_mean_standard_deviation(x, b = None):
    m, n = x.shape
    
    if b is None:
        b = max(10, np.fix(m/20))
        
    inds = range(0, m+1, b)
    nb = len(inds) - 1
    if nb < 2:
        sys.error('too few batches')
    
    y = np.zeros(np,n)
    
    for ii in range(nb):
        y[ii,:] = np.mean(x[inds[ii]:inds[ii+1]-1,:])
        
    # calculate the estimated std of MC estimate
#    s = np.sqrt(sum((y - ...)))

def setup_no_adapt_index(noadaptind, parind):
    # define noadaptind as a boolean - inputted as list of index values not updated
    no_adapt_index = []
    if len(noadaptind) == 0:
        no_adapt_index = np.zeros([len(parind)],dtype=bool)
    else:
        for jj in range(len(noadaptind)):
            for ii in range(len(parind)):
                if noadaptind[jj] == parind[ii]:
                    no_adapt_index[jj] = np.ones([1],dtype=bool)
                else:
                    no_adapt_index[jj] = np.zeros([1],dtype=bool)
    return no_adapt_index

def setup_covariance_matrix(qcov, thetasig, value):
    # check qcov
    if qcov is None: # i.e., qcov is None (not defined)
        qcov = thetasig**2 # variance
        ii1 = np.isinf(qcov)
        ii2 = np.isnan(qcov)
        ii = ii1 + ii2
        qcov[ii] = (np.abs(value[ii])*0.05)**2 # default is 5% stdev
        qcov[qcov==0] = 1 # if initial value was zero, use 1 as stdev
        qcov = np.diagflat(qcov) # create covariance matrix

    return qcov        
    
def check_adascale(adascale, npar):
    # check adascale
    if adascale is None or adascale <= 0:
        qcov_scale = 2.4*(math.sqrt(npar)**(-1)) # scale factor in R
    else:
        qcov_scale = adascale
    
    return qcov_scale

def setup_R_matrix(qcov, parind):
    cm, cn = qcov.shape # number of rows, number of columns
    if min([cm, cn]) == 1: # qcov contains variances!
        s = np.sqrt(qcov[parind[:]])
        R = np.diagflat(s)
        qcovorig = np.diagflat(qcov[:]) # save original qcov
        qcov = np.diag(qcov[parind[:]])
    else: # qcov has covariance matrix in it
        qcovorig = qcov # save qcov
#        qcov = qcov[parind[:],parind[:]] # this operation in matlab maintains matrix (debug)
        R = np.linalg.cholesky(qcov) # cholesky decomposition
        R = R.transpose() # matches output of matlab function
    
    return R, qcov, qcovorig

def setup_RDR_matrix(R, invR, npar, drscale, ntry, options):
    RDR = options.RDR
    # if not empty
    if RDR is None: # check implementation
        RDR = [] # initialize listß
        RDR.append(R)
        invR.append(np.linalg.solve(R, np.eye(npar)))
        for ii in range(1,ntry):
            RDR.append(RDR[ii-1]*(drscale[min(ii,len(drscale))-1]**(-1)))
            invR.append(np.linalg.solve(RDR[ii],np.eye(npar)))
    else: # DR strategy: just scale R's down by DR_scale
        for ii in range(ntry):
            invR.append(np.linalg.solve(RDR[ii], np.eye(npar)))
                
        R = RDR[0]
            
    iacce = np.zeros(ntry)
        
    return RDR, invR, iacce, R

def check_dependent_parameters(N, data, nbatch, N0, S20, sigma2, savesize, nsimu, 
                               updatesigma, ntry, lastadapt, printint, adaptint):

    # check dependent parameters
    if nbatch is None:
        if isinstance(data, mcclass.DataStructure):
            nbatch = 1 # data is the class, not a list of classes
        else:
            nbatch = len(data) 
            # data is a list of classes, where each class constitutes a batch
#        genfun.message(verbosity, 1, 'Setting nbatch to 1\n')
            
    if N is None:
        N = 0
        for ii in range(nbatch):
            for kk in range(len(data[0].n)):
                N += data[ii].n[kk]
        N = np.array([N])
#        sys.exit('Could not determine number of data points, \n please specify model.N')

    # This is for backward compatibility
    # if sigma2 given then default N0=1, else default N0=0
    if N0 is None:
        if sigma2 is None:
            sigma2 = np.ones([1])
            N0 = np.zeros([1])
        else:
            N0 = np.ones([1])
    else:
        # if N0 given, then also turn on updatesigma
        updatesigma = 1    
        
    # save options
    if savesize <= 0 or savesize > nsimu:
        savesize = nsimu
    
    # turn on DR if ntry > 1
    if ntry > 1:
        dodram = 1
    else:
        dodram = 0
        
    # set default value for sigma2    
    # default for sigma2 is S20 or 1
    if sigma2 is None:
        if not(math.isnan(S20)):
            sigma2 = S20
        else:
            sigma2 = np.ones([nbatch])
    
    if np.isnan(S20):
        S20 = sigma2  # prior parameters for the error variance
    
    if N0 is None:
        N0 = np.array([1])
    
    if lastadapt < 1:
        lastadapt = nsimu
        
    if np.isnan(printint):
        printint = max(100,min(1000,adaptint))
        
    # in matlab version, ny = length(ss) where ss is the output from the sos evaluation
    # it should always have a length of 1, so I have chosen to simply define it as such
    ny = nbatch
    if len(S20)==1:
        S20 = np.ones(ny)*S20
        
    if len(N) == 1:
        N = np.ones(ny)*N
        
    if len(N) == ny + 1:
        N = N[1:] # remove first column
        
    if len(N0) == 1:
        N0 = np.ones(ny)*N0
        
    if len(sigma2) == 1:
        sigma2 = np.ones(ny)*sigma2
        
#    print('nbatch = {}, ny = {}'.format(nbatch, ny))
#    print('S20 = {}'.format(S20))
#    print('N = {}'.format(N))
#    print('N0 = {}'.format(N0))
        
    return N, nbatch, N0, updatesigma, savesize, dodram, sigma2, S20, lastadapt, printint, ny