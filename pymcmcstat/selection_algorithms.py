#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 13:08:23 2017

@author: prmiles
"""
import classes as mcclass
#import parameterfunctions as parfun
#import generalfunctions as genfun
import mcmcfunctions as mcfun
import generalfunctions as genfun
import numpy as np
#import sys

# -------------------------------------------
def metropolis_algorithm(oldset, low, upp, parind, npar, R, priorobj, sosobj):
#    def metropolis_algorithm(oldset, low, upp, parind, npar, R, priorobj, sosobj, u = None, u2 = None):
    
    # unpack oldset
    oldpar = oldset.theta
    ss = oldset.ss
    oldprior = oldset.prior
    sigma2 = oldset.sigma2
    
    # Sample new candidate from Gaussian proposal
    u = np.random.randn(1,npar)
    newpar = oldpar + np.dot(u,R)   
    newpar = newpar.reshape(npar)
       
    # Reject points outside boundaries
    if (newpar < low[parind[:]]).any() or (newpar > upp[parind[:]]).any():
        # proposed value outside parameter limits
        accept = 0
        newprior = 0
        alpha = 0
        ss1 = np.inf
        ss2 = ss
        outbound = 1
    else:
        outbound = 0
        # prior SS for the new theta
        newprior = priorobj.evaluate_prior(newpar) 
        
        # calculate sum-of-squares
        ss2 = ss # old ss
        ss1 = sosobj.evaluate_sos(newpar)
        
        # evaluate test
#        print('sigma2 = {}'.format(sigma2))
        alpha = np.exp(-0.5*(sum((ss1 - ss2)*(sigma2**(-1))) + newprior - oldprior))
#        print('metropolis alpha = {}, ss1 = {}, ss2 = {}'.format(alpha, ss1, ss2))
        
#        if alpha == np.inf:
#            alpha = np.array(10) # greater than 1
        if alpha <= 0:
            accept = 0 # print('alpha_test = {:10s} <= 0, accept = {:1d}'.format(alpha_test, accept))
        elif alpha >= 1 or alpha > np.random.rand(1,1):
            accept = 1 # print('alpha_test = {:10s} >= 1, accept = {:1d}'.format(alpha_test, accept))
#        elif alpha >= 1:
#            accept = 1 # print('alpha_test = {:10s} >= 1, accept = {:1d}'.format(alpha_test, accept))
#        elif alpha > np.random.rand(1,1): # u2
#            accept = 1 # print('alpha_test = {:10s} > U(0,1), accept = {:1d}'.format(alpha_test, accept))
        else:
            accept = 0 # print('alpha_test = {:10s} < U(0,1), accept = {:1d}'.format(alpha_test, accept))
            
#        if debug and np.fix(isimu/debug) == isimu/debug:
#            print('{}: pri: {}, alpha: {}, ss: {}\n'.format(isimu, newprior, alpha_test, ss1))

    # store parameter sets in objects    
    newset = mcclass.Parset(theta = newpar, ss = ss1, prior = newprior,
                            sigma2 = sigma2, alpha = alpha)
    
    return accept, newset, outbound, u

# -------------------------------------------
def delayed_rejection(oldset, newset, RDR, ntry, npar, low, upp, 
                      parind, iacce, A_count, invR, sosobj, priorobj):
#                          parind, iacce, A_count, invR, sosobj, priorobj, u, u2):

    # initialize output object
#    outset = mcclass.Parset()
         
    # create trypath
    trypath = [oldset, newset]
    itry = 1; # dr step index
    accept = 0 # initialize acceptance criteria
#    outset = oldset
    while accept == 0 and itry < ntry:
        itry = itry + 1 # update dr step index
        # initialize next step parameter set
        nextset = mcclass.Parset()
        u = np.random.randn(1,npar) # u
        nextset.theta = oldset.theta + np.dot(u,RDR[itry-1])
        nextset.theta = nextset.theta.reshape(npar)
        nextset.sigma2 = newset.sigma2
                
        # Reject points outside boundaries
        if (nextset.theta < low[parind[:]]).any() or (nextset.theta > upp[parind[:]]).any():
            nextset.alpha = 0
            nextset.prior = 0
            nextset.ss = np.inf
            trypath.append(nextset)
            outbound = 1
            outset = oldset
            continue
                
        # Evaluate new proposals
        outbound = 0
        nextset.ss = sosobj.evaluate_sos(nextset.theta)
        nextset.prior = priorobj.evaluate_prior(nextset.theta)
        trypath.append(nextset) # add set to trypath
        alpha, A_count = mcfun.alphafun(trypath, A_count, invR)
        trypath[-1].alpha = alpha # add propability ratio
                       
        # check results of delayed rejection
        # u2 = np.random.rand(1)
        if alpha >= 1 or np.random.rand(1) < alpha: # accept
            accept = 1
            outset = nextset
            iacce[itry-1] += 1 # number accepted from DR
        else:
            outset = oldset
            
    return accept, outset, iacce, outbound, A_count
    
#                    
#        if debug and np.fix(isimu/debug) == isimu/debug:
#            print('try {}: pri: {}, alpha: {}\n'.format(itry, nextset.prior, alpha))
#            print(' p: {}\n'.format(nextset.theta))
# -------------------------------------------
def adaptation(isimu, burnintime, rej, rejl, reju, iiadapt, verbosity, R, burnin_scale, chain, lasti,
               chainind, oldcovchain, oldmeanchain, oldwsum, doram, u, etaparam, alphatarget,
               npar, newset, no_adapt_index, qcov, qcov_scale, qcov_adjust, 
               ntry, drscale):
    # initialize DR components
    RDR = None
    invR = None
        
    if isimu < burnintime:
        # during burnin no adaptation, just scaling down
        if reju*(iiadapt**(-1)) > 0.95:
            genfun.message(verbosity, 2, str(' (burnin/down) {3.2f}'.format(
                    reju*(iiadapt**(-1))*100)))
            R = R*(burnin_scale**(-1))
        elif reju*(iiadapt**(-1)) < 0.05:
            genfun.message(verbosity, 2, str(' (burnin/up) {3.2f}'.format(
                    reju*(iiadapt**(-1))*100)))
            R = R*burnin_scale
                
    else:
        genfun.message(verbosity, 2, str('i:{} adapting ({}, {}, {})'.format(
                isimu, rej*(isimu**(-1))*100, reju*(iiadapt**(-1))*100, 
                rejl*(isimu**(-1))*100)))


#        print('lasti = {}, chainind = {}'.format(lasti, chainind))
#        print('chain[{}:{},:] = {}'.format(lasti, chainind, chain[lasti:chainind,:]))
#        print('covchain = {}'.format(oldcovchain))
#        print('meanchain = {}'.format(oldmeanchain))
#        print('wsum = {}'.format(oldwsum))
#         update covariance matrix - cholesky
        # VERIFIED NOVEMBER 1, 2017
        covchain, meanchain, wsum = mcfun.covupd(
                chain[lasti:chainind,:], np.ones(1), oldcovchain, oldmeanchain, oldwsum)
                
#        print('covchain = {}'.format(covchain))
#        print('meanchain = {}'.format(meanchain))
#        print('wsum = {}'.format(wsum))
#        
#        if isimu > 200:
#            sys.exit()
        
        lasti = isimu
                
        # ram
        if doram:
            uu = u*(np.linalg.norm(u)**(-1))
            eta = (isimu**(etaparam))**(-1)
            ram = np.eye(npar) + eta*(min(1, newset.alpha) - alphatarget)*(
                    np.dot(uu.transpose(), uu))
            upcov = np.dot(np.dot(R.transpose(),ram),R)
        else:
            upcov = covchain
            upcov[no_adapt_index, :] = qcov[no_adapt_index,:]
            upcov[:,no_adapt_index] = qcov[:,no_adapt_index]
#            print('covchain = {}'.format(covchain))
#            print('qcov = {}'.format(qcov))
#            print('upcov = {}'.format(upcov))
#            sys.exit()
                    
        # check if singular covariance matrix
        pos_def, pRa = genfun.is_semi_pos_def_chol(upcov)
        if pos_def == 1: # not singular!
            Ra = pRa # np.linalg.cholesky(upcov)
#            Ra = Ra.transpose()
            R = Ra*qcov_scale
#            print('Ra = {}, qcov_scale = {}, R = {}, upcov = {}'.format(Ra, qcov_scale, R, upcov))
#            print('covchain = {}'.format(covchain))
#            sys.exit()
                    
        else: # singular covariance matrix
            # try to blow it up
            tmp = upcov + np.eye(npar)*qcov_adjust
            pos_def_adjust, pRa = genfun.is_semi_pos_def_chol(tmp)
            if pos_def_adjust == 1: # not singular!
                Ra = pRa
                genfun.message(verbosity, 1, 'adjusted covariance matrix')
                # scale R
                R = Ra*qcov_scale
            else: # still singular...
                errstr = str('convariance matrix singular, no adaptation')
                genfun.message(verbosity, 0, '{} {}'.format(errstr, reju*(iiadapt**(-1))*100))
                
        # update dram covariance matrix
        if ntry > 1: # delayed rejection
            RDR = []
            invR = []
            RDR.append(R)
            invR.append(np.linalg.solve(RDR[0], np.eye(npar)))
#            print('-----------')
#            print('RDR[{}] = {}'.format(0,RDR[0]))
#            print('invR[{}] = {}'.format(0,invR[0]))
            for ii in range(1,ntry):
#                print('DR scale = {}'.format(drscale[min(ii,len(drscale)) - 1]))
                RDR.append(RDR[ii-1]*((drscale[min(ii,len(drscale)) - 1])**(-1))) 
                invR.append(invR[ii-1]*(drscale[min(ii,len(drscale)) - 1]))
#                print('RDR[{}] = {}'.format(ii,RDR[ii]))
#                print('invR[{}] = {}'.format(ii,invR[ii]))
       
    genfun.message(verbosity, 2, '\n')
    reju = 0
    iiadapt = 0 # reset local adaptation index
    
    return R, covchain, meanchain, wsum, lasti, RDR, invR, iiadapt, reju