#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 12:12:31 2017

Original files written in Matlab:
% Marko.Laine@helsinki.fi, 2003
% $Revision: 1.54 $  $Date: 2011/06/23 06:21:20 $
---------------
@author: prmiles
"""

# import required packages
from __future__ import division
import sys
import time
import numpy as np
#from inspect import signature
import classes as mcclass
import parameterfunctions as parfun
import generalfunctions as genfun
import mcmcfunctions as mcfun
import selection_algorithms as selalg
from progressbar import progress_bar as pbar

def mcmcrun(model, data, params, options, results = None):
    
    # unpack options structure
    nsimu = options.nsimu
    adaptint = options.adaptint
    ntry = options.ntry
    method = options.method
    printint = options.printint
    adaptend = options.adaptend
    lastadapt = options.lastadapt
    burnintime = options.burnintime
    progbar = options.progressbar
    debug = options.debug
    qcov = options.qcov
    updatesigma = options.updatesigma
    noadaptind = options.noadaptind
    stats = options.stats
    drscale = options.drscale
    adascale = options.adascale
    savesize = options.savesize
    maxmem = options.maxmem
    chainfile = options.chainfile
    s2chainfile = options.s2chainfile
    sschainfile = options.sschainfile
    savedir = options.savedir
    skip = options.skip
    label = options.label
    RDR = options.RDR
    verbosity = options.verbosity
    maxiter = options.maxiter
    priorupdatestart = options.priorupdatestart
    qcov_adjust = options.qcov_adjust
    burnin_scale = options.burnin_scale
    alphatarget = options.alphatarget
    etaparam = options.etaparam
    initqcovn = options.initqcovn
    doram = options.doram
    
    # unpack model structure
    sigma2 = model.sigma2
    N = model.N
    S20 = model.S20
    N0 = model.N0
    nbatch = model.nbatch
    ssfun = model.ssfun
    priorfun = model.priorfun
    priortype = model.priortype
    priorupdatefun = model.priorupdatefun
    priorpars = model.priorpars
    modelfun = model.modelfun

    # ---------------------------------
    # check dependent parameters
    N, nbatch, N0, updatesigma, savesize, dodram, sigma2, S20, lastadapt, printint, ny = mcfun.check_dependent_parameters(
            N, data, nbatch, N0, S20, sigma2, savesize, nsimu, updatesigma, 
            ntry, lastadapt, printint, adaptint)
            
    # open and parse the parameter structure
    names, value, parind, local, upp, low, thetamu, thetasig, npar = parfun.openparameterstructure(
            params, nbatch)
    
    # add check that thetasig > 0
    genfun.message(verbosity, 2, 'If prior variance <= 0, setting to Inf\n')
    thetasig = genfun.replace_list_elements(thetasig, genfun.less_than_or_equal_to_zero, float('Inf'))
    
    # display parameter settings
    if verbosity > 0:
        parfun.display_parameter_settings(parind, names, value, low, upp, thetamu, 
                               thetasig, noadaptind)
    
    # define noadaptind as a boolean - inputted as list of index values not updated
    no_adapt_index = mcfun.setup_no_adapt_index(noadaptind = noadaptind, parind = parind)
    
    # ----------------
    # setup covariance matrix
    qcov = mcfun.setup_covariance_matrix(qcov, thetasig, value)
        
    # ----------------
    # check adascale
    qcov_scale = mcfun.check_adascale(adascale, npar)
    
    # ----------------
    # setup R matrix (R used to update in metropolis)
    R, qcov, qcovorig = mcfun.setup_R_matrix(qcov, parind)
        
    # ----------------
    # setup RDR matrix (RDR used in DR)
    invR = []
    A_count = 0 # alphafun count
    if method == 'dram' or method == 'dr':
        RDR, invR, iacce, R = mcfun.setup_RDR_matrix(
                R = R, invR = invR, npar = npar, drscale = drscale, ntry = ntry,
                options = options)

    # ------------------
    # Perform initial evaluation
    starttime = time.clock()
    oldpar = value[parind[:]] # initial parameter set
    
    # ---------------------
    # check sum-of-squares function (this may not be necessary)
    ssstyle = 1
    if not ssfun: #isempty(ssfun)
        if not modelfun: #isempty(modelfun)
            sys.exit('No ssfun of modelfun specified!')
        ssstyle = 4
        
    # define sum-of-squares object
    sosobj = mcclass.SSObject(ssfun = ssfun, 
                              ssstyle = ssstyle, modelfun = modelfun,
                              parind = parind, local = local, data = data)
    
    # calculate sos with initial parameter set
    ss = sosobj.evaluate_sos(oldpar)
            
#    print('ss = {}'.format(ss))
    
    # ---------------------
    # define prior object
    priorobj = mcclass.PriorObject(priorfun = priorfun, 
                                   mu = thetamu, sigma = thetasig)
    
    # evaluate prior with initial parameter set
    oldprior = priorobj.evaluate_prior(oldpar)
    # ---------------------
    
    # Memory calculations????
    memneeded = savesize*(npar + 2*ny)*8*1e-6
    if (maxmem > 0) and (memneeded > maxmem):
        savesize = max(1000, np.floor(1e6*maxmem*((npar + 2*ny)*8)**(-1)))
        genfun.message(verbosity, 0, str('savesize decreased to {}\n'.format(savesize)))
    
    # check this part - matlab version: if (savesize < nsimu) || (nargout < 2) 
    saveit = 0
    if (savesize < nsimu):
        saveit = 1

    # ---------------------
    # Initialize chain, error variance, and SS
    chain = np.zeros([savesize,npar])
    sschain = np.zeros([savesize, ny])
    if updatesigma:
        s2chain = np.zeros([savesize, ny])
    else:
        s2chain = []
        
    # Save initialized values to chain, s2chain, sschain
    chainind = 0 # where we are in chain
    chain[chainind,:] = oldpar       
    sschain[chainind,:] = ss
    if updatesigma:
        s2chain[chainind,:] = sigma2
    
    # ---------------------
    # initialize reject test variables
    rej = np.zeros(1)
    reju = np.zeros(1)
    rejl = np.zeros(1)

    # ---------------------
    # initialize variables for covariance updates
    covchain = []
    meanchain = []
    wsum = initqcovn
    lasti = 0
    if wsum != []:
        covchain = qcov
        meanchain = oldpar

    # ---------------------
    # setup progress bar
    if progbar:
        pbarstatus = pbar(int(nsimu))
        
    # ----------------------------------------
    # start clocks
    mtime = []
    drtime = []
    adtime = []
    """
    Start main chain simulator
    """
    iiadapt = 0
    for isimu in range(1,nsimu): # simulation loop
        # update indexing
        iiadapt = iiadapt + 1 # local adaptation index
        chainind = chainind + 1
        # progress bar
        if progbar:
            pbarstatus.update(isimu)
            
        genfun.message(verbosity, 100, str('i:%d/%d\n'.format(isimu,nsimu)));
        
        # ---------------------------------------------------------
        # METROPOLIS ALGORITHM
        mtst = time.clock()
        oldset = mcclass.Parset(theta = oldpar, ss = ss, prior = oldprior,
                                sigma2 = sigma2)
        
        accept, newset, outbound, u = selalg.metropolis_algorithm(
                oldset = oldset, low = low, upp = upp, parind = parind, 
                npar = npar, R = R, priorobj = priorobj, sosobj = sosobj)

        mtend = time.clock()
        mtime.append(mtend-mtst)
        # --------------------------------------------------------
        # DELAYED REJECTION
        if dodram == 1 and accept == 0:
            drtst = time.clock()
            # perform a new try according to delayed rejection            
            accept, newset, iacce, outbound, A_count = selalg.delayed_rejection(
                    oldset = oldset, newset = newset, RDR = RDR, ntry = ntry,
                    npar = npar, low = low, upp = upp, parind = parind, 
                    iacce = iacce, A_count = A_count, invR = invR, 
                    sosobj = sosobj, priorobj = priorobj)
            drtend = time.clock()
            drtime.append(drtend-drtst)

        # ----------------dof----------------------------------------
        # SAVE CHAIN
        if accept:
            # accept
            chain[chainind,:] = newset.theta
            oldpar = newset.theta
            oldprior = newset.prior
            ss = newset.ss
        else:
            # reject
            chain[chainind,:] = oldset.theta
            rej = rej + 1
            reju = reju + 1
            if outbound:
                rejl = rejl + 1
                
        # UPDATE SUM-OF-SQUARES CHAIN
        sschain[chainind,:] = ss
        
        # UPDATE ERROR VARIANCE
        # VERIFIED GAMMAR FUNCTION (10/17/17)
        if updatesigma:
            for jj in range(0,ny):
                sigma2[jj] = (mcfun.gammar(1, 1, 0.5*(N0[jj]+N[jj]), 
                      2*((N0[jj]*S20[jj]+ss[jj])**(-1))))**(-1)
            s2chain[chainind,:] = sigma2
        
        if printint and np.fix(isimu/printint) == isimu/printint:
            genfun.message(verbosity, 2, 
                           str('i:{} ({},{},{})\n'.format(isimu,
                               rej*isimu**(-1)*100, reju*iiadapt**(-1)*100, 
                               rejl*isimu**(-1)*100)))
        
        # --------------------------------------------------------
        # ADAPTATION
        if adaptint > 0 and isimu <= lastadapt and np.fix(isimu/adaptint) == isimu/adaptint:
            adtst = time.clock()
            R, covchain, meanchain, wsum, lasti, RDR, invR, iiadapt, reju = selalg.adaptation(
                    isimu = isimu, burnintime = burnintime, rej = rej, rejl = rejl,
                    reju = reju, iiadapt = iiadapt, verbosity = verbosity, R = R,
                    burnin_scale = burnin_scale, chain = chain, lasti = lasti,
                    chainind = chainind, oldcovchain = covchain, oldmeanchain = meanchain,
                    oldwsum = wsum, doram = doram, u = u, etaparam = etaparam,
                    alphatarget = alphatarget, npar = npar, newset = newset, 
                    no_adapt_index = no_adapt_index, qcov = qcov, 
                    qcov_scale = qcov_scale, qcov_adjust = qcov_adjust, ntry = ntry,
                    drscale = drscale)
            
            adtend = time.clock()
            adtime.append(adtend-adtst)
            
        # --------------------------------------------------------
        # SAVE CHAIN
        if chainind == savesize and saveit == 1:
            genfun.message(verbosity, 2, 'saving chains\n')
            # add functionality
    # -------------------------------------------------------------------------
    """
    End main chain simulator
    """
    endtime = time.clock()
    simutime = endtime - starttime
    
    # SAVE REST OF CHAIN
    if chainind > 0 and saveit == 1:
        # save stuff
        print('add stuff here')
        
    # close waitbar
    
    # define last set of values
    thetalast = oldpar

    # --------------------------------------------                
    # CREATE OPTIONS OBJECT FOR DEBUG
    actual_options = mcclass.Options(nsimu=nsimu, adaptint=adaptint, ntry=ntry, 
                         method=method, printint=printint, adaptend = adaptend,
                         lastadapt = lastadapt, burnintime = burnintime,
                         progressbar = progbar, debug = debug, qcov = qcov,
                         updatesigma = updatesigma, noadaptind = noadaptind, 
                         stats = stats, drscale = drscale, adascale = adascale,
                         savesize = savesize, maxmem = maxmem, chainfile = chainfile,
                         s2chainfile = s2chainfile, sschainfile = sschainfile,
                         savedir = savedir, skip = skip, label = label, RDR = RDR,
                         verbosity = verbosity, maxiter = maxiter,
                         priorupdatestart = priorupdatestart, qcov_adjust = qcov_adjust,
                         burnin_scale = burnin_scale, alphatarget = alphatarget, 
                         etaparam = etaparam, initqcovn = initqcovn, doram = doram)
    
    # --------------------------------------------
    # BUILD RESULTS OBJECT
    tmp = mcclass.Results() # inititialize
        
    tmp.add_basic(label = label, rej = rej, rejl = rejl, R = R, method = method,
                              covchain = covchain, meanchain = meanchain, 
                              names = names, lowerlims = low, upperlims = upp,
                              thetalast = thetalast, parind = parind, 
                              local = local, nbatch = nbatch, N = N, 
                              modelfun = modelfun, ssfun = ssfun, nsimu = nsimu,
                              adaptint = adaptint, lastadapt = lastadapt,
                              adascale = adascale, skip = skip, 
                              simutime = simutime, ntry = ntry, qcovorig = qcovorig)
                  
    tmp.add_updatesigma(updatesigma = updatesigma, sigma2 = sigma2,
                            S20 = S20, N0 = N0)
    
    tmp.add_prior(mu = thetamu[parind[:]], sig = thetasig[parind[:]], 
                      priorfun = priorfun, priortype = priortype, 
                      priorpars = priorpars)
    
    if dodram == 1:
        tmp.add_dram(dodram = dodram, drscale = drscale, iacce = iacce,
                     alpha_count = A_count, RDR = RDR, nsimu = nsimu, rej = rej)
    
    tmp.add_options(options = actual_options)
    
    results = tmp.results # assign dictionary
    
    return results, chain, sschain, s2chain, mtime, drtime, adtime 