#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17, 2018

Original files written in Matlab:
% Marko.Laine@helsinki.fi, 2003
% $Revision: 1.54 $  $Date: 2011/06/23 06:21:20 $
---------------
Python author: prmiles

Description: This module is intended to be the main class from which to run these
Markov Chain Monte Carlo type simulations.  The user will create an MCMC object,
initialize options, model settings, and parameters.  Simulations can then be run
as well as several types of predictive tests.
"""

# import required packages
import sys
#import os
import time
import numpy as np
#from inspect import signature
import classes as mcclass
import parameterfunctions as parfun
import generalfunctions as genfun
import mcmcfunctions as mcfun
import selection_algorithms as selalg
from progressbar import progress_bar as pbar

def mcmcrun(model, data, params, options, previous_results = None):
    
    # unpack model and options structures
    # general settings
    nsimu = options.nsimu # number of chain interates
    method = options.method # sampling method ('mh', 'am', 'dr', 'dram')
    waitbar = options.waitbar # flag to display progress bar
    debug = options.debug # display certain features to assist in code debugging
    noadaptind = options.noadaptind # do not adapt these indices
    stats = options.stats # convergence statistics
    verbosity = options.verbosity
    printint = options.printint
    nbatch = model.nbatch # number of batches
    rndseq = options.rndseq # random numbers for testing
        
    # settings for adaptation
    adaptint = options.adaptint # number of iterates between adaptation
    qcov = options.qcov # proposal covariance
    qcov_adjust = options.qcov_adjust
    initqcovn = options.initqcovn # proposal covariance weight in update
    adascale = options.adascale # user defined covariance scale
    lastadapt = options.lastadapt # last adapt (i.e., no more adaptation beyond this iteration)
    burnintime = options.burnintime
    burnin_scale = options.burnin_scale # scale in burn-in down/up
    
    # settings for updating error variance estimator
    updatesigma = options.updatesigma # flag saying whether or not to update the measurement variance estimate
    sigma2 = model.sigma2 # initial value for error variance
    N = model.N # number of observations
    S20 = model.S20 # error variance prior
    N0 = model.N0
    
    # settings associated with saving to bin files
    savesize = options.savesize # rows of the chain in memory
    maxmem = options.maxmem # memory available in mega bytes
    chainfile = options.chainfile # chain file name
    s2chainfile = options.s2chainfile # s2chain file name
    sschainfile = options.sschainfile # sschain file name
    savedir = options.savedir # directory files saved to
    skip = options.skip
    label = options.label
    
    # settings for delayed rejection
    ntry = options.ntry # number of stages in delayed rejection algorithm
    RDR = options.RDR # R matrix for delayed rejection
    drscale = options.drscale # scale sampling distribution for delayed rejection
    alphatarget = options.alphatarget # acceptance ratio target
    etaparam = options.etaparam
    doram = options.doram
    
    # settings for sum-of-squares function
    ssfun = model.ssfun
    modelfun = model.modelfun
    
    # settings for prior function
    priorfun = model.priorfun
    priortype = model.priortype
    priorupdatefun = model.priorupdatefun
    priorpars = model.priorpars
    priorupdatestart = options.priorupdatestart

    # ---------------------------------
    # check dependent parameters
    N, nbatch, N0, updatesigma, savesize, dodram, sigma2, S20, lastadapt, printint, ny = mcfun.check_dependent_parameters(
            N, data, nbatch, N0, S20, sigma2, savesize, nsimu, updatesigma, 
            ntry, lastadapt, printint, adaptint)
        
#    print('N0 = {}'.format(N0[0]))
#    print('S20 = {}'.format(S20))
    # use values from previous run (if inputted)
    if previous_results != None:
        genfun.message(verbosity, 0, '\nUsing values from the previous run')
        params = parfun.results2params(previous_results, params, 1) # 1 = do local parameters
        qcov = previous_results['qcov'] # previous_results should match the output format for the results class structure

    # open and parse the parameter structure
    names, value, parind, local, upp, low, thetamu, thetasig, npar = parfun.openparameterstructure(
            params, nbatch)
    
    # check initial parameter values are inside range
    if (value < low[parind[:]]).any() or (value > upp[parind[:]]).any():
        # proposed value outside parameter limits
        sys.exit('Proposed value outside parameter limits - select new initial parameter values')
        
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
    if ssfun is None: #isempty(ssfun)
        if modelfun is None: #isempty(modelfun)
            sys.exit('No ssfun of modelfun specified!')
        ssstyle = 4
        
    # define sum-of-squares object
    sosobj = mcclass.SSObject(ssfun = ssfun, 
                              ssstyle = ssstyle, modelfun = modelfun,
                              parind = parind, local = local, data = data,
                              nbatch = nbatch)
    
    # calculate sos with initial parameter set
    ss = sosobj.evaluate_sos(oldpar)
    
    # ---------------------
    # define prior object
    priorobj = mcclass.PriorObject(priorfun = priorfun, 
                                   mu = thetamu, sigma = thetasig)
    
    # evaluate prior with initial parameter set
    oldprior = priorobj.evaluate_prior(oldpar)
  
    # ---------------------
    # Initialize chain, error variance, and SS
    chain = np.zeros([savesize,npar])
    sschain = np.zeros([savesize, ny])
    if updatesigma:
        s2chain = np.zeros([savesize, ny])
    else:
        s2chain = None
        
    # Save initialized values to chain, s2chain, sschain
    chainind = 0 # where we are in chain
    chain[chainind,:] = oldpar       
    sschain[chainind,:] = ss
    if updatesigma:
        s2chain[chainind,:] = sigma2
  
    # ---------------------
    # Memory calculations
    memneeded = savesize*(npar + 2*ny)*8*1e-6
    if (maxmem > 0) and (memneeded > maxmem):
        savesize = max(1000, np.floor(1e6*maxmem*((npar + 2*ny)*8)**(-1)))
        genfun.message(verbosity, 0, str('savesize decreased to {}\n'.format(savesize)))
    
    saveit = 0
    if (savesize < nsimu):
        saveit = 1
        
#    savedir = 'results' # default directory
#    try:
#        os.mkdir(savedir)
#    except Exception:
#        pass
#
#    # save chain
#    if saveit == 1:
##        savebin(chainfile, [], 'chain')
#        with open(os.path.join(savedir, chainfile), "a") as myfile:
#            myfile.write(chain)
    # ---------------------
    # initialize reject test variables
    rej = np.zeros(1)
    reju = np.zeros(1)
    rejl = np.zeros(1)

    # ---------------------
    # initialize variables for covariance updates
    covchain = None
    meanchain = None
    wsum = initqcovn
    lasti = 0
    if wsum is not None:
        covchain = qcov
        meanchain = oldpar

    # ---------------------
    # setup progress bar
    if waitbar:
        wbarstatus = pbar(int(nsimu))


    # display settings going into simulation    
#    print('N = {}, nbatch = {}, N0 = {}, updsig = {}'.format(N, nbatch, N0, updatesigma))
#    print('savesize = {}, dodram = {}, sigma2 = {}'.format(savesize, dodram, sigma2))
#    print('covchain = {}'.format(covchain))
#    print('qcov = {}'.format(qcov))
#    print('adaptint = {}'.format(adaptint))
    
    # unpack random number sequences
#    mhrndseq = rndseq[0]
#    mhrndseq2 = rndseq[1]
#    drrndseq = rndseq[2]
#    drrndseq2 = rndseq[3]
#    
#    rndnum_u_n = data.udobj[0]
    
    # ----------------------------------------
    # start clocks
    mtime = []
    drtime = []
    adtime = []
#    rndseq = []
    """
    Start main chain simulator
    """
    iiadapt = 0
    iiprint = 0
    for isimu in range(1,nsimu): # simulation loop
        # update indexing
        iiadapt += 1 # local adaptation index
        iiprint += 1 # local print index
        chainind += 1
        # progress bar
        if waitbar:
            wbarstatus.update(isimu)
            
        genfun.message(verbosity, 100, str('i:%d/%d\n'.format(isimu,nsimu)));
        
        # ---------------------------------------------------------
        # METROPOLIS ALGORITHM
#        mtst = time.clock()
        oldset = mcclass.Parset(theta = oldpar, ss = ss, prior = oldprior,
                                sigma2 = sigma2)

        accept, newset, outbound, u = selalg.metropolis_algorithm(
                oldset = oldset, low = low, upp = upp, parind = parind, 
                npar = npar, R = R, priorobj = priorobj, sosobj = sosobj)

#        mtend = time.clock()
#        mtime.append(mtend-mtst)
        # --------------------------------------------------------
        # DELAYED REJECTION
#        print('isimu = {}, theta = {}, accept = {}'.format(isimu, newset.theta, accept))
        if dodram == 1 and accept == 0:
#            drtst = time.clock()
            # perform a new try according to delayed rejection 
            accept, newset, iacce, outbound, A_count = selalg.delayed_rejection(
                    oldset = oldset, newset = newset, RDR = RDR, ntry = ntry,
                    npar = npar, low = low, upp = upp, parind = parind, 
                    iacce = iacce, A_count = A_count, invR = invR, 
                    sosobj = sosobj, priorobj = priorobj)
#            drtend = time.clock()
#            drtime.append(drtend-drtst)

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
        # CHECK VECTOR COMPATITIBILITY
        if updatesigma:
            for jj in range(0,ny):
#                print('jj = {}'.format(jj))
#                print('N0[{}] = {}'.format(jj,N0[jj]))
#                print('N[{}] = {}'.format(jj,N[jj]))
#                print('S20[{}] = {}'.format(jj,S20[jj]))
#                print('ss[{}] = {}'.format(jj,ss[jj]))
                sigma2[jj] = (mcfun.gammar(1, 1, 0.5*(N0[jj]+N[jj]),
                      2*((N0[jj]*S20[jj]+ss[jj])**(-1))))**(-1)
#                sigma2[jj] = (mcfun.gammar(1, 1, 0.5*(N0[jj]+N[jj]), rndnum_u_n[isimu,:],
#                      2*((N0[jj]*S20[jj]+ss[jj])**(-1))))**(-1)
            s2chain[chainind,:] = sigma2
        
        if printint and iiprint + 1 == printint:
            genfun.message(verbosity, 2, 
                           str('i:{} ({},{},{})\n'.format(isimu,
                               rej*isimu**(-1)*100, reju*iiadapt**(-1)*100, 
                               rejl*isimu**(-1)*100)))
            iiprint = 0 # reset print counter
        
        # --------------------------------------------------------
        # ADAPTATION
        if adaptint > 0 and iiadapt == adaptint:
#        if adaptint > 0 and isimu <= lastadapt - 1 and np.fix(
#                (isimu+1)*(adaptint**(-1))) == (isimu + 1)*(adaptint**(-1)):
#            print('Adapting on step {} of {}'.format(isimu + 1, nsimu))
#            print('lastadapt = {}, adaptint = {}'.format(lastadapt, adaptint))
#            print('R = {}'.format(R))
#            adtst = time.clock()
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
            
#            adtend = time.clock()
#            adtime.append(adtend-adtst)
            
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
    
    # define last set of values
    thetalast = oldpar
    
#    print('\n Simulation complete - Generate results structure...\n')

    # --------------------------------------------                
    # CREATE OPTIONS OBJECT FOR DEBUG
    actual_options = mcclass.Options(nsimu=nsimu, adaptint=adaptint, ntry=ntry, 
                         method=method, printint=printint,
                         lastadapt = lastadapt, burnintime = burnintime,
                         waitbar = waitbar, debug = debug, qcov = qcov,
                         updatesigma = updatesigma, noadaptind = noadaptind, 
                         stats = stats, drscale = drscale, adascale = adascale,
                         savesize = savesize, maxmem = maxmem, chainfile = chainfile,
                         s2chainfile = s2chainfile, sschainfile = sschainfile,
                         savedir = savedir, skip = skip, label = label, RDR = RDR,
                         verbosity = verbosity,
                         priorupdatestart = priorupdatestart, qcov_adjust = qcov_adjust,
                         burnin_scale = burnin_scale, alphatarget = alphatarget, 
                         etaparam = etaparam, initqcovn = initqcovn, doram = doram)
    
    actual_model_settings = mcclass.Model(
            ssfun = ssfun, priorfun = priorfun, priortype = priortype, 
            priorupdatefun = priorupdatefun, priorpars = priorpars, 
            modelfun = modelfun, sigma2 = sigma2, N = N, S20 = S20, N0 = N0, 
            nbatch = nbatch)
    # --------------------------------------------
    # BUILD RESULTS OBJECT
    tmp = mcclass.Results() # inititialize
        
    tmp.add_basic(nsimu = nsimu, rej = rej, rejl = rejl, R = R, covchain = covchain, 
                  meanchain = meanchain, names = names, lowerlims = low, 
                  upperlims = upp, theta = thetalast, parind = parind, 
                  local = local, simutime = simutime, qcovorig = qcovorig)
                  
    tmp.add_updatesigma(updatesigma = updatesigma, sigma2 = sigma2,
                            S20 = S20, N0 = N0)
    
    tmp.add_prior(mu = thetamu[parind[:]], sig = thetasig[parind[:]], 
                      priorfun = priorfun, priortype = priortype, 
                      priorpars = priorpars)
    
    if dodram == 1:
        tmp.add_dram(dodram = dodram, drscale = drscale, iacce = iacce,
                     alpha_count = A_count, RDR = RDR, nsimu = nsimu, rej = rej)
    
    tmp.add_options(options = actual_options)
    tmp.add_model(model = actual_model_settings)
    
    # add chain, s2chain, and sschain
    tmp.add_chain(chain = chain)
    tmp.add_s2chain(s2chain = s2chain)
    tmp.add_sschain(sschain = sschain)
    
    # add time statistics
    tmp.add_time_stats(mtime, drtime, adtime)
    
    # add random number sequence
    tmp.add_random_number_sequence(rndseq)
    
    results = tmp.results # assign dictionary
    
    return results