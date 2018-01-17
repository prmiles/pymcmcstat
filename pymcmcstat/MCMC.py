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

from DataStructure import DataStructure
from ModelSettings import ModelSettings
from ModelParameters import ModelParameters
from SimulationOptions import SimulationOptions
from ResultsStructure import ResultsStructure
from SumOfSquares import SumOfSquares
import parameterfunctions as parfun
import generalfunctions as genfun
import mcmcfunctions as mcfun
import selection_algorithms as selalg
from progressbar import progress_bar as pbar

class MCMC:
    def __init__(self):
        
#        self.add_empty_data_batch(initial_batch_flag = None) 
        self.data = DataStructure()
        self.model = ModelSettings()
        self.options = SimulationOptions()
        self.parameters = ModelParameters()

#    def add_empty_data_batch(self, initial_batch_flag = 1):
#        # Check if data already initialized
#        if initial_batch_flag is None:
#            self.data = []
#            
#        self.data.append(DataStructure())
            

    def run_simulation(self, previous_results = None):
    
        # ---------------------------------
        # check dependent parameters
        self.options.check_dependent_simulation_options(self.data, self.model)
        self.model.check_dependent_model_settings(self.data, self.options)
        
#        # ******************************************
#        # PENDING UPDATE
#        # use values from previous run (if inputted)
#        if previous_results != None:
#            genfun.message(self.options.verbosity, 0, '\nUsing values from the previous run')
#            self.parameters.results2params(previous_results, 1) # 1 = do local parameters
#            qcov = previous_results['qcov'] # previous_results should match the output format for the results class structure
#        # ******************************************
        
        # open and parse the parameter structure
        self.parameters.openparameterstructure(self.model.nbatch)
        # check initial parameter values are inside range
        self.parameters.check_initial_values_wrt_parameter_limits()
        # add check that prior standard deviation > 0
        self.parameters.check_prior_sigma(self.options.verbosity)

#        
#        # display parameter settings
#        if verbosity > 0:
#            parfun.display_parameter_settings(parind, names, value, low, upp, thetamu, 
#                                   thetasig, noadaptind)
#        
#        # define noadaptind as a boolean - inputted as list of index values not updated
#        no_adapt_index = mcfun.setup_no_adapt_index(noadaptind = noadaptind, parind = parind)
#        
#        # ----------------
#        # setup covariance matrix
#        qcov = mcfun.setup_covariance_matrix(qcov, thetasig, value)
#            
#        # ----------------
#        # check adascale
#        qcov_scale = mcfun.check_adascale(adascale, npar)
#        
#        # ----------------
#        # setup R matrix (R used to update in metropolis)
#        R, qcov, qcovorig = mcfun.setup_R_matrix(qcov, parind)
#            
#        # ----------------
#        # setup RDR matrix (RDR used in DR)
#        invR = []
#        A_count = 0 # alphafun count
#        if method == 'dram' or method == 'dr':
#            RDR, invR, iacce, R = mcfun.setup_RDR_matrix(
#                    R = R, invR = invR, npar = npar, drscale = drscale, ntry = ntry,
#                    options = options)
#    
#        # ------------------
#        # Perform initial evaluation
#        starttime = time.clock()
#        oldpar = value[parind[:]] # initial parameter set
#        
#        # ---------------------
#        # check sum-of-squares function (this may not be necessary)
#        ssstyle = 1
#        if ssfun is None: #isempty(ssfun)
#            if modelfun is None: #isempty(modelfun)
#                sys.exit('No ssfun or modelfun specified!')
#            ssstyle = 4
#            
#        # define sum-of-squares object
#        sosobj = SumOfSquares(sos_function = self.model.ssfun, 
#                                  sos_style = ssstyle, model_function = self.model.modelfun,
#                                  parind = self.parameters.parind, local = self.parameters.local, data = self.data,
#                                  nbatch = self.options.nbatch)
#        
#        # calculate sos with initial parameter set
#        ss = sosobj.evaluate_sos(oldpar)
#        
#        # ---------------------
#        # define prior object
#        priorobj = mcclass.PriorObject(priorfun = priorfun, 
#                                       mu = thetamu, sigma = thetasig)
#        
#        # evaluate prior with initial parameter set
#        oldprior = priorobj.evaluate_prior(oldpar)
#      
#        # ---------------------
#        # Initialize chain, error variance, and SS
#        chain = np.zeros([savesize,npar])
#        sschain = np.zeros([savesize, ny])
#        if updatesigma:
#            s2chain = np.zeros([savesize, ny])
#        else:
#            s2chain = None
#            
#        # Save initialized values to chain, s2chain, sschain
#        chainind = 0 # where we are in chain
#        chain[chainind,:] = oldpar       
#        sschain[chainind,:] = ss
#        if updatesigma:
#            s2chain[chainind,:] = sigma2
#      
#        # ---------------------
#        # initialize reject test variables
#        rej = np.zeros(1)
#        reju = np.zeros(1)
#        rejl = np.zeros(1)
#    
#        # ---------------------
#        # initialize variables for covariance updates
#        covchain = None
#        meanchain = None
#        wsum = initqcovn
#        lasti = 0
#        if wsum is not None:
#            covchain = qcov
#            meanchain = oldpar
#    
#        # ---------------------
#        # setup progress bar
#        if waitbar:
#            wbarstatus = pbar(int(nsimu))
#    
#    
#        # display settings going into simulation    
#    #    print('N = {}, nbatch = {}, N0 = {}, updsig = {}'.format(N, nbatch, N0, updatesigma))
#    #    print('savesize = {}, dodram = {}, sigma2 = {}'.format(savesize, dodram, sigma2))
#    #    print('covchain = {}'.format(covchain))
#    #    print('qcov = {}'.format(qcov))
#    #    print('adaptint = {}'.format(adaptint))
#        
#        # ----------------------------------------
#        # start clocks
#        mtime = []
#        drtime = []
#        adtime = []
#    #    rndseq = []
#        """
#        Start main chain simulator
#        """
#        iiadapt = 0
#        iiprint = 0
#        for isimu in range(1,nsimu): # simulation loop
#            # update indexing
#            iiadapt += 1 # local adaptation index
#            iiprint += 1 # local print index
#            chainind += 1
#            # progress bar
#            if waitbar:
#                wbarstatus.update(isimu)
#                
#            genfun.message(verbosity, 100, str('i:%d/%d\n'.format(isimu,nsimu)));
#            
#            # ---------------------------------------------------------
#            # METROPOLIS ALGORITHM
#    #        mtst = time.clock()
#            oldset = mcclass.Parset(theta = oldpar, ss = ss, prior = oldprior,
#                                    sigma2 = sigma2)
#    
#            accept, newset, outbound, u = selalg.metropolis_algorithm(
#                    oldset = oldset, low = low, upp = upp, parind = parind, 
#                    npar = npar, R = R, priorobj = priorobj, sosobj = sosobj)
#    
#    #        mtend = time.clock()
#    #        mtime.append(mtend-mtst)
#            # --------------------------------------------------------
#            # DELAYED REJECTION
#    #        print('isimu = {}, theta = {}, accept = {}'.format(isimu, newset.theta, accept))
#            if dodram == 1 and accept == 0:
#    #            drtst = time.clock()
#                # perform a new try according to delayed rejection 
#                accept, newset, iacce, outbound, A_count = selalg.delayed_rejection(
#                        oldset = oldset, newset = newset, RDR = RDR, ntry = ntry,
#                        npar = npar, low = low, upp = upp, parind = parind, 
#                        iacce = iacce, A_count = A_count, invR = invR, 
#                        sosobj = sosobj, priorobj = priorobj)
#    #            drtend = time.clock()
#    #            drtime.append(drtend-drtst)
#    
#            # ----------------dof----------------------------------------
#            # SAVE CHAIN
#            if accept:
#                # accept
#                chain[chainind,:] = newset.theta
#                oldpar = newset.theta
#                oldprior = newset.prior
#                ss = newset.ss
#            else:
#                # reject
#                chain[chainind,:] = oldset.theta
#                rej = rej + 1
#                reju = reju + 1
#                if outbound:
#                    rejl = rejl + 1
#                    
#            # UPDATE SUM-OF-SQUARES CHAIN
#            sschain[chainind,:] = ss
#            
#            # UPDATE ERROR VARIANCE
#            # VERIFIED GAMMAR FUNCTION (10/17/17)
#            # CHECK VECTOR COMPATITIBILITY
#            if updatesigma:
#                for jj in range(0,ny):
#    #                print('jj = {}'.format(jj))
#    #                print('N0[{}] = {}'.format(jj,N0[jj]))
#    #                print('N[{}] = {}'.format(jj,N[jj]))
#    #                print('S20[{}] = {}'.format(jj,S20[jj]))
#    #                print('ss[{}] = {}'.format(jj,ss[jj]))
#                    sigma2[jj] = (mcfun.gammar(1, 1, 0.5*(N0[jj]+N[jj]),
#                          2*((N0[jj]*S20[jj]+ss[jj])**(-1))))**(-1)
#    #                sigma2[jj] = (mcfun.gammar(1, 1, 0.5*(N0[jj]+N[jj]), rndnum_u_n[isimu,:],
#    #                      2*((N0[jj]*S20[jj]+ss[jj])**(-1))))**(-1)
#                s2chain[chainind,:] = sigma2
#            
#            if printint and iiprint + 1 == printint:
#                genfun.message(verbosity, 2, 
#                               str('i:{} ({},{},{})\n'.format(isimu,
#                                   rej*isimu**(-1)*100, reju*iiadapt**(-1)*100, 
#                                   rejl*isimu**(-1)*100)))
#                iiprint = 0 # reset print counter
#            
#            # --------------------------------------------------------
#            # ADAPTATION
#            if adaptint > 0 and iiadapt == adaptint:
#    #        if adaptint > 0 and isimu <= lastadapt - 1 and np.fix(
#    #                (isimu+1)*(adaptint**(-1))) == (isimu + 1)*(adaptint**(-1)):
#    #            print('Adapting on step {} of {}'.format(isimu + 1, nsimu))
#    #            print('lastadapt = {}, adaptint = {}'.format(lastadapt, adaptint))
#    #            print('R = {}'.format(R))
#    #            adtst = time.clock()
#                R, covchain, meanchain, wsum, lasti, RDR, invR, iiadapt, reju = selalg.adaptation(
#                        isimu = isimu, burnintime = burnintime, rej = rej, rejl = rejl,
#                        reju = reju, iiadapt = iiadapt, verbosity = verbosity, R = R,
#                        burnin_scale = burnin_scale, chain = chain, lasti = lasti,
#                        chainind = chainind, oldcovchain = covchain, oldmeanchain = meanchain,
#                        oldwsum = wsum, doram = doram, u = u, etaparam = etaparam,
#                        alphatarget = alphatarget, npar = npar, newset = newset, 
#                        no_adapt_index = no_adapt_index, qcov = qcov, 
#                        qcov_scale = qcov_scale, qcov_adjust = qcov_adjust, ntry = ntry,
#                        drscale = drscale)
#                
#    #            adtend = time.clock()
#    #            adtime.append(adtend-adtst)
#                
#            # --------------------------------------------------------
#            # SAVE CHAIN
#            if chainind == savesize and saveit == 1:
#                genfun.message(verbosity, 2, 'saving chains\n')
#                # add functionality
#        # -------------------------------------------------------------------------
#        """
#        End main chain simulator
#        """
#        endtime = time.clock()
#        simutime = endtime - starttime
#        
#        # SAVE REST OF CHAIN
#        if chainind > 0 and saveit == 1:
#            # save stuff
#            print('add stuff here')
#        
#        # define last set of values
#        thetalast = oldpar
#        
#    #    print('\n Simulation complete - Generate results structure...\n')
#    
#        # --------------------------------------------                
#        # CREATE OPTIONS OBJECT FOR DEBUG
#        actual_options = mcclass.Options(nsimu=nsimu, adaptint=adaptint, ntry=ntry, 
#                             method=method, printint=printint,
#                             lastadapt = lastadapt, burnintime = burnintime,
#                             waitbar = waitbar, debug = debug, qcov = qcov,
#                             updatesigma = updatesigma, noadaptind = noadaptind, 
#                             stats = stats, drscale = drscale, adascale = adascale,
#                             savesize = savesize, maxmem = maxmem, chainfile = chainfile,
#                             s2chainfile = s2chainfile, sschainfile = sschainfile,
#                             savedir = savedir, skip = skip, label = label, RDR = RDR,
#                             verbosity = verbosity,
#                             priorupdatestart = priorupdatestart, qcov_adjust = qcov_adjust,
#                             burnin_scale = burnin_scale, alphatarget = alphatarget, 
#                             etaparam = etaparam, initqcovn = initqcovn, doram = doram)
#        
#        actual_model_settings = mcclass.Model(
#                ssfun = ssfun, priorfun = priorfun, priortype = priortype, 
#                priorupdatefun = priorupdatefun, priorpars = priorpars, 
#                modelfun = modelfun, sigma2 = sigma2, N = N, S20 = S20, N0 = N0, 
#                nbatch = nbatch)
#        # --------------------------------------------
#        # BUILD RESULTS OBJECT
#        self.simulation_results = ResultsStructure() # inititialize
#            
#        tmp.add_basic(nsimu = nsimu, rej = rej, rejl = rejl, R = R, covchain = covchain, 
#                      meanchain = meanchain, names = names, lowerlims = low, 
#                      upperlims = upp, theta = thetalast, parind = parind, 
#                      local = local, simutime = simutime, qcovorig = qcovorig)
#                      
#        tmp.add_updatesigma(updatesigma = updatesigma, sigma2 = sigma2,
#                                S20 = S20, N0 = N0)
#        
#        tmp.add_prior(mu = thetamu[parind[:]], sig = thetasig[parind[:]], 
#                          priorfun = priorfun, priortype = priortype, 
#                          priorpars = priorpars)
#        
#        if dodram == 1:
#            tmp.add_dram(dodram = dodram, drscale = drscale, iacce = iacce,
#                         alpha_count = A_count, RDR = RDR, nsimu = nsimu, rej = rej)
#        
#        tmp.add_options(options = actual_options)
#        tmp.add_model(model = actual_model_settings)
#        
#        # add chain, s2chain, and sschain
#        tmp.add_chain(chain = chain)
#        tmp.add_s2chain(s2chain = s2chain)
#        tmp.add_sschain(sschain = sschain)
#        
#        # add time statistics
#        tmp.add_time_stats(mtime, drtime, adtime)
#        
#        # add random number sequence
#        tmp.add_random_number_sequence(rndseq)
#        
#        results = tmp.results # assign dictionary
#        
#        return results