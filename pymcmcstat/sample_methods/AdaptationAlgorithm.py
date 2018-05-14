#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 11:14:11 2018

@author: prmiles
"""
# import required packages
import numpy as np
import math

class AdaptationAlgorithm:
    """
    Adaptive Metropolis (AM) algorithm based on [haario2001adaptive]_ (`URL <https://projecteuclid.org/euclid.bj/1080222083>`_)
        
    .. [haario2001adaptive] Haario, Heikki, Eero Saksman, and Johanna Tamminen. "An adaptive Metropolis algorithm." Bernoulli 7, no. 2 (2001): 223-242. 
        
    """
    def __init__(self):
        self.label = 'Covariance Variables and Methods'
        self.qcov = None
        self.qcov_scale = None
        self.R = None
        self.qcov_original = None
        self.invR = None
        self.iacce = None
        
        self.covchain = None
        self.meanchain = None
        
        self.last_index_since_adaptation = 0
        
    # -------------------------------------------
    def run_adaptation(self, covariance, options, isimu, iiadapt, rejected, chain, chainind, u, npar, new_set):
        """
        Run adaptation step
        
        :param covariance: Covariance methods and variables
        :type covariance: :class:`~pymcmcstat.CovarianceProcedures.CovarianceProcedures`
        :param options: Options for MCMC simulation
        :type options: :class:`~pymcmcstat.SimulationOptions.SimulationOptions`
        :param isimu: simulation count
        :type isimu: :py:class:`int`
        :param iiadapt: adaptation counter
        :type iiadapt: :py:class:`int`
        :param rejected: rejection statistics
        :type rejected: :py:class:`dict`
        :param chain: sampling chain
        :type chain: :class:`~numpy.ndarray`
        :param chainind: relative point in chain       
        :type chainind: :py:class:`int`
        :param u: Latest random sample point
        :type u: :class:`~numpy.ndarray`
        :param npar: Number of parameters being sampled
        :type npar: :py:class:`int`
        :param new_set: Features of newest parameter set
        :type new_set: :class:`~pymcmcstat.ParameterSet.ParameterSet`
        
        :returns: Updated covariance object
        :rtype: :class:`~pymcmcstat.CovarianceProcedures.CovarianceProcedures`
        
        """
        # unpack input arguments
        burnintime = options.burnintime
        burnin_scale = options.burnin_scale
        ntry = options.ntry
        drscale = options.drscale
        alphatarget = options.alphatarget
        etaparam = options.alphatarget
        qcov_adjust = options.qcov_adjust
        doram = options.doram
        
        # covariance 
        last_index_since_adaptation = covariance._last_index_since_adaptation
        R = covariance._R
        oldcovchain = covariance._covchain
        oldmeanchain = covariance._meanchain
        oldwsum = covariance._wsum
        no_adapt_index = covariance._no_adapt_index
        
        qcov_scale = covariance._qcov_scale
        qcov = covariance._qcov
        
        if isimu < burnintime:
            # during burnin no adaptation, just scaling down
            if rejected['in_adaptation_interval']*(iiadapt**(-1)) > 0.95:
                self.__message(options.verbosity, 2, str(' (burnin/down) {3.2f}'.format(
                        rejected['in_adaptation_interval']*(iiadapt**(-1))*100)))
                R = R*(burnin_scale**(-1))
            elif rejected['in_adaptation_interval']*(iiadapt**(-1)) < 0.05:
                self.__message(options.verbosity, 2, str(' (burnin/up) {3.2f}'.format(
                        rejected['in_adaptation_interval']*(iiadapt**(-1))*100)))
                R = R*burnin_scale
                    
        else:
            self.__message(options.verbosity, 2, str('i:{} adapting ({}, {}, {})'.format(
                    isimu, rejected['total']*(isimu**(-1))*100, rejected['in_adaptation_interval']*(iiadapt**(-1))*100, 
                    rejected['outside_bounds']*(isimu**(-1))*100)))
    
            # UPDATE COVARIANCE MATRIX - CHOLESKY
            covchain, meanchain, wsum = self.covupd(
                    chain[last_index_since_adaptation:chainind,:], np.ones(1), oldcovchain, oldmeanchain, oldwsum)
                    
            last_index_since_adaptation = isimu
                    
            # ram
            if doram:
                uu = u*(np.linalg.norm(u)**(-1))
                eta = (isimu**(etaparam))**(-1)
                ram = np.eye(npar) + eta*(min(1, new_set.alpha) - alphatarget)*(
                        np.dot(uu.transpose(), uu))
                upcov = np.dot(np.dot(R.transpose(),ram),R)
            else:
                upcov = covchain
                upcov[no_adapt_index, :] = qcov[no_adapt_index,:]
                upcov[:,no_adapt_index] = qcov[:,no_adapt_index]

            # check if singular covariance matrix
            pos_def, pRa = self.is_semi_pos_def_chol(upcov)
            if pos_def == 1: # not singular!
                Ra = pRa # np.linalg.cholesky(upcov)
                R = Ra*qcov_scale
                        
            else: # singular covariance matrix
                # try to blow it up
                tmp = upcov + np.eye(npar)*qcov_adjust
                pos_def_adjust, pRa = self.__is_semi_pos_def_chol(tmp)
                if pos_def_adjust == 1: # not singular!
                    Ra = pRa
                    self.__message(options.verbosity, 1, 'adjusted covariance matrix')
                    # scale R
                    R = Ra*qcov_scale
                else: # still singular...
                    errstr = str('convariance matrix singular, no adaptation')
                    self.__message(options.verbosity, 0, '{} {}'.format(errstr, rejected['in_adaptation_interval']*(iiadapt**(-1))*100))
        
            # update dram covariance matrix
            RDR = None
            invR = None
            if ntry > 1: # delayed rejection
                RDR = []
                invR = []
                RDR.append(R)
                invR.append(np.linalg.solve(RDR[0], np.eye(npar)))
                for ii in range(1,ntry):
                    RDR.append(RDR[ii-1]*((drscale[min(ii,len(drscale)) - 1])**(-1))) 
                    invR.append(invR[ii-1]*(drscale[min(ii,len(drscale)) - 1]))
          
        
        
        covariance._update_covariance_from_adaptation(R, covchain, meanchain, wsum, 
                                          last_index_since_adaptation, iiadapt)

        covariance._update_covariance_for_delayed_rejection_from_adaptation(RDR = RDR, invR = invR)
        
        return covariance
    
    def covupd(self, x, w, oldcov, oldmean, oldwsum, oldR = None):
        """
        Update covariance chain, local mean, local sum
        
        :param x: chain segment
        :type x: :class:`~numpy.ndarray`
        :param w: weights
        :type w: :class:`~numpy.ndarray`
        :param oldcov: previous covariance matrix
        :type oldcov: :class:`~numpy.ndarray` or None
        :param oldmean: previous mean chain value
        :type oldmean: :class:`~numpy.ndarray`
        :param oldwsum: previous weighted sum
        :type oldwsum: :class:`~numpy.ndarray`
        :param oldR: previous cholesky decomposition of covariance matrix
        :type oldR: :class:`~numpy.ndarray`
        :returns: Updated covariance, mean, and sum
        """
        n, p = x.shape
        
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
            wsum = sum(w)
            xmean = np.zeros(p)
            xcov = np.zeros([p,p])
            for ii in range(p):
                xmean[ii] = sum(x[:,ii]*w)*(wsum**(-1))
            if wsum > 1:
                for ii in range(0,p):
                    for jj in range(0,ii+1):
                        term1 = x[:,ii] - xmean[ii]
                        term2 = x[:,jj] - xmean[jj]
                        xcov[ii,jj] = np.dot(term1.transpose(), ((term2)*w)*((wsum-1)**(-1)))
                        if ii != jj:
                            xcov[jj,ii] = xcov[ii,jj]
                            
        else:
            for ii in range(0,n):
                xi = x[ii,:]
                wsum = w[ii]
                xmean = oldmean + wsum*((wsum+oldwsum)**(-1.0))*(xi - oldmean)
                
                    
                if R is not None:
                    print('R = \n{}\n'.format(R))
                    print('np.sqrt((oldwsum-1)*((wsum+oldwsum-1)**(-1))) = {}\n'.format(np.sqrt((oldwsum-1)*((wsum+oldwsum-1)**(-1)))))
                
                    R = self.cholupdate(np.sqrt((oldwsum-1)*((wsum+oldwsum-1)**(-1)))*R, np.dot((xi - oldmean).transpose(), 
                                          np.sqrt(((wsum*oldwsum)*((wsum+oldwsum-1)**(-1))*((wsum+oldwsum)**(-1))))))
            
                
                xcov = (((oldwsum-1)*((wsum + oldwsum - 1)**(-1)))*oldcov 
                        + (wsum*oldwsum*((wsum+oldwsum-1)**(-1)))*((wsum 
                               + oldwsum)**(-1))*(np.dot((xi-oldmean).reshape(p,1),(xi-oldmean).reshape(1,p))))
            
                wsum = wsum + oldwsum
                oldcov = xcov
                oldmean = xmean
                oldwsum = wsum
       
        return xcov, xmean, wsum
    
    # Cholesky Update
    def cholupdate(self, R, x):
        """
        Update Cholesky decomposition
        
        :param R: weighted Cholesky decomposition
        :type R: :class:`~numpy.ndarray`
        :param x: weighted sum based on local chain update
        :returns: Updated Cholesky decomposition
        :rtype: :class:`~numpy.ndarray`
        """
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
    
    def is_semi_pos_def_chol(self, x):
        """
        Check if matrix is semi-positive definite using Cholesky Decomposition
        
        :param x: Covariance matrix
        :type x: :class:`~numpy.ndarray`
        :returns: Boolean, Cholesky Decomposition (Upper Triangular Form) or None
        """
        
        c = None
        try:
            c = np.linalg.cholesky(x)
            return True, c.transpose()
        except np.linalg.linalg.LinAlgError:
            return False, c
        
    def __message(self, verbosity, level, printthis):
        printed = False
        if verbosity >= level:
            print(printthis)
            printed = True
        return printed