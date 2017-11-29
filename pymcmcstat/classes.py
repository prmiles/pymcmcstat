#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 12:31:03 2017

@author: prmiles

Common classes for pymcmc
"""

# import required packages
import numpy as np
from datetime import datetime

class DataStructure:
    """Simple data structure"""        
    def __init__(self):
        self.xdata = [] # initialize list
        self.ydata = [] # initialize list
        self.n = [] # initialize list - number of data points
        self.weight = [] # initialize list - weight of data set
        self.udobj = [] # user defined object
        
    def add_data_set(self, x, y, n = None, weight = 1, udobj = 0):
        # append new data set
        self.xdata.append(x)
        self.ydata.append(y)
        
        if n is None:
            if isinstance(y, list): # y is a list
                self.n.append(len(y))
            else:
                self.n.append(y.size) # assume y is a numpy array
        
        self.weight.append(weight)
        # add user defined objects option
        self.udobj.append(udobj)
        
class Options:
    """Simulator Options"""
    def __init__(self, nsimu=10000, adaptint = None, ntry = None, method='dram',
                 printint=np.nan, adaptend = 0, lastadapt = 0, burnintime = 0,
                 waitbar = 1, debug = 0, qcov = None, updatesigma = 0, 
                 noadaptind = [], stats = 0, drscale = np.array([5, 4, 3], dtype = float),
                 adascale = None, savesize = 0, maxmem = 0, chainfile = None,
                 s2chainfile = None, sschainfile = None, savedir = None, skip = 1,
                 label = None, RDR = None, verbosity = 1, maxiter = None, 
                 priorupdatestart = 0, qcov_adjust = 1e-8, burnin_scale = 10, 
                 alphatarget = 0.234, etaparam = 0.7, initqcovn = None,
                 doram = None, rndseq = None):
        
        method_dictionary = {
            'mh': {'adaptint': 0, 'ntry': 1, 'doram': 0, 'adascale': adascale}, 
            'am': {'adaptint': 100, 'ntry': 1, 'doram': 0, 'adascale': adascale},
            'dr': {'adaptint': 0, 'ntry': 2, 'doram': 0, 'adascale': adascale},
            'dram': {'adaptint': 100, 'ntry': 2, 'doram': 0, 'adascale': adascale},
            'ram': {'adaptint': 1, 'ntry': 1, 'doram': 1, 'adascale': 1},
            }
        
        # define items from dictionary
        if adaptint is None:
            self.adaptint = method_dictionary[method]['adaptint']  # update interval for adaptation
        elif method == 'mh' or method == 'dr':
            self.adaptint = method_dictionary[method]['adaptint']  # no adaptation - enforce!
        else:
            self.adaptint = adaptint
        
        if ntry is None:
            self.ntry = method_dictionary[method]['ntry']
        else:
            self.ntry = ntry
            
        if adascale is None:
            self.adascale = method_dictionary[method]['adascale']  # qcov_scale
        else:
            self.adascale = adascale
            
        if doram is None:
            self.doram = method_dictionary[method]['doram']
        else:
            self.doram = doram
        
        
        self.nsimu = nsimu  # length of chain to simulate
        self.method = method
        
        self.printint = printint  # print interval
        self.adaptend = adaptend  # last adapt
        self.lastadapt = lastadapt # last adapt
        self.burnintime = burnintime
        self.waitbar = waitbar # use waitbar
        self.debug = debug  # show some debug information
        self.qcov = qcov  # proposal covariance
        self.initqcovn = initqcovn  # proposal covariance weight in update
        self.updatesigma = updatesigma  # 
        self.noadaptind = noadaptind  # do not adapt these indices
        self.priorupdatestart = priorupdatestart
        self.qcov_adjust = qcov_adjust  # eps adjustment
        self.burnin_scale = burnin_scale
        self.alphatarget = alphatarget  # acceptance ratio target
        self.etaparam = etaparam  #
        self.stats = stats  # convergence statistics
        self.drscale = drscale

        self.skip = skip
        
        if label is None:
            self.label = str('MCMC run at {}'.format(datetime.now().strftime("%Y%m%d_%H%M%S")))
        else:
            self.label = label
            
        self.RDR = RDR
        self.verbosity = verbosity # amount of information to print
        self.maxiter = maxiter
        self.savesize = savesize
        self.maxmem = maxmem
        self.chainfile = chainfile
        self.s2chainfile = s2chainfile
        self.sschainfile = sschainfile
        self.savedir = savedir
        
        self.rndseq = rndseq # define random number sequence for testing
        
class Model:
    def __init__(self, ssfun = None, priorfun = None, priortype = 1, 
                 priorupdatefun = None, priorpars = None, modelfun = None, 
                 sigma2 = None, N = None, 
                 S20 = np.nan, N0 = None, nbatch = None):
    
        self.ssfun = ssfun
        self.priorfun = priorfun
        self.priortype = priortype
        self.priorupdatefun = priorupdatefun
        self.priorpars = priorpars
        self.modelfun = modelfun
        
        if sigma2 is None:
            self.sigma2 = sigma2
        else:
            self.sigma2 = np.array(sigma2)
        
        if N is None:
            self.N = N
        else:
            self.N = np.array(N)
            
        if N0 is None:
            self.N0 = N0
        else:
            self.N0 = np.array(N0)
        
        
        if nbatch is None:
            self.nbatch = len(N0)
        else:
            self.nbatch = np.array(nbatch)
            
        self.S20 = np.array([S20])
        
        
class Parameters:
    def __init__(self):
#        self.parameters = [] # initialize list
        self.parameters = [] # initialize list
        self.label = 'MCMC model parameters'
        
    def add_parameter(self, name, theta0, minimum = -np.inf,
                      maximum = np.inf, mu = np.zeros([1]), sig = np.inf,
                      sample = None, local = 0):
        
        # append dictionary element
        self.parameters.append({'name': name, 'theta0': theta0, 'minimum': minimum,
                                'maximum': maximum, 'mu': mu, 'sig': sig,
                                'sample': sample, 'local': local})
#        self.parameters.append([name, theta0, minimum, maximum, mu, sig, 
#                          sample, local])
        
#        parameter = [name, theta0, minimum, maximum, mu, sig, 
#                          sample, local]
        
#        self.parameters.append(parameter)
        
class Parset:
    def __init__(self, theta = None, ss= None, prior = None, sigma2 = None, alpha = None):
        self.theta = theta
        self.ss = ss
        self.prior = prior
        self.sigma2 = sigma2
        self.alpha = alpha
        
class PriorObject:
    def __init__(self, priorfun = None, mu = None, sigma = None):

        self.mu = mu
        self.sigma = sigma
        
        # Setup prior function and evaluate
        if priorfun is None:
            priorfun = self.default_priorfun
        
        self.priorfun = priorfun # function handle
            
    def default_priorfun(self, theta, mu, sigma):
        # consider converting everything to numpy array - should allow for optimized performance
        n = len(theta)
        pf = np.zeros(1)
        for ii in range(n):
            pf = pf + ((theta[ii]-mu[ii])*(sigma[ii]**(-1)))**2
        
#        pf = np.sum(((theta - mu)*(sigma**(-1)))**(2))
#        print('pf = {}, pftest = {}'.format(pf, pftest))
#        sys.exit()
        return pf
        
    def evaluate_prior(self, theta):
        return self.priorfun(theta, self.mu, self.sigma)
    
class SSObject:
    def __init__(self, ssfun = None, ssstyle = None, modelfun = None, 
                 parind = None, local = None, data = None, nbatch = None):
        self.ssfun = ssfun
        self.ssstyle = ssstyle
        self.modelfun = modelfun
        self.parind = parind
        self.local = local
        self.data = data
        self.nbatch = nbatch
        
    def evaluate_sos(self, theta):
        # evaluate sum-of-squares function
        if self.ssstyle == 1:
            ss = self.ssfun(theta, self.data)
        elif self.ssstyle == 4:
            ss = self.mcmcssfun(theta, self.data, self.local, self.modelfun)
        else:
            ss = self.ssfun(theta, self.data, self.local)
        
        # make sure sos is a numpy array
        if not isinstance(ss, np.ndarray):
            ss = np.array([ss])
            
        return ss
                     
    def mcmcssfun(self, theta, data, local, modelfun):
        # initialize
        ss = np.zeros(self.nbatch)

        for ibatch in range(self.nbatch):
            for iset in range(len(data[ibatch].n)):
                xdata = data[ibatch].xdata[iset]
                ydata = data[ibatch].ydata[iset]
                weight = data[ibatch].weight[iset]
            
                # evaluate model
                ymodel = modelfun(xdata, theta)
    
                # calculate sum-of-squares error    
                ss[ibatch] += sum(weight*(ydata-ymodel)**2)
        
        return ss

        
class Results:
    def __init__(self):
        self.results = {} # initialize empty dictionary
        
    def add_basic(self, nsimu, rej, rejl, R, covchain, meanchain, names, 
                  lowerlims, upperlims, theta, parind, local, simutime, qcovorig):
    
        self.results['theta'] = theta
        
        self.results['parind'] = parind
        self.results['local'] = local
        
        self.results['rejected'] = rej*(nsimu**(-1)) # total rejected
        self.results['ulrejected'] = rejl*(nsimu**(-1)) # rejected due to sampling outside limits
        self.results['R'] = R
        self.results['qcov'] = np.dot(R.transpose(),R)
        self.results['cov'] = covchain
        self.results['mean'] = meanchain
        self.results['names'] = names
        self.results['limits'] = [lowerlims[parind[:]], upperlims[parind[:]]]
        
        self.results['nsimu'] = nsimu
        self.results['simutime'] = simutime
        self.results['qcovorig'] = qcovorig
    
    def add_updatesigma(self, updatesigma, sigma2, S20, N0):
        self.results['updatesigma'] = updatesigma
        if updatesigma:
            self.results['sigma2'] = np.nan
            self.results['S20'] = S20
            self.results['N0'] = N0
        else:
            self.results['sigma2'] = sigma2
            self.results['S20'] = np.nan
            self.results['N0'] = np.nan
    
    def add_dram(self, dodram, drscale, iacce, alpha_count, RDR, nsimu, rej):
        if dodram == 1:
            self.results['drscale'] = drscale
            iacce[0] = nsimu - rej - sum(iacce[1:])
            # 1 - number accepted without DR, 2 - number accepted via DR try 1, 
            # 3 - number accepted via DR try 2, etc.
            self.results['iacce'] = iacce 
            self.results['alpha_count'] = alpha_count
            self.results['RDR'] = RDR
    
    def add_prior(self, mu, sig, priorfun, priortype, priorpars):
        self.results['prior'] = [mu, sig]
        self.results['priorfun'] = priorfun
        self.results['priortype'] = priortype
        self.results['priorpars'] = priorpars
        
    def add_options(self, options = None):
        self.results['options'] = options
        
    def add_model(self, model = None):
        self.results['model'] = model
        
    def add_chain(self, chain = None):
        self.results['chain'] = chain
        
    def add_s2chain(self, s2chain = None):
        self.results['s2chain'] = s2chain
        
    def add_sschain(self, sschain = None):
        self.results['sschain'] = sschain
        
    def add_time_stats(self, mtime, drtime, adtime):
        self.results['time [mh, dr, am]'] = [mtime, drtime, adtime]
        
    def add_random_number_sequence(self, rndseq):
        self.results['rndseq'] = rndseq
    