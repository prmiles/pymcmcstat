#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 15:35:08 2018

@author: prmiles
"""
import numpy as np
from pymcmcstat.MCMC import MCMC
from pymcmcstat.settings.DataStructure import DataStructure
from pymcmcstat.structures.ParameterSet import ParameterSet
import os

def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

# define test model function
def modelfun(xdata, theta):
    m = theta[0]
    b = theta[1]
    nrow = xdata.shape[0]
    y = np.zeros([nrow,1])
    y[:,0] = m*xdata.reshape(nrow,) + b
    return y

def ssfun(theta, data, local=None):
    xdata = data.xdata[0]
    ydata = data.ydata[0]
    # eval model
    ymodel = modelfun(xdata, theta)
    # calc sos
    ss = sum((ymodel[:,0] - ydata[:,0])**2)
    return ss

def custom_ssfun(theta, data, custom=None):
    return custom

def basic_mcmc():
    # Initialize MCMC object
    mcstat = MCMC()
    # Add data
    nds = 100
    x = np.linspace(2, 3, num=nds)
    y = 2.*x + 3. + 0.1*np.random.standard_normal(x.shape)
    mcstat.data.add_data_set(x, y)

    mcstat.simulation_options.define_simulation_options(nsimu = int(5.0e3), updatesigma = 1, method = 'dram', verbosity = 0)

    # update model settings
    mcstat.model_settings.define_model_settings(sos_function = ssfun)

    mcstat.parameters.add_model_parameter(name = 'm', theta0 = 2., minimum = -10, maximum = np.inf, sample = 1)
    mcstat.parameters.add_model_parameter(name = 'b', theta0 = -5., minimum = -10, maximum = 100, sample = 0)
    mcstat.parameters.add_model_parameter(name = 'b2', theta0 = -5., minimum = -10, maximum = 100, sample = 1)
    return mcstat

def setup_initialize_chains(CL, updatesigma = True, nsos = 1):
    mcstat = setup_mcmc_case_cp()
    mcstat.simulation_options.updatesigma = updatesigma
    mcstat.model_settings.nsos = nsos
    mcstat._MCMC__old_set = ParameterSet(theta = CL['theta'], ss = CL['ss'], prior = CL['prior'], sigma2 = CL['sigma2'])
    mcstat._MCMC__chain_index = mcstat.simulation_options.nsimu - 1
    mcstat._MCMC__initialize_chains(chainind = 0, nsimu = mcstat.simulation_options.nsimu, npar = mcstat.parameters.npar, nsos = mcstat.model_settings.nsos, updatesigma = mcstat.simulation_options.updatesigma, sigma2 = mcstat.model_settings.sigma2)
    return mcstat
        
def setup_case():
    mcstat = basic_mcmc()
    mcstat._MCMC__chain = np.random.random_sample(size = (100,2))
    mcstat._MCMC__sschain = np.random.random_sample(size = (100,2))
    mcstat._MCMC__s2chain = np.random.random_sample(size = (100,2))
    mcstat._covariance._R = np.array([[0.5, 0.2],[0., 0.3]])
    
    mcstat._MCMC__chains = []
    mcstat._MCMC__chains.append(dict(file = mcstat.simulation_options.chainfile, mtx = mcstat._MCMC__chain))
    mcstat._MCMC__chains.append(dict(file = mcstat.simulation_options.sschainfile, mtx = mcstat._MCMC__sschain))
    mcstat._MCMC__chains.append(dict(file = mcstat.simulation_options.s2chainfile, mtx = mcstat._MCMC__s2chain))
        
    return mcstat

def setup_mcmc():
    mcstat = basic_mcmc()
    mcstat._initialize_simulation()
    # extract components
    model = mcstat.model_settings
    options = mcstat.simulation_options
    parameters = mcstat.parameters
    data = mcstat.data
    return model, options, parameters, data

def setup_mcmc_case_mh():
    mcstat = basic_mcmc()
    mcstat._initialize_simulation()
    
    # extract components
    sos_object = mcstat._MCMC__sos_object
    prior_object = mcstat._MCMC__prior_object
    parameters = mcstat.parameters
    return sos_object, prior_object, parameters

def setup_mcmc_case_dr():
    mcstat = basic_mcmc()
    mcstat._initialize_simulation()
    
    # extract components
    model = mcstat.model_settings
    options = mcstat.simulation_options
    parameters = mcstat.parameters
    data = mcstat.data
    covariance = mcstat._covariance
    rejected = {'total': 10, 'outside_bounds': 2}
    chain = np.zeros([options.nsimu, 2])
    s2chain = np.zeros([options.nsimu, 1])
    sschain = np.zeros([options.nsimu, 1])
    return model, options, parameters, data, covariance, rejected, chain, s2chain, sschain

# define test model function
def predmodelfun(data, theta):
    m = theta[0]
    b = theta[1]
    nrow = data.xdata[0].shape[0]
    y = np.zeros([nrow,1])
    y[:,0] = m*data.xdata[0].reshape(nrow,) + b
    return y

def basic_data_structure():
    DS = DataStructure()
    x = np.random.random_sample(size = (100, 1))
    y = np.random.random_sample(size = (100, 1))
    DS.add_data_set(x = x, y = y)
    return DS

def non_basic_data_structure():
    DS = basic_data_structure()
    x = np.random.random_sample(size = (100, 1))
    y = np.random.random_sample(size = (100, 2))
    DS.add_data_set(x = x, y = y)
    return DS

def setup_pseudo_results():
    results = {
            'chain': np.random.random_sample(size = (100,2)),
            's2chain': np.random.random_sample(size = (100,1)),
            'sschain': np.random.random_sample(size = (100,1)),
            'parind': np.array([[0, 1]]),
            'local': np.array([[0, 0]]),
            'model_settings': {'nbatch': np.random.random_sample(size = (100,1))},
            'theta': np.random.random_sample(size = (2,)),
            'sstype': np.random.random_sample(size = (1,1)),
            }
    return results

def setup_pseudo_ci():
    ci = []
    ci1 = []
    ci1.append([np.random.random_sample(size = (100,)),np.random.random_sample(size = (100,)),np.random.random_sample(size = (100,))])
    ci.append(ci1)
    return ci

def setup_mcmc_case_cp(initialize = True):
    mcstat = basic_mcmc()
    if initialize:
        mcstat._initialize_simulation()
    
    return mcstat

def generate_temp_folder():
    tmpfolder = 'temp0'
    count = 0
    flag = True
    while flag is True:
        if os.path.isdir(str('{}'.format(tmpfolder))):
            count += 1
            tmpfolder = str('{}{}'.format('temp',count))
        else:
            flag = False
    return tmpfolder

def generate_temp_file(extension = 'h5'):
    tmpfile = str('temp0.{}'.format(extension))
    count = 0
    flag = True
    while flag is True:
        if os.path.isfile(str('{}'.format(tmpfile))):
            count += 1
            tmpfile = str('{}{}.{}'.format('temp',count,extension))
        else:
            flag = False
    return tmpfile

class CustomSampler:
    def __init__(self, nsimu):
        self.name = 'Gibbs'
        self.nsimu = nsimu
        
    def setup(self):
        self.status = 'setup'
        self.tau = np.zeros([self.nsimu, 1])
        self.counter = 0
        self.tau[self.counter,0] = np.random.gamma(0.1, 0.1)
        self.chains = []
        self.chains.append(dict(file = 'tauchain', mtx = self.tau))
        return self.tau[self.counter, 0]
    
    def update(self, **kwargs):
        self.counter += 1
        self.tau[self.counter,0] = np.random.gamma(0.1, 0.1)
        self.status = 'updating'
        self.tau[self.counter, 0]
        return self.tau[self.counter, 0]
