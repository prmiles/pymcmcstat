#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 08:02:45 2018

@author: prmiles
"""

from pymcmcstat.structures.ResultsStructure import ResultsStructure
from pymcmcstat.settings.SimulationOptions import SimulationOptions
from pymcmcstat.settings.ModelSettings import ModelSettings
from pymcmcstat.samplers.DelayedRejection import DelayedRejection
from pymcmcstat.MCMC import MCMC
import unittest
import numpy as np
import os

# define test model function
def modelfun(xdata, theta):
    m = theta[0]
    b = theta[1]
    nrow = xdata.shape[0]
    y = np.zeros([nrow,1])
    y[:,0] = m*xdata.reshape(nrow,) + b
    return y

def ssfun(theta, data, local = None):
    xdata = data.xdata[0]
    ydata = data.ydata[0]
    # eval model
    ymodel = modelfun(xdata, theta)
    # calc sos
    ss = sum((ymodel[:,0] - ydata[:,0])**2)
    return ss

def setup_mcmc():
    # Initialize MCMC object
    mcstat = MCMC()
    # Add data
    nds = 100
    x = np.linspace(2, 3, num=nds)
    y = 2.*x + 3. + 0.1*np.random.standard_normal(x.shape)
    mcstat.data.add_data_set(x, y)

    mcstat.simulation_options.define_simulation_options(nsimu = int(2.0e2), updatesigma = 1, method = 'dram', verbosity = 0)

    # update model settings
    mcstat.model_settings.define_model_settings(sos_function = ssfun)

    mcstat.parameters.add_model_parameter(name = 'm', theta0 = 2., minimum = -10, maximum = np.inf, sample = 1)
    mcstat.parameters.add_model_parameter(name = 'b', theta0 = -5., minimum = -10, maximum = 100, sample = 1)
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

def generate_temp_file():
    tmpfile = 'temp0.json'
    count = 0
    flag = True
    while flag is True:
        if os.path.isfile(str('{}'.format(tmpfile))):
            count += 1
            tmpfile = str('{}{}.json'.format('temp',count))
        else:
            flag = False
    return tmpfile
    
# -------------------
class SaveLoadJSONObject(unittest.TestCase):
    
    def test_dump_to_json_file(self):
        RS = ResultsStructure()
        tmpfile = generate_temp_file()
        results = {'model': 4, 'data': np.random.random_sample(size = (1000,2))}
        RS.save_json_object(results = results, filename = tmpfile)
        retres = RS.load_json_object(filename = tmpfile)
        self.assertEqual(retres['model'], results['model'], msg = 'Model set to 4')
        self.assertTrue(np.array_equal(retres['data'], results['data']), msg = 'Arrays should be equal')
        os.remove(tmpfile)

# -------------------
class DetermineFilename(unittest.TestCase):
    def test_resfilename_is_none(self):
        model, options, parameters, data, covariance, rejected, chain, s2chain, sschain = setup_mcmc()
        RS = ResultsStructure()
        RS.add_options(options = options)
        RS.results['simulation_options']['results_filename'] = None
        filename = RS.determine_filename(options = RS.results['simulation_options'])
        self.assertEqual(filename, str('{}{}{}'.format(RS.results['simulation_options']['datestr'],'_','mcmc_simulation.json')), msg = 'Filename matches')
        
    def test_resfilename_is_not_none(self):
        model, options, parameters, data, covariance, rejected, chain, s2chain, sschain = setup_mcmc()
        RS = ResultsStructure()
        RS.add_options(options = options)
        RS.results['simulation_options']['results_filename'] = 'test'
        filename = RS.determine_filename(options = RS.results['simulation_options'])
        self.assertEqual(filename, 'test', msg = 'Filename matches')
        
# -------------------
class AddBasic(unittest.TestCase):
    def test_addbasic_false(self):
        RS = ResultsStructure()
        self.assertFalse(RS.basic, msg = 'basic features not added to result structure')
        
    def test_addbasic_true(self):
        model, options, parameters, data, covariance, rejected, chain, s2chain, sschain = setup_mcmc()
        RS = ResultsStructure()
        RS.add_basic(nsimu = options.nsimu, covariance=covariance, parameters=parameters, rejected=rejected, simutime = 0.001, theta = chain[-1,:])
        self.assertTrue(RS.basic, msg = 'basic features added to result structure')
        self.assertTrue(np.array_equal(RS.results['theta'], np.array([0,0])), msg = 'Last elements of chain are zero')
        
    def test_addbasic_rejection(self):
        model, options, parameters, data, covariance, rejected, chain, s2chain, sschain = setup_mcmc()
        RS = ResultsStructure()
        RS.add_basic(nsimu = options.nsimu, covariance=covariance, parameters=parameters, rejected=rejected, simutime = 0.001, theta = chain[-1,:])
        self.assertEqual(RS.results['total_rejected'], 10*(options.nsimu**(-1)), msg = 'rejection reported as fraction of nsimu')
        self.assertEqual(RS.results['rejected_outside_bounds'], 2*(options.nsimu**(-1)), msg = 'rejection reported as fraction of nsimu')
        
    def test_addbasic_covariance(self):
        model, options, parameters, data, covariance, rejected, chain, s2chain, sschain = setup_mcmc()
        covariance._R[0,0] = 1.1
        covariance._R[0,1] = 2.3
        RS = ResultsStructure()
        RS.add_basic(nsimu = options.nsimu, covariance=covariance, parameters=parameters, rejected=rejected, simutime = 0.001, theta = chain[-1,:])
        self.assertTrue(np.array_equal(RS.results['R'], covariance._R), msg = 'Cholesky matches')
        self.assertTrue(np.array_equal(RS.results['qcov'], np.dot(covariance._R.transpose(),covariance._R)), msg = 'Covariance matches')

# -------------------
class AddDRAM(unittest.TestCase):
    def test_addbasic_false(self):
        model, options, parameters, data, covariance, rejected, chain, s2chain, sschain = setup_mcmc()
        DR = DelayedRejection()
        DR._initialize_dr_metrics(options)
        RS = ResultsStructure()
        self.assertFalse(RS.add_dram(drscale = options.drscale, RDR=covariance._RDR, total_rejected=rejected['total'], drsettings = DR), msg = 'basic features not added to result structure')
        
    def test_addbasic_true(self):
        model, options, parameters, data, covariance, rejected, chain, s2chain, sschain = setup_mcmc()
        covariance._RDR = np.random.random_sample(size = (2,2))
        DR = DelayedRejection()
        DR._initialize_dr_metrics(options)
        DR.dr_step_counter = 12000
        RS = ResultsStructure()
        RS.add_basic(nsimu = options.nsimu, covariance=covariance, parameters=parameters, rejected=rejected, simutime = 0.001, theta = chain[-1,:])
        self.assertTrue(RS.add_dram(drscale = options.drscale, RDR=covariance._RDR, total_rejected=rejected['total'], drsettings = DR), msg = 'basic features added to result structure')
        self.assertTrue(np.array_equal(RS.results['RDR'], covariance._RDR), msg = 'RDR matches')
        self.assertEqual(RS.results['alpha_count'], DR.dr_step_counter, msg = 'Alpha count matches dr step counter')
        
# -------------------
class AddUpdateSigma(unittest.TestCase):
    
    def test_updatesigma_true(self):
        updatesigma = True
        sigma2 = np.ones([2,1])
        S20 = np.ones([2,1])
        N0 = np.ones([2,1])
        RS = ResultsStructure()
        RS.add_updatesigma(updatesigma = updatesigma, sigma2 = sigma2, S20 = S20, N0 = N0)
        self.assertTrue(np.isnan(RS.results['sigma2']), msg = 'sigma2 -> np.nan')
        self.assertTrue(np.array_equal(RS.results['S20'], S20), msg = str('S20 -> {}'.format(S20)))
        self.assertTrue(np.array_equal(RS.results['N0'], N0), msg = str('N0 -> {}'.format(N0)))
        
        
    def test_updatesigma_false(self):
        updatesigma = False
        sigma2 = np.ones([2,1])
        S20 = np.ones([2,1])
        N0 = np.ones([2,1])
        RS = ResultsStructure()
        RS.add_updatesigma(updatesigma = updatesigma, sigma2 = sigma2, S20 = S20, N0 = N0)
        self.assertTrue(np.array_equal(RS.results['sigma2'], sigma2), msg = str('sigma2 -> {}'.format(sigma2)))
        self.assertTrue(np.isnan(RS.results['S20']), msg = 'S20 -> np.nan')
        self.assertTrue(np.isnan(RS.results['N0']), msg = 'N0 -> np.nan')
        
# -------------------
class AddArrays(unittest.TestCase):
    
    def test_add_rndnumseq(self):
        rnd = np.random.random_sample(size = (3,2))
        key = 'rndseq'
        RS = ResultsStructure()
        RS.add_random_number_sequence(rndseq = rnd)
        self.assertTrue(np.array_equal(RS.results[key], rnd),
                        msg = str('Expect equal arrays: {} != {}'.format(RS.results[key], rnd)))
        
    def test_add_chain(self):
        rnd = np.random.random_sample(size = (3,2))
        key = 'chain'
        RS = ResultsStructure()
        RS.add_chain(chain = rnd)
        self.assertTrue(np.array_equal(RS.results[key], rnd),
                        msg = str('Expect equal arrays: {} != {}'.format(RS.results[key], rnd)))
        
    def test_add_sschain(self):
        rnd = np.random.random_sample(size = (3,2))
        key = 'sschain'
        RS = ResultsStructure()
        RS.add_sschain(sschain = rnd)
        self.assertTrue(np.array_equal(RS.results[key], rnd),
                        msg = str('Expect equal arrays: {} != {}'.format(RS.results[key], rnd)))
        
    def test_add_s2chain(self):
        rnd = np.random.random_sample(size = (3,2))
        key = 's2chain'
        RS = ResultsStructure()
        RS.add_s2chain(s2chain = rnd)
        self.assertTrue(np.array_equal(RS.results[key], rnd),
                        msg = str('Expect equal arrays: {} != {}'.format(RS.results[key], rnd)))
    
# -------------------
class AddOptions(unittest.TestCase):
    
    def test_key_removal(self):
        options = SimulationOptions()
        options.define_simulation_options(doram = 3, nsimu = 500)
        RS = ResultsStructure()
        RS.add_options(options = options)
        self.assertEqual(RS.results['simulation_options']['nsimu'], options.nsimu, msg = str('nsimu = {}'.format(options.nsimu)))
        self.assertFalse('doram' in RS.results['simulation_options'].items(), msg = 'doram option should not be saved')
        
# -------------------
class AddModel(unittest.TestCase):
    
    def test_key_removal(self):
        model = ModelSettings()
        model.define_model_settings(N = 100, sos_function = 'hello world')
        RS = ResultsStructure()
        RS.add_model(model = model)
        self.assertEqual(RS.results['model_settings']['N'], model.N, msg = str('N = {}'.format(model.N)))
        self.assertFalse('sos_function' in RS.results['model_settings'].items(), msg = 'sos_function setting should not be saved')
        
# -------------------
class AddTimeStats(unittest.TestCase):
    
    def test_add_time_stats(self):
        key = 'time [mh, dr, am]'
        mtime = 0.01
        drtime = 0.10
        adtime = 0.05
        RS = ResultsStructure()
        RS.add_time_stats(mtime = mtime, drtime = drtime, adtime = adtime)
        self.assertEqual(RS.results[key], [mtime, drtime, adtime], msg = 'Lists should be equal.')
        
# -------------------
class AddPrior(unittest.TestCase):
    
    def test_add_prior(self):
        RS = ResultsStructure()
        mu = 0.0
        sigma = 1.0
        priorfun = 'priorfun'
        priortype = 1
        priorpars = 3
        RS.add_prior(mu = mu, sig = sigma, priorfun = priorfun, priorpars = priorpars, priortype = priortype)
        self.assertEqual(RS.results['prior'], [mu, sigma], msg = str('Expected [{},{}]'.format(mu, sigma)))
        self.assertEqual(RS.results['priorfun'], priorfun, msg = str('Expected {}'.format(priorfun)))
        self.assertEqual(RS.results['priortype'], priortype, msg = str('Expected {}'.format(priortype)))
        self.assertEqual(RS.results['priorpars'], priorpars, msg = str('Expected {}'.format(priorpars)))