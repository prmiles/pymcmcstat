#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 10:10:43 2018

@author: prmiles
"""

from pymcmcstat.chain import ChainProcessing as CP
from pymcmcstat.MCMC import MCMC
import unittest
from mock import patch
import numpy as np
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

def ssfun(theta, data, local = None):
    xdata = data.xdata[0]
    ydata = data.ydata[0]
    # eval model
    ymodel = modelfun(xdata, theta)
    # calc sos
    ss = sum((ymodel[:,0] - ydata[:,0])**2)
    return ss

def setup_mcmc(initialize = True):
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

# --------------------------
class CreatePathWithExtensionforAllLogs(unittest.TestCase):
    
    def test_cpwe_creation_with_h5(self):
        mcstat = setup_mcmc()
        mcstat.simulation_options.savedir = 'test'
        chainfile, s2chainfile, sschainfile, covchainfile = CP._create_path_with_extension_for_all_logs(options = mcstat.simulation_options, extension = 'h5')
        self.assertEqual(chainfile, 'test/chainfile.h5', msg = 'Expect matching string')
        self.assertEqual(s2chainfile, 'test/s2chainfile.h5', msg = 'Expect matching string')
        self.assertEqual(sschainfile, 'test/sschainfile.h5', msg = 'Expect matching string')
        self.assertEqual(covchainfile, 'test/covchainfile.h5', msg = 'Expect matching string')
        
    def test_cpwe_creation_with_txt(self):
        mcstat = setup_mcmc()
        mcstat.simulation_options.savedir = 'test'
        chainfile, s2chainfile, sschainfile, covchainfile = CP._create_path_with_extension_for_all_logs(options = mcstat.simulation_options, extension = 'txt')
        self.assertEqual(chainfile, 'test/chainfile.txt', msg = 'Expect matching string')
        self.assertEqual(s2chainfile, 'test/s2chainfile.txt', msg = 'Expect matching string')
        self.assertEqual(sschainfile, 'test/sschainfile.txt', msg = 'Expect matching string')
        self.assertEqual(covchainfile, 'test/covchainfile.txt', msg = 'Expect matching string')
        
# --------------------------
class CreatePathWithExtension(unittest.TestCase):
    def test_cpwe_basic_h5(self):
        file = CP._create_path_with_extension(savedir = 'test', file = 'file', extension = 'h5')
        self.assertEqual(file, 'test/file.h5', msg = 'Expect matching string')
        
    def test_cpwe_basic_txt(self):
        file = CP._create_path_with_extension(savedir = 'test', file = 'file', extension = 'txt')
        self.assertEqual(file, 'test/file.txt', msg = 'Expect matching string')
        
# --------------------------
class CreatePathWithOutExtension(unittest.TestCase):
    def test_cpwe_basic(self):
        file = CP._create_path_without_extension(savedir = 'test', file = 'file')
        self.assertEqual(file, 'test/file', msg = 'Expect matching string')

# --------------------------
class ReadInBinFile(unittest.TestCase):
    def test_readinbinaryfile_warning(self):
        out = CP.read_in_bin_file(filename = 'abcdedf0123')
        self.assertEqual(out, [], msg = 'expect empty list')
    
    def test_readingbinaryfile_array(self):
        tmpfile = generate_temp_file(extension = 'h5')
        CP._save_to_bin_file(filename = tmpfile, datasetname = 'd1', mtx = np.array([[0.1, 0.2]]))
        self.assertTrue(os.path.isfile(tmpfile), msg = 'File exists')
        out = CP.read_in_bin_file(filename = tmpfile)
        self.assertTrue(np.array_equal(out, np.array([[0.1, 0.2]])), msg = str('Expected array match: {}'.format(out)))
        os.remove(tmpfile)
        
# --------------------------
class ReadInTextFile(unittest.TestCase):
    def test_readintextfile_warning(self):
        out = CP.read_in_txt_file(filename = 'abcdedf0123')
        self.assertEqual(out, [], msg = 'expect empty list')
    
    @patch('numpy.loadtxt', return_value = np.array([0.1, 0.2]))
    def test_readingtextfile_array(self, mock_load):
        out = CP.read_in_txt_file(filename = 'abcdedf0123')
        self.assertTrue(np.array_equal(out, np.array([0.1, 0.2])), msg = str('Expected array match: {}'.format(out)))

# -------------------
class CheckDirectory(unittest.TestCase):
    def test_check_directory(self):
        tmpfolder = generate_temp_folder()
        CP._check_directory(tmpfolder)
        self.assertTrue(os.path.isdir(tmpfolder), msg = 'Directory exists')
        os.removedirs(tmpfolder)
        
# -------------------
class CheckSaveToTextFile(unittest.TestCase):
    def test_save_to_text_file(self):
        tmpfile = generate_temp_file(extension = 'txt')
        CP._save_to_txt_file(filename = tmpfile, mtx = np.array([0.1, 0.2]))
        self.assertTrue(os.path.isfile(tmpfile), msg = 'File exists')
        loadmtx = np.loadtxt(tmpfile)
        self.assertTrue(np.array_equal(loadmtx, np.array([0.1, 0.2])), msg = str('Arrays should match: {}'.format(loadmtx)))
        os.remove(tmpfile)
        
# -------------------
class CheckSaveToBinaryFile(unittest.TestCase):
    def test_save_to_binary_file(self):
        tmpfile = generate_temp_file(extension = 'h5')
        CP._save_to_bin_file(filename = tmpfile, datasetname = 'd1', mtx = np.array([0.1, 0.2]))
        self.assertTrue(os.path.isfile(tmpfile), msg = 'File exists')
        os.remove(tmpfile)
        
# -------------------
class AddToLog(unittest.TestCase):
    def test_add_to_log(self):
        tmpfile = generate_temp_file(extension = 'txt')
        CP._add_to_log(filename = tmpfile, logstr = 'hello world')
        self.assertTrue(os.path.isfile(tmpfile), msg = 'File exists')
        with open(tmpfile, 'r') as file:
            loadstr = file.read()
        self.assertEqual(loadstr, 'hello world', msg = 'Message should match')
        os.remove(tmpfile)