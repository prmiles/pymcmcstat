#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 10:10:43 2018

@author: prmiles
"""

from pymcmcstat.chain import ChainProcessing as CP
import test.general_functions as gf
import unittest
from mock import patch
import numpy as np
import os
import shutil
import io
import sys

def setup_case():
    mcstat = gf.basic_mcmc()
    mcstat._MCMC__chain = np.random.random_sample(size = (100,2))
    mcstat._MCMC__sschain = np.random.random_sample(size = (100,2))
    mcstat._MCMC__s2chain = np.random.random_sample(size = (100,2))
    mcstat._covariance._R = np.array([[0.5, 0.2],[0., 0.3]])
    return mcstat

# --------------------------
class CreatePathWithExtensionforAllLogs(unittest.TestCase):
    
    def test_cpwe_creation_with_h5(self):
        mcstat = gf.setup_mcmc_case_cp()
        mcstat.simulation_options.savedir = 'test'
        chainfile, s2chainfile, sschainfile, covchainfile = CP._create_path_with_extension_for_all_logs(options = mcstat.simulation_options, extension = 'h5')
        self.assertEqual(chainfile, 'test/chainfile.h5', msg = 'Expect matching string')
        self.assertEqual(s2chainfile, 'test/s2chainfile.h5', msg = 'Expect matching string')
        self.assertEqual(sschainfile, 'test/sschainfile.h5', msg = 'Expect matching string')
        self.assertEqual(covchainfile, 'test/covchainfile.h5', msg = 'Expect matching string')
        
    def test_cpwe_creation_with_txt(self):
        mcstat = gf.setup_mcmc_case_cp()
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
        tmpfile = gf.generate_temp_file(extension = 'h5')
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
        tmpfolder = gf.generate_temp_folder()
        CP._check_directory(tmpfolder)
        self.assertTrue(os.path.isdir(tmpfolder), msg = 'Directory exists')
        os.removedirs(tmpfolder)
        
# -------------------
class CheckSaveToTextFile(unittest.TestCase):
    def test_save_to_text_file(self):
        tmpfile = gf.generate_temp_file(extension = 'txt')
        CP._save_to_txt_file(filename = tmpfile, mtx = np.array([0.1, 0.2]))
        self.assertTrue(os.path.isfile(tmpfile), msg = 'File exists')
        loadmtx = np.loadtxt(tmpfile)
        self.assertTrue(np.array_equal(loadmtx, np.array([0.1, 0.2])), msg = str('Arrays should match: {}'.format(loadmtx)))
        os.remove(tmpfile)
        
# -------------------
class CheckSaveToBinaryFile(unittest.TestCase):
    def test_save_to_binary_file(self):
        tmpfile = gf.generate_temp_file(extension = 'h5')
        CP._save_to_bin_file(filename = tmpfile, datasetname = 'd1', mtx = np.array([0.1, 0.2]))
        self.assertTrue(os.path.isfile(tmpfile), msg = 'File exists')
        os.remove(tmpfile)
        
# -------------------
class AddToLog(unittest.TestCase):
    def test_add_to_log(self):
        tmpfile = gf.generate_temp_file(extension = 'txt')
        CP._add_to_log(filename = tmpfile, logstr = 'hello world')
        self.assertTrue(os.path.isfile(tmpfile), msg = 'File exists')
        with open(tmpfile, 'r') as file:
            loadstr = file.read()
        self.assertEqual(loadstr, 'hello world', msg = 'Message should match')
        os.remove(tmpfile)
# -------------------
class ReadInSavedirFiles(unittest.TestCase):
    
    def test_read_in_savedir_files_h5(self):
        mcstat = setup_case()
        savedir = gf.generate_temp_folder()
        mcstat.simulation_options.savedir = savedir
        
        mcstat._MCMC__save_chains_to_bin(start = 0, end = 100)
        
        out = CP.read_in_savedir_files(savedir, extension = 'h5')
        self.assertTrue(np.array_equal(out['chain'], mcstat._MCMC__chain), msg = str('Expect arrays to match: chain'))
        self.assertTrue(np.array_equal(out['sschain'], mcstat._MCMC__sschain), msg = str('Expect arrays to match: sschain'))
        self.assertTrue(np.array_equal(out['s2chain'], mcstat._MCMC__s2chain), msg = str('Expect arrays to match: s2chain'))
        self.assertTrue(np.array_equal(out['covchain'], np.dot(mcstat._covariance._R.transpose(),mcstat._covariance._R)), msg = str('Expect arrays to match: chain'))
        shutil.rmtree(savedir)
        
    def test_read_in_savedir_files_txt(self):
        mcstat = setup_case()
        savedir = gf.generate_temp_folder()
        mcstat.simulation_options.savedir = savedir
        
        mcstat._MCMC__save_chains_to_txt(start = 0, end = 100)
        
        out = CP.read_in_savedir_files(savedir, extension = 'txt')
        self.assertTrue(np.array_equal(out['chain'], mcstat._MCMC__chain), msg = str('Expect arrays to match: chain'))
        self.assertTrue(np.array_equal(out['sschain'], mcstat._MCMC__sschain), msg = str('Expect arrays to match: sschain'))
        self.assertTrue(np.array_equal(out['s2chain'], mcstat._MCMC__s2chain), msg = str('Expect arrays to match: s2chain'))
        self.assertTrue(np.array_equal(out['covchain'], np.dot(mcstat._covariance._R.transpose(),mcstat._covariance._R)), msg = str('Expect arrays to match: chain'))
        shutil.rmtree(savedir)
        
    def test_read_in_savedir_files_unknown(self):
        mcstat = setup_case()
        savedir = gf.generate_temp_folder()
        mcstat.simulation_options.savedir = savedir
        
        mcstat._MCMC__save_chains_to_txt(start = 0, end = 100)
        
        out = CP.read_in_savedir_files(savedir, extension = 'unknown')
        self.assertEqual(out, None, msg = 'Expect None')
        shutil.rmtree(savedir)

# -------------------
class ReadInParallelSavedirFiles(unittest.TestCase):
    def chain_comp(self, out, mcstat):
        self.assertTrue(np.array_equal(out['chain'], mcstat._MCMC__chain), msg = str('Expect arrays to match: chain'))
        self.assertTrue(np.array_equal(out['sschain'], mcstat._MCMC__sschain), msg = str('Expect arrays to match: sschain'))
        self.assertTrue(np.array_equal(out['s2chain'], mcstat._MCMC__s2chain), msg = str('Expect arrays to match: s2chain'))
        self.assertTrue(np.array_equal(out['covchain'], np.dot(mcstat._covariance._R.transpose(),mcstat._covariance._R)), msg = str('Expect arrays to match: chain'))

    def test_read_in_parallel_bin(self):
        mcstat = setup_case()
        parallel_dir = gf.generate_temp_folder()
        for ii in range(3):
            chain_dir = str('chain_{}'.format(ii))
            mcstat.simulation_options.savedir = str('{}{}{}'.format(parallel_dir,os.sep,chain_dir))
            mcstat._MCMC__save_chains_to_bin(start = 0, end = 100)
        
        out = CP.read_in_parallel_savedir_files(parallel_dir = parallel_dir, extension = 'h5')
        self.chain_comp(out[0], mcstat)
        self.chain_comp(out[1], mcstat)
        self.chain_comp(out[2], mcstat)
        shutil.rmtree(parallel_dir)
        
    def test_read_in_parallel_txt(self):
        mcstat = setup_case()
        parallel_dir = gf.generate_temp_folder()
        for ii in range(3):
            chain_dir = str('chain_{}'.format(ii))
            mcstat.simulation_options.savedir = str('{}{}{}'.format(parallel_dir,os.sep,chain_dir))
            mcstat._MCMC__save_chains_to_txt(start = 0, end = 100)
        
        out = CP.read_in_parallel_savedir_files(parallel_dir = parallel_dir, extension = 'txt')
        self.chain_comp(out[0], mcstat)
        self.chain_comp(out[1], mcstat)
        self.chain_comp(out[2], mcstat)
        shutil.rmtree(parallel_dir)
        
    def test_read_in_parallel_unknown(self):
        mcstat = setup_case()
        parallel_dir = gf.generate_temp_folder()
        for ii in range(3):
            chain_dir = str('chain_{}'.format(ii))
            mcstat.simulation_options.savedir = str('{}{}{}'.format(parallel_dir,os.sep,chain_dir))
            mcstat._MCMC__save_chains_to_bin(start = 0, end = 100)
        
        out = CP.read_in_parallel_savedir_files(parallel_dir = parallel_dir, extension = 'unknown')
        self.assertEqual(out[0], None)
        self.assertEqual(out[1], None)
        self.assertEqual(out[2], None)
        shutil.rmtree(parallel_dir)
# -------------------
class PrintLogFiles(unittest.TestCase):
    def test_print_log_files(self):
        mcstat = setup_case()
        savedir = gf.generate_temp_folder()
        mcstat.simulation_options.savedir = savedir
        
        mcstat._MCMC__save_chains_to_txt(start = 0, end = 100)
        
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        CP.print_log_files(savedir)
        sys.stdout = sys.__stdout__
        
        self.assertTrue(isinstance(capturedOutput.getvalue(), str), msg = 'Should contain a string')
        self.assertTrue('Display log file:' in capturedOutput.getvalue(), msg = 'Expect string to contain these works')
        shutil.rmtree(savedir)