#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 09:12:06 2018

@author: prmiles
"""
import h5py
import numpy as np
import os
#import fnmatch
import warnings

def print_log_files(savedir):
    '''
    Print log files to screen.

    Args:
        * **savedir** (:py:class:`str`): Directory where log files are saved.
        
    The output display will include a date/time stamp, as well as indices of
    the chain that were saved during that export sequence.
    
    Example display:
    ::
        
        --------------------------
        Display log file: <savedir>/binlogfile.txt
        2018-05-03 14:15:54     0       999
        2018-05-03 14:15:54     1000    1999
        2018-05-03 14:15:55     2000    2999
        2018-05-03 14:15:55     3000    3999
        2018-05-03 14:15:55     4000    4999
        --------------------------
    '''
    # file files in savedir with h5 extension
    bf = []
    for file in os.listdir(savedir):
        if 'log' in file:
            bf.append(file)
            
    nlogf = len(bf) # number of log files found
    
    for ii in range(nlogf):
        logfile = _create_path_without_extension(savedir, bf[ii])
        
        with open(logfile, 'r') as file:
            logstr = file.read()
        
        print('--------------------------')
        print('Display log file: {}'.format(logfile))
        print(logstr)
        print('--------------------------\n')
                
      
def read_in_savedir_files(savedir, extension = 'h5', chainfile = 'chainfile', sschainfile = 'sschainfile', s2chainfile = 's2chainfile', covchainfile = 'covchainfile'):
    '''
    Read in log files from directory.
    
    Args:
        * **savedir** (:py:class:`str`): Directory where log files are saved.
        * **extension** (:py:class:`str`): Extension of files being loaded.
        * **chainfile** (:py:class:`str`): Name of chain log file.
        * **sschainfile** (:py:class:`str`): Name of sschain log file.
        * **s2chainfile** (:py:class:`str`): Name of s2chain log file.
        * **covchainfile** (:py:class:`str`): Name of covchain log file.
    '''
    # file files in savedir with h5 extension
    bf = []
    for file in os.listdir(savedir):
        if file.endswith(str('.{}'.format(extension))):
            bf.append(file)
            
    # create full path names with extension
    chainfile = _create_path_with_extension(savedir, chainfile, extension = extension)
    sschainfile = _create_path_with_extension(savedir, sschainfile, extension = extension)
    s2chainfile = _create_path_with_extension(savedir, s2chainfile, extension = extension)
    covchainfile = _create_path_with_extension(savedir, covchainfile, extension = extension)
   
    if extension == 'h5':
        # read in binary data -> assign to dictionary
        out = {
                'chain': read_in_bin_file(chainfile),
                'sschain': read_in_bin_file(sschainfile),
                's2chain': read_in_bin_file(s2chainfile),
                'covchain': read_in_bin_file(covchainfile)
                }
    elif extension == 'txt':
        # read in text data -> assign to dictionary
        out = {
                'chain': read_in_txt_file(chainfile),
                'sschain': read_in_txt_file(sschainfile),
                's2chain': read_in_txt_file(s2chainfile),
                'covchain': read_in_txt_file(covchainfile)
                }
    else:
        out = None
        warnings.warn('Unknown extension specified -> log files saved as either h5 (binary) or txt (text).')
    
    return out
    
def read_in_parallel_savedir_files(parallel_dir, extension = 'h5', chainfile = 'chainfile', sschainfile = 'sschainfile', s2chainfile = 's2chainfile', covchainfile = 'covchainfile'):
    '''
    Read in log files from directory containing results from parallel MCMC simulation.
    
    Args:
        * **parallel_dir** (:py:class:`str`): Directory where parallel log files are saved.
        * **extension** (:py:class:`str`): Extension of files being loaded.
        * **chainfile** (:py:class:`str`): Name of chain log file.
        * **sschainfile** (:py:class:`str`): Name of sschain log file.
        * **s2chainfile** (:py:class:`str`): Name of s2chain log file.
        * **covchainfile** (:py:class:`str`): Name of covchain log file.
    '''
    # find folders in parallel_dir with name 'chain_#'
    chainfolders = os.listdir(parallel_dir)
    out = []
    for folder in chainfolders:
        # create full path names with extension
        savedir = _create_path_without_extension(parallel_dir, folder)
        out.append(read_in_savedir_files(savedir, extension = extension, chainfile = chainfile, sschainfile = sschainfile, s2chainfile = s2chainfile, covchainfile = covchainfile))
        
    return out
    
def read_in_bin_file(filename):
    '''
    Read in information from file containing binary data.
    
    If file exists, it will read in the array elements.  Otherwise, it will return
    and empty list.
    
    Args:
        * **filename** (:py:class:`str`): Name of file to read.

    Returns:
        * **out** (:class:`~numpy.ndarray`): Array of chain elements.
    '''
    try:
        hf = h5py.File(filename, 'r')
        ds = list(hf.keys()) # data sets
        # check size of each set
        sh = np.zeros([len(ds),2], dtype=int)
        for ii, dsii in enumerate(ds):
            tmp = hf.get(ds[ii])
            sh[ii,:] = tmp.shape
            
        # initialize array for merged datasets
        # assume all have same number of columns
        out = np.zeros([sum(sh[:,0]),sh[0,1]])
        for ii, dsii in enumerate(ds):
            if ii == 0:
                a = 0;
                b = sh[ii,0]
            else:
                a = a + sh[ii-1,0]
                b = b + sh[ii-1,0]
                
            out[a:b,:] = np.array(hf.get(dsii))
            
        hf.close()
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
        warnings.warn('Exception raised reading {} - does not exist - check path/file name.  Assigning to empty list.'.format(filename))
        out = []
    
    return out
    
def read_in_txt_file(filename):
    '''
    Read in information from file containing text data.
    
    If file exists, it will read in the array elements.  Otherwise, it will return
    and empty list.
    
    Args:
        * **filename** (:py:class:`str`): Name of file to read.
    
    Returns:
        * **out** (:class:`~numpy.ndarray`): Array of chain elements.
    '''
    try:
        out = np.loadtxt(filename)
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
        warnings.warn('Exception raised reading {} - does not exist - check path/file name.  Assigning to empty list.'.format(filename))
        out = []
    
    return out
    
def _create_path_with_extension_for_all_logs(options, extension = 'h5'):
    chainfile = _create_path_with_extension(options.savedir, options.chainfile, extension)
    s2chainfile = _create_path_with_extension(options.savedir, options.s2chainfile, extension)
    sschainfile = _create_path_with_extension(options.savedir, options.sschainfile, extension)
    covchainfile = _create_path_with_extension(options.savedir, options.covchainfile, extension)
    return chainfile, s2chainfile, sschainfile, covchainfile
    
def _create_path_with_extension(savedir, file, extension = 'h5'):
    file = os.path.join(savedir, str('{}.{}'.format(file,extension)))
    return file
    
def _create_path_without_extension(savedir, file):
    file = os.path.join(savedir, file)
    return file
    
def _add_to_log(filename, logstr):
    with open(filename, 'a') as logfile:
        logfile.write(logstr)
    
def _save_to_bin_file(filename, datasetname, mtx):
    hf = h5py.File(filename, 'a')
    hf.create_dataset(datasetname, data = mtx)
    hf.close()
        
def _save_to_txt_file(filename, mtx):
    handle = open(filename, 'ab')
    np.savetxt(handle,mtx)
    handle.close()
    
def _check_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)