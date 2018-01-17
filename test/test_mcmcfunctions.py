# -*- coding: utf-8 -*-

"""
Created on Mon Jan. 15, 2018

Description: This file contains a series of utilities designed to test the
features in the 'mcmcfunctions.py" package of the pymcmcstat module.  The 
functions tested include:
    - alphafun(trypath, A_count, invR):
    - qfun(iq, trypath, invR):
    - logposteriorratio(x1, x2):
    - gammar(m,n,a,b = 1):
    - gammar_mt(m, n, a, b = 1):
    - gammar_mt1(a,b):
    - covupd(x, w, oldcov, oldmean, oldwsum, oldR = None): 
    - cholupdate(R, x):
    - chainstats(chain, results = []):
    - batch_mean_standard_deviation(x, b = None):
    - setup_no_adapt_index(noadaptind, parind):
    - setup_covariance_matrix(qcov, thetasig, value):
    - check_adascale(adascale, npar):
    - setup_R_matrix(qcov, parind):
    - setup_RDR_matrix(R, invR, npar, drscale, ntry, options):
    - check_dependent_parameters(N, data, nbatch, N0, S20, sigma2, savesize, nsimu, 
                               updatesigma, ntry, lastadapt, printint, adaptint):

@author: prmiles
"""
from pymcmcstat.mcmcfunctions import alphafun, qfun, logposteriorratio
from pymcmcstat.mcmcfunctions import gammar, gammar_mt, gammar_mt1, covupd
from pymcmcstat.mcmcfunctions import cholupdate, chainstats, batch_mean_standard_deviation
from pymcmcstat.mcmcfunctions import setup_no_adapt_index, setup_covariance_matrix
from pymcmcstat.mcmcfunctions import check_adascale, setup_R_matrix, setup_RDR_matrix
from pymcmcstat.mcmcfunctions import check_dependent_parameters
import unittest
import numpy as np

# --------------------------
# alphafun
# --------------------------
#class Alphafun_Test(unittest.TestCase):