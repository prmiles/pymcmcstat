#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 12:21:24 2018

@author: prmiles
"""

from pymcmcstat.plotting import MCMCPlotting as MP
import numpy as np
import math

import unittest

# --------------------------
class PlotChainPanel(unittest.TestCase):
    def test_basic_plot_features_nsimu_lt_maxpoints(self):
        chains = np.random.random_sample(size = (100,2))
        f = MP.plot_chain_panel(chains = chains)
        x1, y1 = f.axes[0].lines[0].get_xydata().T
        x2, y2 = f.axes[1].lines[0].get_xydata().T
        self.assertTrue(np.array_equal(y1, chains[:,0]), msg = 'Expect y1 to match column 1')
        self.assertTrue(np.array_equal(y2, chains[:,1]), msg = 'Expect y2 to match column 2')
        self.assertEqual(f.axes[0].get_xlabel(), '', msg = 'Should be blank')
        self.assertEqual(f.axes[1].get_xlabel(), 'Iteration', msg = 'Should be Iteration')
        
    def test_basic_plot_features_nsimu_gt_maxpoints(self):
        nsimu = 1000
        chains = np.random.random_sample(size = (nsimu,2))
        f = MP.plot_chain_panel(chains = chains)
        x1, y1 = f.axes[0].lines[0].get_xydata().T
        x2, y2 = f.axes[1].lines[0].get_xydata().T
        skip = int(math.floor(nsimu/500))
        self.assertTrue(np.array_equal(y1, chains[range(0,nsimu,skip),0]), msg = 'Expect y1 to match column 1')
        self.assertTrue(np.array_equal(y2, chains[range(0,nsimu,skip),1]), msg = 'Expect y2 to match column 2')
        self.assertEqual(f.axes[0].get_xlabel(), '', msg = 'Should be blank')
        self.assertEqual(f.axes[1].get_xlabel(), 'Iteration', msg = 'Should be Iteration')