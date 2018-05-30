#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 07:34:40 2018

@author: prmiles
"""
from pymcmcstat.utilities import progressbar as pbar
import unittest

# --------------------------
class InitializeProgress_Bar(unittest.TestCase):

    def test_PB_iteration_assignment(self):
        PB = pbar.progress_bar(iters = 100)
        self.assertEqual(PB.iterations, 100, msg = 'Iteration assignment should be 100')
        
    def test_PB_animation_interval(self):
        PB = pbar.progress_bar(iters = 100)
        self.assertEqual(PB.animation_interval, 0.5, msg = 'Animation interval should be 0.5')
        
# --------------------------
class BaseProgressBar(unittest.TestCase):   

    def test_initialize_progress_bar(self):
        PB = pbar.ProgressBar(iterations=100)
        self.assertEqual(PB.iterations, 100, msg = 'Iterations = 100')
        self.assertEqual(PB.animation_interval, 0.5, msg = 'Animation interval = 0.5')
        self.assertEqual(PB.last, 0, msg = 'last = 0')
        
    def test_initialize_progress_bar_change_animation_interval(self):
        PB = pbar.ProgressBar(iterations=100, animation_interval=0.78)
        self.assertEqual(PB.iterations, 100, msg = 'Iterations = 100')
        self.assertEqual(PB.animation_interval, 0.78, msg = 'Animation interval = 0.5')
        self.assertEqual(PB.last, 0, msg = 'last = 0')
        
    def test_return_percentage(self):
        PB = pbar.ProgressBar(iterations=100)
        i = 37
        self.assertEqual(PB.percentage(i = 37), 100 * i / float(100), msg = 'Percentage = 37%')
        
# --------------------------
class TextProgressBar(unittest.TestCase): 
    
    def test_initialization_text(self):
        TPB = pbar.TextProgressBar(iterations = 100, printer=pbar.consoleprint)
        self.assertEqual(TPB.width, 40, msg = 'Width is 40')
        self.assertEqual(TPB.fill_char, '-', msg = '-')
        self.assertEqual(TPB.printer, pbar.consoleprint, msg = 'consoleprint')
        
    def test_initialization_text_with_ipythonprint(self):
        TPB = pbar.TextProgressBar(iterations = 100, printer=pbar.ipythonprint)
        self.assertEqual(TPB.width, 40, msg = 'Width is 40')
        self.assertEqual(TPB.fill_char, '-', msg = '-')
        self.assertEqual(TPB.printer, pbar.ipythonprint, msg = 'ipythonprint')
        
    def test_progbar(self):
        TPB = pbar.TextProgressBar(iterations = 100, printer=pbar.consoleprint)
        self.assertTrue(isinstance(TPB.progbar(i = 50, elapsed=50),str), msg = 'Expected string return')