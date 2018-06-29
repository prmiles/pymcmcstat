#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 07:34:40 2018

@author: prmiles
"""
from __future__ import print_function

from pymcmcstat.utilities import progressbar as pbar
import unittest
from mock import patch
import io
import sys

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
        self.assertEqual(PB.percentage(i = i), 100 * i / float(100), msg = 'Percentage = 37%')
        
#    def test_update(self):
#        PB = pbar.progress_bar(iters=100)
#        i = 37
#        self.assertEqual(PB.update(i = i), 1, msg = 'None')
        
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

# --------------------------
class IpythonProgressBarMock(unittest.TestCase):
    @patch('pymcmcstat.utilities.progressbar.run_from_ipython')
    def test_ipython_console(self, mock_simple_func):
        mock_simple_func.return_value = True
        PB = pbar.progress_bar(iters = 100)
        self.assertEqual(PB.printer, pbar.ipythonprint, msg = 'ipythonprint')

# --------------------------
class IpythonPrint(unittest.TestCase):
    
    def test_ipythonprint(self):
        capturedOutput = io.StringIO()                  # Create StringIO object
        sys.stdout = capturedOutput                     #  and redirect stdout.
        pbar.ipythonprint('test')                                     # Call function.
        sys.stdout = sys.__stdout__                     # Reset redirect.
        self.assertEqual(capturedOutput.getvalue(), '\r test', msg = 'Expected string')

# --------------------------
class ConsolePrint(unittest.TestCase):
    
    def test_standard_print(self):
        capturedOutput = io.StringIO()                  # Create StringIO object
        sys.stdout = capturedOutput                     #  and redirect stdout.
        pbar.consoleprint('test')                                     # Call function.
        sys.stdout = sys.__stdout__                     # Reset redirect.
        self.assertEqual(capturedOutput.getvalue(), 'test\n', msg = 'Expected string')
       
    @patch('pymcmcstat.utilities.progressbar.check_windows_platform')
    def test_windows_print(self, mock_simple_func):
        mock_simple_func.return_value = True
        capturedOutput = io.StringIO()                  # Create StringIO object
        sys.stdout = capturedOutput                     #  and redirect stdout.
        pbar.consoleprint('test')                                     # Call function.
        sys.stdout = sys.__stdout__                     # Reset redirect.
        self.assertEqual(capturedOutput.getvalue(), 'test \r', msg = 'Expected string')