#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 06:17:24 2018

@author: prmiles
"""
from pymcmcstat.utilities.general import message, removekey, check_settings
from pymcmcstat.utilities.general import format_number_to_str
import unittest
import io
import sys

# --------------------------
class MessageDisplay(unittest.TestCase):
    def test_standard_print(self):
        capturedOutput = io.StringIO()                  # Create StringIO object
        sys.stdout = capturedOutput                     #  and redirect stdout.
        flag = message(verbosity = 1, level = 0, printthis = 'test')
        sys.stdout = sys.__stdout__                     # Reset redirect.
        self.assertEqual(capturedOutput.getvalue(), 'test\n', msg = 'Expected string')
        self.assertTrue(flag, msg = 'Statement was printed')

    def test_no_print(self):
        capturedOutput = io.StringIO()                  # Create StringIO object
        sys.stdout = capturedOutput                     #  and redirect stdout.
        flag = message(verbosity = 0, level = 1, printthis = 'test')
        sys.stdout = sys.__stdout__                     # Reset redirect.
        self.assertFalse(flag, msg = 'Statement not printed because verbosity less than level')

        
# --------------------------
class FormatNumberToStr(unittest.TestCase):
    def test_format_number_to_str(self):
        self.assertEqual(str('{:9.2f}'.format(1.0)), format_number_to_str(1.0), msg = str('Exect: {:9.2f}'.format(1.0)))
        self.assertEqual(str('{:9.2e}'.format(1.0e4)), format_number_to_str(1.0e4), msg = str('Exect: {:9.2f}'.format(1.0e4)))
        self.assertEqual(str('{:9.2e}'.format(1.0e-2)), format_number_to_str(1.0e-2), msg = str('Exect: {:9.2f}'.format(1.0e-2)))


# --------------------------
class RemoveKey(unittest.TestCase):
    def test_removekey(self):
        d = {'a1': 0.1, 'a2': 0.2, 'a3': 'hello'}
        r = removekey(d, 'a2')
        check_these = ['a1', 'a3']
        for ct in check_these:
            self.assertEqual(d[ct], r[ct], msg = str('Expect element agreement: {}'.format(ct)))
            
        self.assertFalse('a2' in r, msg = 'Expect removal')


# --------------------------
class CheckSettings(unittest.TestCase):

    def test_settings_with_user_none(self):
        user_settings = None
        default_settings = dict(a = False, linewidth = 3, marker = dict(markersize = 5, color = 'g'))
        settings = check_settings(default_settings = default_settings, user_settings = user_settings)
        self.assertEqual(settings, default_settings, msg = str('Expect dictionaries to match: {} neq {}'.format(settings, default_settings)))
        
    def test_settings_with_subdict(self):
        user_settings = dict(a = True, fontsize = 12)
        default_settings = dict(a = False, linewidth = 3, marker = dict(markersize = 5, color = 'g'))
        settings = check_settings(default_settings = default_settings, user_settings = user_settings)
        self.assertEqual(settings['a'], user_settings['a'], msg = 'Expect user setting to overwrite')
        self.assertEqual(settings['marker'], default_settings['marker'], msg = 'Expect default to persist')
        
    def test_settings_with_subdict_user_ow(self):
        user_settings = dict(a = True, fontsize = 12, marker = dict(color = 'b'))
        default_settings = dict(a = False, linewidth = 3, marker = dict(markersize = 5, color = 'g'))
        settings = check_settings(default_settings = default_settings, user_settings = user_settings)
        self.assertEqual(settings['a'], user_settings['a'], msg = 'Expect user setting to overwrite')
        self.assertEqual(settings['marker']['color'], user_settings['marker']['color'], msg = 'Expect user to overwrite')
        self.assertEqual(settings['marker']['markersize'], default_settings['marker']['markersize'], msg = 'Expect default to persist')
        
    def test_settings_with_subdict_user_has_new_setting(self):
        user_settings = dict(a = True, fontsize = 12, marker = dict(color = 'b'), linestyle = '--')
        default_settings = dict(a = False, linewidth = 3, marker = dict(markersize = 5, color = 'g'))
        settings = check_settings(default_settings = default_settings, user_settings = user_settings)
        self.assertEqual(settings['a'], user_settings['a'], msg = 'Expect user setting to overwrite')
        self.assertEqual(settings['marker']['color'], user_settings['marker']['color'], msg = 'Expect user to overwrite')
        self.assertEqual(settings['marker']['markersize'], default_settings['marker']['markersize'], msg = 'Expect default to persist')
        self.assertEqual(settings['linestyle'], user_settings['linestyle'], msg = 'Expect user setting to be added')

