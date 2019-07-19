# -*- coding: utf-8 -*-

import unittest
import pymcmcstat
import re

class ImportPymcmcstat(unittest.TestCase):

    def test_version_attribute(self):
        version = pymcmcstat.__version__
        self.assertTrue(isinstance(version, str),
                        msg='Expect string output')
        pattern = '\d+\.\d+\.\d.*'
        mo = re.search(pattern, version, re.M)
        divs = mo.group().split('.')
        ndot = len(divs)
        self.assertGreaterEqual(ndot, 3,
                        msg='Expect min of two dots to match symver.')
        self.assertTrue(isinstance(float(divs[0]), float),
                        msg='Major version should be float')
        self.assertTrue(isinstance(float(divs[1]), float),
                        msg='Minor version should be float')
        self.assertTrue(isinstance(float(divs[2][0]), float),
                        msg='First element of bug version should be float')
        
