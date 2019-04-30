# -*- coding: utf-8 -*-

from pymcmcstat.procedures.LikelihoodFunction import LikelihoodFunction
import unittest
import test.general_functions as gf
import numpy as np


# --------------------------
class InitializeLikelihoodFunction(unittest.TestCase):

    def check_attributes(self, LF, check):
        for ca in check:
            self.assertTrue(hasattr(LF, ca),
                            msg=str('Object should'
                                    + ' have {} attribute'.format(ca)))
    def test_init_with_sos(self):
        model, options, parameters, data = gf.setup_mcmc()
        LF = LikelihoodFunction(model=model,
                                data=data,
                                parameters=parameters)
        check_atts = ['data', 'local', 'value', 'parind', 'type',
                      'likelihood', 'sos_function', 'model_function']
        self.check_attributes(LF, check_atts)
        self.assertEqual(LF.type, 'default',
                         msg='Should use default')

    def test_init_with_no_sos(self):
        model, options, parameters, data = gf.setup_mcmc()
        model.likelihood = model.sos_function
        LF = LikelihoodFunction(model=model,
                                data=data,
                                parameters=parameters)
        check_atts = ['data', 'local', 'value', 'parind', 'type',
                      'likelihood']
        self.check_attributes(LF, check_atts)
        self.assertEqual(LF.type, 'custom',
                         msg='User defined likelihood')


# --------------------------
class CheckSOS(unittest.TestCase):

    def test_check_sos_1_elem(self):
        model, options, parameters, data = gf.setup_mcmc()
        LF = LikelihoodFunction(model=model,
                                data=data,
                                parameters=parameters)
        ssq = LF._check_sos(ssq=1.0)
        self.assertTrue(isinstance(ssq, np.ndarray),
                        msg='Expect array return')
        self.assertEqual(ssq, np.array([1.0]))
        ssq = LF._check_sos(ssq=np.array(1.0))
        self.assertTrue(isinstance(ssq, np.ndarray),
                        msg='Expect array return')
        self.assertEqual(ssq, np.array([1.0]))
        with self.assertRaises(SystemExit):
            LF._check_sos(ssq=int(1))

    def test_check_sos_n_elem(self):
        model, options, parameters, data = gf.setup_mcmc()
        LF = LikelihoodFunction(model=model,
                                data=data,
                                parameters=parameters)
        ssq = LF._check_sos(ssq=[1.0, 2.0])
        self.assertTrue(isinstance(ssq, np.ndarray),
                        msg='Expect array return')
        self.assertTrue(np.array_equal(ssq, np.array([1.0, 2.0])),
                        msg=str('{} ne {}'.format(ssq, np.array([1.0, 2.0]))))
        ssq = LF._check_sos(ssq=np.array([1.0, 2.0]))
        self.assertTrue(isinstance(ssq, np.ndarray),
                        msg='Expect array return')
        self.assertTrue(np.array_equal(ssq, np.array([1.0, 2.0])),
                        msg=str('{} ne {}'.format(ssq, np.array([1.0, 2.0]))))
        with self.assertRaises(SystemExit):
            LF._check_sos(ssq=np.array([[2, 2], [2, 2]]))