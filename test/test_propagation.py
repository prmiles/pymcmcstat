import unittest
from mock import patch
import pymcmcstat.propagation as uqp
import numpy as np

# --------------------------
class CheckLimits(unittest.TestCase):

    def test_check(self):
        limits = uqp._check_limits(None, [50, 90])
        self.assertEqual(limits, [50, 90],
                         msg='Expect default return')
        limits = uqp._check_limits([75, 95], [50, 90])
        self.assertEqual(limits, [75, 95],
                         msg='Expect non-default return')


# --------------------------
class ConvertLimits(unittest.TestCase):

    def test_conversion(self):
        limits = uqp._convert_limits([50, 90])
        rng = []
        rng.append([0.25, 0.75])
        rng.append([0.05, 0.95])
        self.assertTrue(np.allclose(np.array(limits), np.array(rng)),
                        msg='Expect matching lists')


# --------------------------------------------
class DefineSamplePoints(unittest.TestCase):

    def test_define_sample_points_nsample_gt_nsimu(self):
        iisample, nsample = uqp._define_sample_points(nsample=1000,
                                                      nsimu=500)
        self.assertEqual(iisample, range(500),
                         msg='Expect range(500)')
        self.assertEqual(nsample, 500,
                         msg='Expect nsample updated to 500')
        
    @patch('numpy.random.rand')
    def test_define_sample_points_nsample_lte_nsimu(self, mock_rand):
        aa = np.random.rand([400, 1])
        mock_rand.return_value = aa
        iisample, nsample = uqp._define_sample_points(nsample=400,
                                                      nsimu=500)
        self.assertTrue(np.array_equal(iisample, np.ceil(aa*500) - 1),
                        msg='Expect range(500)')
        self.assertEqual(nsample, 400,
                         msg='Expect nsample to stay 400')


# --------------------------------------------
class Observation_Sample_Test(unittest.TestCase):

    def test_sstype(self):
        s2elem = np.array([[2.0]])
        ypred = np.linspace(2.0, 3.0, num=5)
        ypred = ypred.reshape(5, 1)
        with self.assertRaises(SystemExit, msg='Unrecognized sstype'):
            uqp._observation_sample(s2elem, ypred, 3)
        opred = uqp._observation_sample(s2elem, ypred, 0)
        self.assertEqual(opred.shape, ypred.shape,
                         msg='Shapes should match')
        opred = uqp._observation_sample(s2elem, ypred, 1)
        self.assertEqual(opred.shape, ypred.shape,
                         msg='Shapes should match')
        opred = uqp._observation_sample(s2elem, ypred, 2)
        self.assertEqual(opred.shape, ypred.shape,
                         msg='Shapes should match')


# --------------------------------------------
class CheckS2Chain(unittest.TestCase):

    def test_checks2chain(self):
        s2elem = np.array([[2.0, 10.]])
        with self.assertRaises(SystemExit,
                               msg='Expect s2chain as float or array of size nsimu'):
            uqp.check_s2chain(s2elem, 5)
        s2elem = None
        s2chain = uqp.check_s2chain(s2elem, 5)
        self.assertEqual(s2chain, None, msg='Expect None return')
        s2elem = np.zeros(shape=(5,))
        s2chain = uqp.check_s2chain(s2elem, 5)
        self.assertTrue(np.allclose(s2chain, s2elem),
                        msg='Expect as-is return')
        s2elem = 0.01
        s2chain = uqp.check_s2chain(s2elem, 5)
        self.assertTrue(np.allclose(s2chain, np.ones(5,)*s2elem),
                        msg='Expect float to be extended to array to match size of chain')
        