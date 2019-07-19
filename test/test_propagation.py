import unittest
from mock import patch
import pymcmcstat.propagation as uqp
from pymcmcstat.MCMC import DataStructure
import general_functions as gf
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
class CheckLimits(unittest.TestCase):

    def test_check(self):
        limits = uqp._check_limits(None, [50, 90])
        self.assertEqual(limits, [90, 50],
                         msg='Expect default return')
        limits = uqp._check_limits([75, 95], [50, 90])
        self.assertEqual(limits, [95, 75],
                         msg='Expect non-default return')


# --------------------------
class ConvertLimits(unittest.TestCase):

    def test_conversion(self):
        limits = uqp._convert_limits([90, 50])
        rng = []
        rng.append([0.05, 0.95])
        rng.append([0.25, 0.75])
        self.assertTrue(np.allclose(np.array(limits), np.array(rng)),
                        msg='Expect matching lists')
        limits = uqp._convert_limits([90, 50])
        rng = []
        rng.append([0.05, 0.95])
        rng.append([0.25, 0.75])
        self.assertTrue(np.allclose(np.array(limits), np.array(rng)),
                        msg='Expect matching lists')


# --------------------------------------------
class DefineSamplePoints(unittest.TestCase):

    def test_define_sample_points_nsample_gt_nsimu(self):
        iisample, nsample = uqp.define_sample_points(nsample=1000,
                                                      nsimu=500)
        self.assertEqual(iisample, range(500),
                         msg='Expect range(500)')
        self.assertEqual(nsample, 500,
                         msg='Expect nsample updated to 500')
        
    @patch('numpy.random.rand')
    def test_define_sample_points_nsample_lte_nsimu(self, mock_rand):
        aa = np.random.rand([400, 1])
        mock_rand.return_value = aa
        iisample, nsample = uqp.define_sample_points(nsample=400,
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
            uqp.observation_sample(s2elem, ypred, 3)
        opred = uqp.observation_sample(s2elem, ypred, 0)
        self.assertEqual(opred.shape, ypred.shape,
                         msg='Shapes should match')
        opred = uqp.observation_sample(s2elem, ypred, 1)
        self.assertEqual(opred.shape, ypred.shape,
                         msg='Shapes should match')
        opred = uqp.observation_sample(s2elem, ypred, 2)
        self.assertEqual(opred.shape, ypred.shape,
                         msg='Shapes should match')


# --------------------------------------------
class CheckS2Chain(unittest.TestCase):

    def test_checks2chain(self):
        s2elem = np.array([2.0, 10.])
        s2chain = uqp.check_s2chain(s2elem, 5)
        self.assertEqual(s2chain.shape, (5, 2),
                         msg='Expect (5, 2) array')
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


# --------------------------------------------
class GenerateQuantiles(unittest.TestCase):

    def test_does_default_empirical_quantiles_return_3_element_array(self):
        test_out = uqp.generate_quantiles(np.random.rand(10, 1))
        self.assertEqual(test_out.shape, (3, 1),
                         msg='Default output shape is (3,1)')

    def test_does_non_default_empirical_quantiles_return_2_element_array(self):
        test_out = uqp.generate_quantiles(np.random.rand(10, 1), p=np.array([0.2, 0.5]))
        self.assertEqual(test_out.shape, (2, 1),
                         msg='Non-default output shape should be (2, 1)')

    def test_empirical_quantiles_should_not_support_list_input(self):
        with self.assertRaises(AttributeError):
            uqp.generate_quantiles([-1, 0, 1])

    def test_empirical_quantiles_vector(self):
        out = uqp.generate_quantiles(np.linspace(10,20, num=10).reshape(10, 1),
                                            p=np.array([0.22, 0.57345]))
        exact = np.array([[12.2], [15.7345]])
        comp = np.linalg.norm(out - exact)
        self.assertAlmostEqual(comp, 0)


# --------------------------------------------
class SetupDisplaySettings(unittest.TestCase):

    def test_setup_display_settings(self):
        model_display = {'label': 'hello'}
        data_display = {'linewidth': 7}
        interval_display = {'edgecolor': 'b'}
        intd, modd, datd = uqp.setup_display_settings(
                interval_display=interval_display,
                model_display=model_display,
                data_display=data_display)
        self.assertEqual(modd['label'], model_display['label'],
                         msg='Expect label to match')
        self.assertEqual(intd['edgecolor'], interval_display['edgecolor'],
                         msg='Expect edgecolor to match')
        self.assertEqual(datd['linewidth'], data_display['linewidth'],
                         msg='Expect linewidth to match')


# --------------------------------------------
class SetupLabels(unittest.TestCase):

    def test_label_setup(self):
        limits = [90, 50]
        labels = uqp._setup_labels(limits, inttype='HH')
        self.assertEqual(labels[0], '90% HH',
                         msg='Expect matching strings')
        self.assertEqual(labels[1], '50% HH',
                         msg='Expect matching strings')


class SetupIntervalColors(unittest.TestCase):

    def test_sic_colors_none(self):
        iset = dict(
                limits=[90, 50],
                cmap=None,
                colors=None)
        ic = uqp.setup_interval_colors(iset)
        self.assertEqual(len(ic), 2,
                         msg='Expect 2 colors')

    def test_sic_colors_not_none(self):
        iset = dict(
                limits=[90, 50],
                cmap=None,
                colors=['r', 'g'])
        ic = uqp.setup_interval_colors(iset)
        self.assertEqual(len(ic), 2,
                         msg='Expect 2 colors')
        self.assertEqual(ic, ['r', 'g'],
                         msg='Expect matching lists')

    def test_sic_colors_not_none_but_wrong_size(self):
        iset = dict(
                limits=[90, 50],
                cmap=None,
                colors=['r', 'g', 'b'])
        ic = uqp.setup_interval_colors(iset)
        self.assertEqual(len(ic), 2,
                         msg='Expect 2 colors')
        self.assertNotEqual(ic, ['r', 'g'],
                         msg='Expect non-matching lists')


def model(q, data):
    m, b = q
    return m*data.xdata[0] + b


def model3D(q, data):
    m, b = q
    return m*data.xdata[0][:, 0] + b*data.xdata[0][:, 1]

def modelmultiple(q, data):
    m, b = q
    x = data.xdata[0]
    y1 = m*x + b
    y2 = m*x**2 + b
    return np.stack((y1.reshape(y1.size,), y2.reshape(y2.size,)), axis=1)


class CalculateIntervals(unittest.TestCase):

    def test_credintcreation(self):
        data = DataStructure()
        data.add_data_set(x=np.linspace(0, 1), y=None)
        results = gf.setup_pseudo_results()
        chain = results['chain']
        intervals = uqp.calculate_intervals(
                chain, results, data, model, waitbar=True)
        self.assertTrue('credible' in intervals.keys(),
                        msg='Expect credible intervals')
        self.assertTrue('prediction' in intervals.keys(),
                        msg='Expect prediction intervals')
        self.assertTrue(isinstance(intervals['credible'], np.ndarray),
                        msg='Expect numpy array')
        self.assertEqual(intervals['prediction'], None,
                        msg='Expect None')

    def test_predintcreation(self):
        data = DataStructure()
        data.add_data_set(x=np.linspace(0, 1), y=None)
        results = gf.setup_pseudo_results()
        chain = results['chain']
        s2chain = results['s2chain']
        intervals = uqp.calculate_intervals(
                chain, results, data, model, s2chain=s2chain)
        self.assertTrue('credible' in intervals.keys(),
                        msg='Expect credible intervals')
        self.assertTrue('prediction' in intervals.keys(),
                        msg='Expect prediction intervals')
        self.assertTrue(isinstance(intervals['credible'], np.ndarray),
                        msg='Expect numpy array')
        self.assertTrue(isinstance(intervals['prediction'], np.ndarray),
                        msg='Expect numpy array')
        intervals = uqp.calculate_intervals(
                chain, results, data, model, s2chain=0.1)
        self.assertTrue('credible' in intervals.keys(),
                        msg='Expect credible intervals')
        self.assertTrue('prediction' in intervals.keys(),
                        msg='Expect prediction intervals')
        self.assertTrue(isinstance(intervals['credible'], np.ndarray),
                        msg='Expect numpy array')
        self.assertTrue(isinstance(intervals['prediction'], np.ndarray),
                        msg='Expect numpy array')

    def test_predintcreation_multimodel(self):
        data = DataStructure()
        data.add_data_set(x=np.linspace(0, 1), y=None)
        results = gf.setup_pseudo_results()
        chain = results['chain']
        s2chain = results['s2chain']
        mintervals = uqp.calculate_intervals(
                chain, results, data, modelmultiple, s2chain=s2chain)
        for intervals in mintervals:
            self.assertTrue('credible' in intervals.keys(),
                            msg='Expect credible intervals')
            self.assertTrue('prediction' in intervals.keys(),
                            msg='Expect prediction intervals')
            self.assertTrue(isinstance(intervals['credible'], np.ndarray),
                            msg='Expect numpy array')
            self.assertTrue(isinstance(intervals['prediction'], np.ndarray),
                            msg='Expect numpy array')
        mintervals = uqp.calculate_intervals(
                chain, results, data, modelmultiple, s2chain=0.1)
        for intervals in mintervals:
            self.assertTrue('credible' in intervals.keys(),
                            msg='Expect credible intervals')
            self.assertTrue('prediction' in intervals.keys(),
                            msg='Expect prediction intervals')
            self.assertTrue(isinstance(intervals['credible'], np.ndarray),
                            msg='Expect numpy array')
            self.assertTrue(isinstance(intervals['prediction'], np.ndarray),
                            msg='Expect numpy array')
        mintervals = uqp.calculate_intervals(
                chain, results, data, modelmultiple,
                s2chain=np.hstack((s2chain, s2chain)))
        for intervals in mintervals:
            self.assertTrue('credible' in intervals.keys(),
                            msg='Expect credible intervals')
            self.assertTrue('prediction' in intervals.keys(),
                            msg='Expect prediction intervals')
            self.assertTrue(isinstance(intervals['credible'], np.ndarray),
                            msg='Expect numpy array')
            self.assertTrue(isinstance(intervals['prediction'], np.ndarray),
                            msg='Expect numpy array')


# --------------------------------------------
class Plot2DIntervals(unittest.TestCase):
    
    def test_plot_intervals_basic(self):
        data = DataStructure()
        data.add_data_set(x=np.linspace(0, 1), y=None)
        results = gf.setup_pseudo_results()
        chain = results['chain']
        s2chain = results['s2chain']
        intervals = uqp.calculate_intervals(
                chain, results, data, model, s2chain=s2chain)
        fig, ax = uqp.plot_intervals(intervals, data.xdata[0], limits=[95])
        self.assertEqual(ax.get_legend_handles_labels()[1],
                         ['Model', '95% PI', '95% CI'],
                         msg=str('Strings should match: {}'.format(
                                 ax.get_legend_handles_labels()[1])))
        plt.close()
        fig, ax = uqp.plot_intervals(intervals, data.xdata[0], limits=[95],
                                     adddata=True, ydata=data.xdata[0])
        self.assertEqual(ax.get_legend_handles_labels()[1],
                         ['Model', 'Data', '95% PI', '95% CI'],
                         msg=str('Strings should match: {}'.format(
                                 ax.get_legend_handles_labels()[1])))
        plt.close()

    def test_check_settings_plot_intervals_basic(self):
        data = DataStructure()
        data.add_data_set(x=np.linspace(0, 1), y=None)
        results = gf.setup_pseudo_results()
        chain = results['chain']
        s2chain = results['s2chain']
        intervals = uqp.calculate_intervals(
                chain, results, data, model, s2chain=s2chain)
        fig, ax, isets = uqp.plot_intervals(
                intervals, data.xdata[0], limits=[95],
                return_settings=True)
        self.assertEqual(ax.get_legend_handles_labels()[1],
                         ['Model', '95% PI', '95% CI'],
                         msg=str('Strings should match: {}'.format(
                                 ax.get_legend_handles_labels()[1])))
        self.assertTrue(isinstance(isets, dict),
                        msg='Expect dictionary')
        self.assertEqual(isets['ciset']['limits'], [95])
        self.assertEqual(isets['piset']['limits'], [95])
        plt.close()
        plt.close()
        fig, ax, isets = uqp.plot_intervals(
                intervals, data.xdata[0], limits=[95],
                adddata=True, ydata=data.xdata[0],
                return_settings=True)
        self.assertEqual(ax.get_legend_handles_labels()[1],
                         ['Model', 'Data', '95% PI', '95% CI'],
                         msg=str('Strings should match: {}'.format(
                                 ax.get_legend_handles_labels()[1])))
        self.assertTrue(isinstance(isets, dict),
                        msg='Expect dictionary')
        self.assertEqual(isets['ciset']['limits'], [95])
        self.assertEqual(isets['piset']['limits'], [95])
        plt.close()


# --------------------------------------------
class Plot3DIntervals(unittest.TestCase):
    
    def test_plot_intervals_basic(self):
        data = DataStructure()
        data.add_data_set(x=np.random.random_sample((100, 2)), y=None)
        results = gf.setup_pseudo_results()
        chain = results['chain']
        s2chain = results['s2chain']
        intervals = uqp.calculate_intervals(
                chain, results, data, model3D, s2chain=s2chain)
        fig, ax = uqp.plot_3d_intervals(intervals, data.xdata[0], limits=[95])
        self.assertEqual(ax.get_legend_handles_labels()[1],
                         ['Model', '95% PI', '95% CI'],
                         msg=str('Strings should match: {}'.format(
                                 ax.get_legend_handles_labels()[1])))
        plt.close()
        fig, ax = uqp.plot_3d_intervals(intervals, data.xdata[0], limits=[95],
                                     adddata=True, ydata=data.xdata[0][:, 0])
        self.assertEqual(ax.get_legend_handles_labels()[1],
                         ['Model', 'Data', '95% PI', '95% CI'],
                         msg=str('Strings should match: {}'.format(
                                 ax.get_legend_handles_labels()[1])))
        plt.close()

    def test_check_settings_plot_intervals_basic(self):
        data = DataStructure()
        data.add_data_set(x=np.random.random_sample((100, 2)), y=None)
        results = gf.setup_pseudo_results()
        chain = results['chain']
        s2chain = results['s2chain']
        intervals = uqp.calculate_intervals(
                chain, results, data, model3D, s2chain=s2chain)
        fig, ax, isets = uqp.plot_3d_intervals(
                intervals, data.xdata[0], limits=[95],
                return_settings=True)
        self.assertEqual(ax.get_legend_handles_labels()[1],
                         ['Model', '95% PI', '95% CI'],
                         msg=str('Strings should match: {}'.format(
                                 ax.get_legend_handles_labels()[1])))
        self.assertTrue(isinstance(isets, dict),
                        msg='Expect dictionary')
        self.assertEqual(isets['ciset']['limits'], [95])
        self.assertEqual(isets['piset']['limits'], [95])
        plt.close()
        plt.close()
        fig, ax, isets = uqp.plot_3d_intervals(
                intervals, data.xdata[0], limits=[95],
                adddata=True, ydata=data.xdata[0][:, 0],
                return_settings=True)
        self.assertEqual(ax.get_legend_handles_labels()[1],
                         ['Model', 'Data', '95% PI', '95% CI'],
                         msg=str('Strings should match: {}'.format(
                                 ax.get_legend_handles_labels()[1])))
        self.assertTrue(isinstance(isets, dict),
                        msg='Expect dictionary')
        self.assertEqual(isets['ciset']['limits'], [95])
        self.assertEqual(isets['piset']['limits'], [95])
        plt.close()
        