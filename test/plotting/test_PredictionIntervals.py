# -*- coding: utf-8 -*-

"""
Created on Mon Jan. 15, 2018

Description: This file contains a series of utilities designed to test the
features in the 'PredictionIntervals.py" package of the pymcmcstat module.  The
functions tested include:
    - empirical_quantiles(x, p = np.array([0.25, 0.5, 0.75])):

@author: prmiles
"""
from pymcmcstat.plotting.PredictionIntervals import PredictionIntervals
from pymcmcstat.utilities.progressbar import progress_bar
import test.general_functions as gf
import matplotlib.pyplot as plt
import unittest
from mock import patch
import numpy as np

# --------------------------------------------
class Observation_Sample_Test(unittest.TestCase):
    def test_does_observation_sample_unknown_sstype_cause_system_exit(self):
        PI = PredictionIntervals()
        s2elem = np.array([[2.0]])
        ypred = np.linspace(2.0, 3.0, num = 5)
        ypred = ypred.reshape(5,1)
        sstype = 4
        with self.assertRaises(SystemExit, msg = 'Unrecognized sstype'):
            PI._observation_sample(s2elem, ypred, sstype)

        sstype = 0
        opred = PI._observation_sample(s2elem, ypred, sstype)
        self.assertEqual(opred.shape, ypred.shape, msg = 'Shapes should match')
        
    def test_does_observation_sample_wrong_size_s2elem_break_right_size_array(self):
        PI = PredictionIntervals()
        s2elem = np.array([[2.0, 1.0]])
        ypred = np.linspace(2.0, 3.0, num = 5)
        ypred = ypred.reshape(5,1)
        sstype = 0
        with self.assertRaises(SystemExit, msg = 'Mismatched size for s2chain and model return'):
            PI._observation_sample(s2elem, ypred, sstype)
            
    def test_does_observation_sample_2_column_ypred_right_size_array(self):
        PI = PredictionIntervals()
        ypred = np.linspace(2.0, 3.0, num = 10)
        ypred = ypred.reshape(5,2)
        
        s2elem = np.array([[2.0, 1.0]])
        sstype = 0
        opred = PI._observation_sample(s2elem, ypred, sstype)
        self.assertEqual(opred.shape, ypred.shape, msg = 'Shapes are compatible')
        
        s2elem = np.array([[2.0]])
        opred = PI._observation_sample(s2elem, ypred, sstype)
        self.assertEqual(opred.shape, ypred.shape, msg = 'Shapes are compatible')
        
        sstype = 1
        opred = PI._observation_sample(s2elem, ypred, sstype)
        self.assertEqual(opred.shape, ypred.shape, msg = 'Shapes are compatible')
        
        sstype = 2
        opred = PI._observation_sample(s2elem, ypred, sstype)
        self.assertEqual(opred.shape, ypred.shape, msg = 'Shapes are compatible')
        
    def test_does_observation_sample_off_s2elem_greater_than_1_cause_system_exit(self):
        PI = PredictionIntervals()
        s2elem = np.array([[2.0, 1.0]])
        ypred = np.linspace(2.0, 3.0, num = 15)
        ypred = ypred.reshape(5,3)
        sstype = 0
        with self.assertRaises(SystemExit):
            PI._observation_sample(s2elem, ypred, sstype)

# --------------------------------------------
class AnalyzeDataStructure(unittest.TestCase):
    def test_basic_ds(self):
        PI = PredictionIntervals()
        DS = gf.basic_data_structure()
        ndatabatches, ncols = PI._analyze_data_structure(data = DS)
        self.assertEqual(ndatabatches, 1, msg = 'Expect 1 batch')
        self.assertEqual(ncols, [1], msg = 'Expect [1]')
        
    def test_ds_2_nbatch(self):
        PI = PredictionIntervals()
        DS = gf.non_basic_data_structure()
        ndatabatches, ncols = PI._analyze_data_structure(data = DS)
        self.assertEqual(ndatabatches, 2, msg = 'Expect 2 batches')
        self.assertEqual(ncols, [1, 2], msg = 'Expect [1, 1]')
        
    def test_basic_ds_shape_0(self):
        PI = PredictionIntervals()
        DS = gf.basic_data_structure()
        DS.shape = [(100,)] # remove column - code should add it back
        ndatabatches, ncols = PI._analyze_data_structure(data = DS)
        self.assertEqual(ndatabatches, 1, msg = 'Expect 1 batch')
        self.assertEqual(ncols, [1], msg = 'Expect [1]')
# --------------------------------------------
class SetupDataStructureForPrediction(unittest.TestCase):
    def test_basic_datapred(self):
        PI = PredictionIntervals()
        DS = gf.basic_data_structure()
        datapred = PI._setup_data_structure_for_prediction(data = DS, ndatabatches = 1)
        self.assertTrue(np.array_equal(datapred[0].xdata[0], DS.xdata[0]), msg = 'Arrays should match')
        self.assertTrue(np.array_equal(datapred[0].ydata[0], DS.ydata[0]), msg = 'Arrays should match')
        
    def test_non_basic_datapred(self):
        PI = PredictionIntervals()
        DS = gf.non_basic_data_structure()
        datapred = PI._setup_data_structure_for_prediction(data = DS, ndatabatches = 2)
        self.assertTrue(np.array_equal(datapred[0].xdata[0], DS.xdata[0]), msg = 'Arrays should match')
        self.assertTrue(np.array_equal(datapred[0].ydata[0], DS.ydata[0]), msg = 'Arrays should match')
        self.assertTrue(np.array_equal(datapred[1].xdata[0], DS.xdata[1]), msg = 'Arrays should match')
        self.assertTrue(np.array_equal(datapred[1].ydata[0], DS.ydata[1]), msg = 'Arrays should match')
# --------------------------------------------
class AssignFeaturesFromResultsStructure(unittest.TestCase):
    def check_dictionary(self, check_these, PID, results):
        for ct in check_these:
            self.assertTrue(np.array_equal(PID[str('_PredictionIntervals__{}'.format(ct))], results[ct]), msg = str('Arrays should match: {}'.format(ct)))
        self.assertTrue(np.array_equal(PID['_PredictionIntervals__nbatch'], results['model_settings']['nbatch']), msg = 'Arrays should match: nbatch')
        
    def test_feature_assignment(self):
        PI = PredictionIntervals()
        results = gf.setup_pseudo_results()
        PI._assign_features_from_results_structure(results = results)
        PID = PI.__dict__
        check_these = ['chain', 's2chain', 'parind', 'local', 'theta', 'sstype']
        self.check_dictionary(check_these, PID, results)
        
    def test_second_feature_assignment(self):
        PI = PredictionIntervals()
        results = gf.setup_pseudo_results()
        results = gf.removekey(results, 'sstype')
        PI._assign_features_from_results_structure(results = results)
        PID = PI.__dict__
        check_these = ['chain', 's2chain', 'parind', 'local', 'theta']
        self.check_dictionary(check_these, PID, results)
        self.assertEqual(PID['_PredictionIntervals__sstype'], 0, msg = 'Should default to 0')
        
# --------------------------------------------
class DetermineShapeOfReponse(unittest.TestCase):
    def test_basic_modelfunction(self):
        PI = PredictionIntervals()
        DS = gf.basic_data_structure()
        datapred = PI._setup_data_structure_for_prediction(data = DS, ndatabatches = 1)
        nrow, ncol = PI._determine_shape_of_response(modelfunction = gf.predmodelfun, ndatabatches = 1, datapred = datapred, theta = [3.0, 5.0])
        self.assertEqual(nrow, [100], msg = 'Expect [100]')
        self.assertEqual(ncol, [1], msg = 'Expect [1]')
        
    def test_basic_modelfunction_list_nbatch_2(self):
        PI = PredictionIntervals()
        DS = gf.non_basic_data_structure()
        datapred = PI._setup_data_structure_for_prediction(data = DS, ndatabatches = 2)
        nrow, ncol = PI._determine_shape_of_response(modelfunction = [gf.predmodelfun, gf.predmodelfun], ndatabatches = 2, datapred = datapred, theta = [3.0, 5.0])
        self.assertEqual(nrow, [100, 100], msg = 'Expect [100, 100]')
        self.assertEqual(ncol, [1, 1], msg = 'Expect [1, 1]')
        
    def test_non_basic_modelfunction(self):
        PI = PredictionIntervals()
        DS = gf.non_basic_data_structure()
        datapred = PI._setup_data_structure_for_prediction(data = DS, ndatabatches = 2)
        nrow, ncol = PI._determine_shape_of_response(modelfunction = gf.predmodelfun, ndatabatches = 2, datapred = datapred, theta = [3.0, 5.0])
        self.assertEqual(nrow, [100, 100], msg = 'Expect [100, 100]')
        self.assertEqual(ncol, [1, 1], msg = 'Expect [1, 1]')
 
# --------------------------------------------
class AnalyzeS2Chain(unittest.TestCase):
    def test_s2chain_index_n_eq_1(self):
        PI = PredictionIntervals()
        ndatabatches = 1
        s2chain = np.random.random_sample(size = (100,1))
        ncol = [1]
        s2chain_index = PI._analyze_s2chain(ndatabatches = ndatabatches, s2chain = s2chain, ncol = ncol)
        self.assertTrue(np.array_equal(s2chain_index, np.array([[0,1]])), msg = str('Arrays should match: {}'.format(s2chain_index)))
        
    def test_s2chain_index_n_neq_1_tc_eq_n(self):
        PI = PredictionIntervals()
        ndatabatches = 3
        s2chain = np.random.random_sample(size = (100,3))
        ncol = [1, 1, 1]
        s2chain_index = PI._analyze_s2chain(ndatabatches = ndatabatches, s2chain = s2chain, ncol = ncol)
        self.assertTrue(np.array_equal(s2chain_index, np.array([[0,1],[1,2],[2,3]])), msg = str('Arrays should match: {}'.format(s2chain_index)))
        
        ndatabatches = 3
        s2chain = np.random.random_sample(size = (100,4))
        ncol = [1, 2, 1]
        s2chain_index = PI._analyze_s2chain(ndatabatches = ndatabatches, s2chain = s2chain, ncol = ncol)
        self.assertTrue(np.array_equal(s2chain_index, np.array([[0,1],[1,3],[3,4]])), msg = str('Arrays should match: {}'.format(s2chain_index)))
        
    def test_s2chain_index_n_neq_1_tc_neq_n(self):
        PI = PredictionIntervals()
        ndatabatches = 3
        s2chain = np.random.random_sample(size = (100,3))
        ncol = [1, 2, 1]
        s2chain_index = PI._analyze_s2chain(ndatabatches = ndatabatches, s2chain = s2chain, ncol = ncol)
        self.assertTrue(np.array_equal(s2chain_index, np.array([[0,1],[1,2],[2,3]])), msg = str('Arrays should match: {}'.format(s2chain_index)))
        
    def test_s2chain_index_n_neq_1_tc_neq_n_neq_ndatabatches(self):
        PI = PredictionIntervals()
        ndatabatches = 5
        s2chain = np.random.random_sample(size = (100,3))
        ncol = [1, 2, 1]
        with self.assertRaises(SystemExit, msg = 'Unrecognized data structure'):
            PI._analyze_s2chain(ndatabatches = ndatabatches, s2chain = s2chain, ncol = ncol)

# --------------------------------------------
class SetupSstype(unittest.TestCase):
    def test_setup_sstype(self):
        PI = PredictionIntervals()
        PI._PredictionIntervals__sstype = 3
        sstype = PI._setup_sstype(sstype = None)
        self.assertEqual(sstype, PI._PredictionIntervals__sstype, msg = 'Expected 3')
        sstype = PI._setup_sstype(sstype = 2)
        self.assertEqual(sstype, 0, msg = 'Expected 0')

# --------------------------------------------
class CheckNsample(unittest.TestCase):
    def test_check_nsample(self):
        PI = PredictionIntervals()
        nsample = PI._check_nsample(nsample = 400, nsimu = 500)
        self.assertEqual(nsample, 400, msg = 'Expect 400')
        nsample = PI._check_nsample(nsample = None, nsimu = 500)
        self.assertEqual(nsample, 500, msg = 'Expect 500')
# --------------------------------------------
class DefineSamplePoints(unittest.TestCase):
    def test_define_sample_points_nsample_gt_nsimu(self):
        PI = PredictionIntervals()
        iisample, nsample = PI._define_sample_points(nsample = 1000, nsimu = 500)
        self.assertEqual(iisample, range(500), msg = 'Expect range(500)')
        self.assertEqual(nsample, 500, msg = 'Expect nsample updated to 500')
        
    @patch('numpy.random.rand')
    def test_define_sample_points_nsample_lte_nsimu(self, mock_rand):
        PI = PredictionIntervals()
        aa = np.random.rand([400,1])
        mock_rand.return_value = aa
        iisample, nsample = PI._define_sample_points(nsample = 400, nsimu = 500)
        self.assertTrue(np.array_equal(iisample, np.ceil(aa*500) - 1), msg = 'Expect range(500)')
        self.assertEqual(nsample, 400, msg = 'Expect nsample to stay 400')
        
# --------------------------------------------
class InitializePlotFeatures(unittest.TestCase):
    def test_initialize_plot_features(self):
        PI = PredictionIntervals()
        htmp, ax = PI._initialize_plot_features(ii = 0, jj = 0, ny = 1, figsizeinches = [10, 11])
        self.assertEqual(htmp.get_figwidth(), 10.0, msg = 'Figure width is 10in')
        self.assertEqual(htmp.get_figheight(), 11.0, msg = 'Figure height is 11in')
        self.assertEqual(htmp.get_label(), 'Batch # 0 | Column # 0', msg = 'Strings should match')
        self.assertEqual(ax.get_label(), '', msg = 'Expect empty string')
        
# --------------------------------------------
class AddBatchColumnTitle(unittest.TestCase):
    def test_add_batch_column_title_nbatch_gt_1(self):
        PI = PredictionIntervals()
        htmp, ax = PI._initialize_plot_features(ii = 0, jj = 0, ny = 1, figsizeinches = [10, 11])
        PI._add_batch_column_title(nbatch = 2, ny = 1, ii = 0, jj = 0)
        self.assertEqual(ax.get_title(), 'Batch #0, Column #0', msg = 'Strings should match')
        self.assertEqual(htmp.get_figwidth(), 10.0, msg = 'Expect 10.0')
        self.assertEqual(htmp.get_figheight(), 11.0, msg = 'Expect 11.0')
                         
    def test_add_batch_column_title_ny_gt_1(self):
        PI = PredictionIntervals()
        htmp, ax = PI._initialize_plot_features(ii = 0, jj = 0, ny = 1, figsizeinches = [10, 11])
        PI._add_batch_column_title(nbatch = 1, ny = 2, ii = 0, jj = 0)
        self.assertEqual(ax.get_title(), 'Column #0', msg = 'Strings should match')
        self.assertEqual(htmp.get_figwidth(), 10.0, msg = 'Expect 10.0')
        self.assertEqual(htmp.get_figheight(), 11.0, msg = 'Expect 11.0')
                         
# --------------------------------------------
class SetupLabels(unittest.TestCase):
    def test_setup_labels_pi_none_nlines_2(self):
        PI = PredictionIntervals()
        clabels, plabels = PI._setup_labels(prediction_intervals = None, nlines = 2)
        self.assertEqual(clabels, ['99% CI', '95% CI', '90% CI', '50% CI'], msg = 'String should match')
        self.assertEqual(plabels, ['95% PI'], msg = 'String should match')
        
    def test_setup_labels_pi_not_none_nlines_2(self):
        PI = PredictionIntervals()
        clabels, plabels = PI._setup_labels(prediction_intervals = 0, nlines = 2)
        self.assertEqual(clabels, ['95% CI'], msg = 'String should match')
        self.assertEqual(plabels, ['95% PI'], msg = 'String should match')
        
    def test_setup_labels_pi_none_nlines_1(self):
        PI = PredictionIntervals()
        clabels, plabels = PI._setup_labels(prediction_intervals = None, nlines = 1)
        self.assertEqual(clabels, ['95% CI'], msg = 'String should match')
        self.assertEqual(plabels, ['95% PI'], msg = 'String should match')
        
# --------------------------------------------
class CheckPIFlag(unittest.TestCase):
    def test_check_pi_flag_true_0(self):
        PI = PredictionIntervals()
        prediction_intervals = PI._check_prediction_interval_flag(plot_pred_int = True, prediction_intervals = 0)
        self.assertEqual(prediction_intervals, 0, msg = 'Expect 0')
        
    def test_check_pi_flag_true_none(self):
        PI = PredictionIntervals()
        prediction_intervals = PI._check_prediction_interval_flag(plot_pred_int = True, prediction_intervals = None)
        self.assertEqual(prediction_intervals, None, msg = 'Expect None')
        
    def test_check_pi_flag_false_0(self):
        PI = PredictionIntervals()
        prediction_intervals = PI._check_prediction_interval_flag(plot_pred_int = False, prediction_intervals = 0)
        self.assertEqual(prediction_intervals, None, msg = 'Expect None')
        
    def test_check_pi_flag_false_none(self):
        PI = PredictionIntervals()
        prediction_intervals = PI._check_prediction_interval_flag(plot_pred_int = False, prediction_intervals = None)
        self.assertEqual(prediction_intervals, None, msg = 'Expect None')
        
        
# --------------------------------------------
class SetupIntervalLimits(unittest.TestCase):
    def test_setup_int_lims_s2chain_none(self):
        PI = PredictionIntervals()
        lims = PI._setup_interval_limits(s2chain = None)
        self.assertTrue(np.array_equal(lims, np.array([0.005,0.025,0.05,0.25,0.5,0.75,0.9,0.975,0.995])), msg = str('Arrays should match: {}'.format(lims)))
        
    def test_setup_int_lims_s2chain_not_none(self):
        PI = PredictionIntervals()
        lims = PI._setup_interval_limits(s2chain = 1)
        self.assertTrue(np.array_equal(lims, np.array([0.025, 0.5, 0.975])), msg = str('Arrays should match: {}'.format(lims)))
        
# --------------------------------------------
class SetupCountingMetrics(unittest.TestCase):
    def test_setup_counting_metrics(self):
        PI = PredictionIntervals()
        ci = gf.setup_pseudo_ci()
        nbatch, nn, nlines = PI._setup_counting_metrics(credible_intervals = ci)
        self.assertEqual(nbatch, 1, msg = 'Expect nbatch = 1')
        self.assertEqual(nn, 2, msg = 'Expect nn = 2')
        self.assertEqual(nlines, 1, msg = 'Expect nlines = 1')

# --------------------------------------------
class SetupIntervalPlotting(unittest.TestCase):
    def check_these_settings(self, infigsizeinches, expected):
        PI = PredictionIntervals()
        ci = gf.setup_pseudo_ci()
        prediction_intervals, figsizeinches, nbatch, nn, clabels, plabels = PI._setup_interval_plotting(plot_pred_int = True, prediction_intervals = None, credible_intervals = ci, figsizeinches = infigsizeinches)
        self.assertEqual(figsizeinches, expected, msg = str('Expect {}'.format(expected)))
        self.assertEqual(prediction_intervals, None, msg = 'Expect None')
        self.assertEqual(nbatch, 1, msg = 'Expect nbatch = 1')
        self.assertEqual(nn, 2, msg = 'Expect nn = 2')
        self.assertEqual(clabels, ['95% CI'], msg = 'String should match')
        self.assertEqual(plabels, ['95% PI'], msg = 'String should match')
        
    def test_setup_interval_plotting(self):
        self.check_these_settings(infigsizeinches = [10, 11], expected = [10, 11])
        self.check_these_settings(infigsizeinches = None, expected = [7,5])

# --------------------------------------------
class SetupGenerationRequirements(unittest.TestCase):
    @classmethod
    def setup_generation(cls):
        aa = np.random.rand([400,1])
        PI = PredictionIntervals()
        results = gf.setup_pseudo_results()
        PI._assign_features_from_results_structure(results = results)
        return PI, aa, results
        
    def common_checks(self, results, chain, sstype, nsample, iisample):
        self.assertTrue(np.array_equal(chain, results['chain']), msg = str('Arrays should match: {}'.format(chain)))
        self.assertEqual(sstype, 0, msg = 'Expect 0')
        self.assertEqual(nsample, 100, msg = 'Expect nsample to go to 100')
        self.assertTrue(np.array_equal(iisample, range(100)), msg = str('Arrays should match: {}'.format(iisample)))

    @patch('numpy.random.rand')
    def test_setup_generation(self, mock_rand):
        PI, aa, results = self.setup_generation()
        mock_rand.return_value = aa
        chain, s2chain, lims, sstype, nsample, iisample = PI._setup_generation_requirements(nsample = 400, calc_pred_int = False, sstype = 0)
        
        self.assertTrue(np.array_equal(lims, np.array([0.005,0.025,0.05,0.25,0.5,0.75,0.9,0.975,0.995])), msg = str('Arrays should match: {}'.format(lims)))
        self.assertEqual(s2chain, None, msg = 'Expect None')

        self.common_checks(results, chain, sstype, nsample, iisample)
        
    @patch('numpy.random.rand')
    def test_setup_generation_with_pi(self, mock_rand):
        PI, aa, results = self.setup_generation()
        mock_rand.return_value = aa
        chain, s2chain, lims, sstype, nsample, iisample = PI._setup_generation_requirements(nsample = 400, calc_pred_int = True, sstype = 0)

        self.assertTrue(np.array_equal(lims, np.array([0.025, 0.5, 0.975])), msg = str('Arrays should match: {}'.format(lims)))
        self.assertTrue(np.array_equal(s2chain, results['s2chain']), msg = str('Arrays should match: {}'.format(s2chain)))
        
        self.common_checks(results, chain, sstype, nsample, iisample)
        
# --------------------------------------------
class SetupIntervalii(unittest.TestCase):
    @classmethod
    def setup_interval(cls):
        PI = PredictionIntervals()
        DS = gf.basic_data_structure()
        datapred = PI._setup_data_structure_for_prediction(data = DS, ndatabatches = 1)
        return PI, datapred

    def common_checks(self, datapredii, datapred, nrow, ncol, modelfun, predmodelfun, test):
        self.assertTrue(np.array_equal(datapredii.xdata[0], datapred[0].xdata[0]), msg = 'Arrays should match')
        self.assertTrue(np.array_equal(datapredii.ydata[0], datapred[0].ydata[0]), msg = 'Arrays should match')
        self.assertEqual(nrow, 100, msg = 'Expect nrow = 100')
        self.assertEqual(ncol, 1, msg = 'Expect ncol = 1')
        self.assertEqual(modelfun, predmodelfun, msg = 'Functions should match')
        self.assertTrue(np.array_equal(test, np.array([True, True])), msg = str('Arrays should match: {}'.format(test)))
        
    def test_setup_predii_nonlist_mdlfun(self):
        PI, datapred = self.setup_interval()
        datapredii, nrow, ncol, modelfun, test = PI._setup_interval_ii(ii = 0, datapred = datapred, nrow = [100], ncol = [1], modelfunction = gf.predmodelfun, local = np.array([0, 0]))
        self.common_checks(datapredii, datapred, nrow, ncol, modelfun, gf.predmodelfun, test)
        
    def test_setup_predii_list_mdlfun(self):
        PI, datapred = self.setup_interval()
        datapredii, nrow, ncol, modelfun, test = PI._setup_interval_ii(ii = 0, datapred = datapred, nrow = [100], ncol = [1], modelfunction = [gf.predmodelfun], local = np.array([0, 0]))
        self.common_checks(datapredii, datapred, nrow, ncol, modelfun, gf.predmodelfun, test)
        
# --------------------------------------------
def gq_settings():
    PI = PredictionIntervals()
    nsample = 500
    nrow = 100
    ncol = 1
    return PI, nsample, nrow, ncol

class GenerateQuantiles(unittest.TestCase):
    def test_generate_quantiles(self):
        PI, nsample, nrow, ncol = gq_settings()
        ysave = np.zeros([nsample, nrow, ncol])
        lims = np.array([0.025, 0.5, 0.975])
        cq = PI._generate_quantiles(response = ysave, lims = lims, ncol = ncol)
        self.assertTrue(isinstance(cq, list), msg = 'Expect list')
        self.assertEqual(cq[0].shape, (3,100), msg = 'Expect shape = (3, 100)')
        
    def test_generate_quantiles_s2chain_not_none(self):
        PI, nsample, nrow, ncol = gq_settings()
        ysave = np.zeros([nsample, nrow, ncol])
        osave = np.zeros([nsample, nrow, ncol])
        lims = np.array([0.025, 0.5, 0.975])
        cq = PI._generate_quantiles(response = ysave, lims = lims, ncol = ncol)
        pq = PI._generate_quantiles(response = osave, lims = lims, ncol = ncol)
        self.assertTrue(isinstance(cq, list), msg = 'Expect list')
        self.assertTrue(isinstance(pq, list), msg = 'Expect list')
        self.assertEqual(cq[0].shape, (3,100), msg = 'Expect shape = (3, 100)')
        self.assertEqual(pq[0].shape, (3,100), msg = 'Expect shape = (3, 100)')

# --------------------------------------------
def cc_setup(calc_pred_int = True):
    PI = PredictionIntervals()
    results = gf.setup_pseudo_results()
    PI._assign_features_from_results_structure(results = results)
    testchain, s2chain, lims, sstype, nsample, iisample = PI._setup_generation_requirements(nsample = 400, calc_pred_int = calc_pred_int, sstype = 0)
    DS = gf.basic_data_structure()
    datapred = PI._setup_data_structure_for_prediction(data = DS, ndatabatches = 1)
    return PI, results, testchain, s2chain, lims, sstype, nsample, iisample, datapred

class CalcCredii(unittest.TestCase):
    def test_calc_credii(self):
        PI, __, testchain, __, lims, sstype, nsample, iisample, datapred = cc_setup()
        
        ysave = PI._calc_credible_ii(testchain = testchain, nrow = 100, ncol = 1, waitbar = False, test = np.array([True, True]), modelfun = gf.predmodelfun, datapredii = datapred[0])
        self.assertTrue(isinstance(ysave, np.ndarray), msg = 'Expect array')
        self.assertEqual(ysave.shape[0], 100, msg = 'Expect 1st dim = 100')
        
    def test_calc_credii_with_waitbar(self):
        PI, __, testchain, __, lims, sstype, nsample, iisample, datapred = cc_setup()
        
        PI._PredictionIntervals__wbarstatus = progress_bar(iters = 200)
            
        ysave = PI._calc_credible_ii(testchain = testchain, nrow = 100, ncol = 1, waitbar = True, test = np.array([True, True]), modelfun = gf.predmodelfun, datapredii = datapred[0])
        self.assertTrue(isinstance(ysave, np.ndarray), msg = 'Expect array')
        self.assertEqual(ysave.shape[0], 100, msg = 'Expect 1st dim = 100')
        self.assertEqual(PI._PredictionIntervals__wbarstatus.percentage(3), 1.5, msg = 'Expect 1.5')
        
# --------------------------------------------
class CalcPredii(unittest.TestCase):
    def common_checks(self, ysave, osave):
        self.assertTrue(isinstance(ysave, np.ndarray), msg = 'Expect array')
        self.assertTrue(isinstance(osave, np.ndarray), msg = 'Expect array')
        self.assertEqual(ysave.shape[0], 100, msg = 'Expect 1st dim = 100')
        self.assertEqual(osave.shape[0], 100, msg = 'Expect 1st dim = 100')
        
    def test_calc_predii(self):
        PI, __, testchain, tests2chain, __, sstype, nsample, iisample, datapred = cc_setup()
                
        ysave, osave = PI._calc_credible_and_prediction_ii(testchain = testchain, tests2chain = tests2chain, nrow = 100, ncol = 1, waitbar = False, sstype = 0, test = np.array([True, True]), modelfun = gf.predmodelfun, datapredii = datapred[0])
        self.common_checks(ysave, osave)
        
    def test_calc_predii_with_waitbar(self):
        PI, __, testchain, tests2chain, __, sstype, nsample, iisample, datapred = cc_setup()
        
        PI._PredictionIntervals__wbarstatus = progress_bar(iters = 200)
        
        ysave, osave = PI._calc_credible_and_prediction_ii(testchain = testchain, tests2chain = tests2chain, nrow = 100, ncol = 1, waitbar = True, sstype = 0, test = np.array([True, True]), modelfun = gf.predmodelfun, datapredii = datapred[0])
        self.common_checks(ysave, osave)
        self.assertEqual(PI._PredictionIntervals__wbarstatus.percentage(3), 1.5, msg = 'Expect 1.5')
        
    def test_calc_predii_s2chain_none(self):
        PI, results, testchain, tests2chain, __, sstype, nsample, iisample, datapred = cc_setup(calc_pred_int = False)
        tests2chain = None
        
        with self.assertRaises(TypeError, msg = 'This function should not be called in tests2chain is None'):
            PI._calc_credible_and_prediction_ii(testchain = testchain, tests2chain = tests2chain, nrow = 100, ncol = 1, waitbar = False, sstype = 0, test = np.array([True, True]), modelfun = gf.predmodelfun, datapredii = datapred[0])
                   
    def test_calc_predii_s2chain_tran(self):
        PI, results, testchain, tests2chain, __, sstype, nsample, iisample, datapred = cc_setup(calc_pred_int = False)
        tests2chain = np.random.random_sample(size = (1,100))
        
        with self.assertRaises(SystemExit, msg = 'Unknown structure'):
            PI._calc_credible_and_prediction_ii(testchain = testchain, tests2chain = tests2chain, nrow = 100, ncol = 1, waitbar = False, sstype = 0, test = np.array([True, True]), modelfun = gf.predmodelfun, datapredii = datapred[0])
            
# --------------------------------------------
def pi_setup():
    PI = PredictionIntervals()
    results = gf.setup_pseudo_results()
    DS = gf.basic_data_structure()
    PI.setup_prediction_interval_calculation(results = results, data = DS, modelfunction = gf.predmodelfun)
    return PI, results

class SetupPredictionIntervalCalculation(unittest.TestCase):
    def test_setup_pi_calc(self):
        PI, results = pi_setup()
        self.assertEqual(PI._PredictionIntervals__ndatabatches, 1, msg = 'Expect ndatabatches = 1')
        self.assertEqual(PI.modelfunction, gf.predmodelfun, msg = 'Functions should match')
        self.assertTrue(np.array_equal(PI._PredictionIntervals__s2chain_index, np.array([[0,1]])), msg = str('Arrays should match: {}'.format(PI._PredictionIntervals__s2chain_index)))
        self.assertTrue(np.array_equal(PI._PredictionIntervals__chain, results['chain']), msg = 'Arrays should match')
        
    def test_setup_pi_calc_with_s2chain_none(self):
        PI = PredictionIntervals()
        DS = gf.basic_data_structure()
        results = gf.setup_pseudo_results()
        results['s2chain'] = None
        PI.setup_prediction_interval_calculation(results = results, data = DS, modelfunction = gf.predmodelfun)
        self.assertEqual(PI._PredictionIntervals__ndatabatches, 1, msg = 'Expect ndatabatches = 1')
        self.assertEqual(PI.modelfunction, gf.predmodelfun, msg = 'Functions should match')
        self.assertFalse(hasattr(PI, '_PredictionIntervals__s2chain_index'), msg = 'Expect False')
        self.assertTrue(np.array_equal(PI._PredictionIntervals__chain, results['chain']), msg = 'Arrays should match')

# --------------------------------------------
class CalculateCIForDataSets(unittest.TestCase):
    def test_calc_ci_for_ds(self):
        PI, results = pi_setup()
        PI._assign_features_from_results_structure(results = results)
        lims = np.array([0.025, 0.5, 0.975])
        testchain = np.random.random_sample(size = (100,2))
        ci = PI._calculate_ci_for_data_sets(testchain = testchain, waitbar = False, lims = lims)
        
        self.assertTrue(isinstance(ci, list), msg = 'Expect list return')
        self.assertEqual(ci[0][0].shape[0], 3, msg = 'Expect 1st dim = 3')
        self.assertEqual(ci[0][0].shape[1], 100, msg = 'Expect 2nd dim = 100')
        
        
# --------------------------------------------
class CalculateCIandPIForDataSets(unittest.TestCase):
    def test_calc_ci_and_pi_for_ds(self):
        PI, results = pi_setup()
        PI._assign_features_from_results_structure(results = results)
        lims = np.array([0.025, 0.5, 0.975])
        testchain = np.random.random_sample(size = (100,2))
        s2chain = np.random.random_sample(size = (100,1))
        sstype = 0
        iisample, nsample = PI._define_sample_points(nsample = 400, nsimu = 100)
        ci, pi = PI._calculate_ci_and_pi_for_data_sets(testchain = testchain, s2chain = s2chain, iisample = iisample, waitbar = False, sstype = sstype, lims = lims)
        
        self.assertTrue(isinstance(ci, list), msg = 'Expect list return')
        self.assertEqual(ci[0][0].shape[0], 3, msg = 'Expect 1st dim = 3')
        self.assertEqual(ci[0][0].shape[1], 100, msg = 'Expect 2nd dim = 100')
        self.assertTrue(isinstance(pi, list), msg = 'Expect list return')
        self.assertEqual(pi[0][0].shape[0], 3, msg = 'Expect 1st dim = 3')
        self.assertEqual(pi[0][0].shape[1], 100, msg = 'Expect 2nd dim = 100')
        self.assertEqual(nsample, 100, msg = 'Expect nsample -> 100')
        
# --------------------------------------------
class GeneratePredictionIntervals(unittest.TestCase):
    def common_set_1(self, cint, pint):
        self.assertTrue(isinstance(cint, list), msg = 'Expect list return')
        self.assertEqual(cint[0][0].shape[0], 9, msg = 'Expect 1st dim = 9')
        self.assertEqual(cint[0][0].shape[1], 100, msg = 'Expect 2nd dim = 100')
        self.assertEqual(pint, None, msg = 'Expect None')
        
    def test_generate_credible_intervals(self):
        PI = PredictionIntervals()
        results = gf.setup_pseudo_results()
        results['s2chain'] = None
        DS = gf.basic_data_structure()
        PI.setup_prediction_interval_calculation(results = results, data = DS, modelfunction = gf.predmodelfun)
        PI.generate_prediction_intervals(sstype = None, nsample = 500, calc_pred_int = False, waitbar = False)
        cint = PI.intervals['credible_intervals']
        pint = PI.intervals['prediction_intervals']
        self.common_set_1(cint, pint)
        
    def test_generate_credible_intervals_with_waitbar(self):
        PI = PredictionIntervals()
        results = gf.setup_pseudo_results()
        results['s2chain'] = None
        DS = gf.basic_data_structure()
        PI.setup_prediction_interval_calculation(results = results, data = DS, modelfunction = gf.predmodelfun)
        PI.generate_prediction_intervals(sstype = None, nsample = 500, calc_pred_int = False, waitbar = True)
        cint = PI.intervals['credible_intervals']
        pint = PI.intervals['prediction_intervals']
        self.common_set_1(cint, pint)
        self.assertEqual(PI._PredictionIntervals__wbarstatus.percentage(3), 3.0, msg = 'Expect 3.0')
        
    def test_generate_credible_and_prediction_intervals(self):
        PI, __ = pi_setup()
        PI.generate_prediction_intervals(sstype = 0, nsample = 500, calc_pred_int = True, waitbar = False)
        cint = PI.intervals['credible_intervals']
        pint = PI.intervals['prediction_intervals']
        self.assertTrue(isinstance(cint, list), msg = 'Expect list return')
        self.assertEqual(cint[0][0].shape[0], 3, msg = 'Expect 1st dim = 3')
        self.assertEqual(cint[0][0].shape[1], 100, msg = 'Expect 2nd dim = 100')
        self.assertTrue(isinstance(pint, list), msg = 'Expect list return')
        self.assertEqual(pint[0][0].shape[0], 3, msg = 'Expect 1st dim = 3')
        self.assertEqual(pint[0][0].shape[1], 100, msg = 'Expect 2nd dim = 100')
        
    def test_generate_credible_and_prediction_intervals_with_pi_off(self):
        PI, results = pi_setup()
        PI.generate_prediction_intervals(sstype = 0, nsample = 500, calc_pred_int = False, waitbar = False)
        cint = PI.intervals['credible_intervals']
        pint = PI.intervals['prediction_intervals']
        self.common_set_1(cint, pint)

# --------------------------------------------
class SetupDisplaySettings(unittest.TestCase):
    def test_setup_display_settings(self):
        PI = PredictionIntervals()
        model_display = {'label': 'hello'}
        data_display = {'linewidth': 7}
        interval_display = {'edgecolor': 'b'}
        intd, modd, datd = PI._setup_display_settings(interval_display = interval_display, model_display = model_display, data_display = data_display)
        self.assertEqual(modd['label'], model_display['label'], msg = 'Expect label to match')
        self.assertEqual(intd['edgecolor'], interval_display['edgecolor'], msg = 'Expect edgecolor to match')
        self.assertEqual(datd['linewidth'], data_display['linewidth'], msg = 'Expect linewidth to match')
        
# --------------------------------------------
class PlotPredictionIntervals(unittest.TestCase):
    def common_checks(self, fighandle, axhandle):
        self.assertTrue(isinstance(fighandle, list), msg = 'Expect list return')
        self.assertTrue(isinstance(axhandle, list), msg = 'Expect list return')
        self.assertEqual(fighandle[0].get_label(), 'Batch # 0 | Column # 0', msg = str('Strings should match: {}'.format(fighandle[0].get_label())))

    @classmethod
    def setup_complex_pi(cls):
        PI = PredictionIntervals()
        results = gf.setup_pseudo_results()
        results['s2chain'] = None
        DS = gf.basic_data_structure()
        PI.setup_prediction_interval_calculation(results = results, data = DS, modelfunction = gf.predmodelfun)
        PI.generate_prediction_intervals(sstype = 0, nsample = 500, calc_pred_int = False, waitbar = False)
        return PI
    
    def test_plotting_credible_and_prediction_intervals(self):
        PI, results = pi_setup()
        PI.generate_prediction_intervals(sstype = 0, nsample = 500, calc_pred_int = True, waitbar = False)
        fighandle, axhandle = PI.plot_prediction_intervals(plot_pred_int = True, adddata = False, addlegend = True)
        self.common_checks(fighandle, axhandle)
        self.assertEqual(axhandle[0].get_legend_handles_labels()[1], ['model', '95% PI', '95% CI'], msg = str('Strings should match: {}'.format(axhandle[0].get_legend_handles_labels()[1])))
        plt.close()
        
    def test_plotting_credible_intervals(self):
        PI = self.setup_complex_pi()
        fighandle, axhandle = PI.plot_prediction_intervals(plot_pred_int = False, adddata = False, addlegend = True)
        
        self.common_checks(fighandle, axhandle)
        self.assertEqual(axhandle[0].get_legend_handles_labels()[1], ['model', '99% CI', '95% CI', '90% CI', '50% CI'], msg = str('Strings should match: {}'.format(axhandle[0].get_legend_handles_labels()[1])))
        plt.close()
        
    def test_plotting_credible_intervals_with_data(self):
        PI = self.setup_complex_pi()
        fighandle, axhandle = PI.plot_prediction_intervals(plot_pred_int = False, adddata = True, addlegend = True)
        
        self.common_checks(fighandle, axhandle)
        self.assertEqual(axhandle[0].get_legend_handles_labels()[1], ['model', 'data', '99% CI', '95% CI', '90% CI', '50% CI'], msg = str('Strings should match: {}'.format(axhandle[0].get_legend_handles_labels()[1])))
        plt.close()