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
from pymcmcstat.settings.DataStructure import DataStructure
import unittest
from mock import patch
import numpy as np

def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

# define test model function
def predmodelfun(data, theta):
    m = theta[0]
    b = theta[1]
    nrow = data.xdata[0].shape[0]
    y = np.zeros([nrow,1])
    y[:,0] = m*data.xdata[0].reshape(nrow,) + b
    return y

def basic_data_structure():
    DS = DataStructure()
    x = np.random.random_sample(size = (100, 1))
    y = np.random.random_sample(size = (100, 1))
    DS.add_data_set(x = x, y = y)
    return DS

def non_basic_data_structure():
    DS = DataStructure()
    x = np.random.random_sample(size = (100, 1))
    y = np.random.random_sample(size = (100, 1))
    DS.add_data_set(x = x, y = y)
    x = np.random.random_sample(size = (100, 1))
    y = np.random.random_sample(size = (100, 2))
    DS.add_data_set(x = x, y = y)    
    return DS

def setup_pseudo_results():
    results = {
            'chain': np.random.random_sample(size = (100,2)),
            's2chain': np.random.random_sample(size = (100,1)),
            'sschain': np.random.random_sample(size = (100,1)),
            'parind': np.array([[0, 1]]),
            'local': np.array([[0, 0]]),
            'model_settings': {'nbatch': np.random.random_sample(size = (100,1))},
            'theta': np.random.random_sample(size = (2,)),
            'sstype': np.random.random_sample(size = (1,1)),
            }
    return results

def setup_pseudo_ci():
    ci = []
    ci1 = []
    ci1.append([np.random.random_sample(size = (100,)),np.random.random_sample(size = (100,)),np.random.random_sample(size = (100,))])
    ci.append(ci1)
    return ci
        
# --------------------------------------------
class Observation_Sample_Test(unittest.TestCase):
    
    def test_does_observation_sample_unknown_sstype_cause_system_exit(self):
        PI = PredictionIntervals()
        s2elem = np.array([[2.0]])
        ypred = np.linspace(2.0, 3.0, num = 5)
        ypred = ypred.reshape(5,1)
        sstype = 4
        with self.assertRaises(SystemExit):
            PI._observation_sample(s2elem, ypred, sstype)
            
    def test_does_observation_sample_return_right_size_array(self):
        PI = PredictionIntervals()
        s2elem = np.array([[2.0]])
        ypred = np.linspace(2.0, 3.0, num = 5)
        ypred = ypred.reshape(5,1)
        sstype = 0
        opred = PI._observation_sample(s2elem, ypred, sstype)
        self.assertEqual(opred.shape, ypred.shape)
        
    def test_does_observation_sample_wrong_size_s2elem_break_right_size_array(self):
        PI = PredictionIntervals()
        s2elem = np.array([[2.0, 1.0]])
        ypred = np.linspace(2.0, 3.0, num = 5)
        ypred = ypred.reshape(5,1)
        sstype = 0
        with self.assertRaises(SystemExit):
            PI._observation_sample(s2elem, ypred, sstype)
            
    def test_does_observation_sample_2_column_ypred_right_size_array(self):
        PI = PredictionIntervals()
        s2elem = np.array([[2.0, 1.0]])
        ypred = np.linspace(2.0, 3.0, num = 10)
        ypred = ypred.reshape(5,2)
        sstype = 0
        opred = PI._observation_sample(s2elem, ypred, sstype)
        self.assertEqual(opred.shape, ypred.shape)
        
    def test_does_observation_sample_2_column_ypred_with_1_s2elem_right_size_array(self):
        PI = PredictionIntervals()
        s2elem = np.array([[2.0]])
        ypred = np.linspace(2.0, 3.0, num = 10)
        ypred = ypred.reshape(5,2)
        sstype = 0
        opred = PI._observation_sample(s2elem, ypred, sstype)
        self.assertEqual(opred.shape, ypred.shape)
        
    def test_does_observation_sample_off_s2elem_greater_than_1_cause_system_exit(self):
        PI = PredictionIntervals()
        s2elem = np.array([[2.0, 1.0]])
        ypred = np.linspace(2.0, 3.0, num = 15)
        ypred = ypred.reshape(5,3)
        sstype = 0
        with self.assertRaises(SystemExit):
            PI._observation_sample(s2elem, ypred, sstype)
            
    def test_sstype_1(self):
        PI = PredictionIntervals()
        s2elem = np.array([[2.0]])
        ypred = np.linspace(2.0, 3.0, num = 10)
        ypred = ypred.reshape(5,2)
        sstype = 1
        opred = PI._observation_sample(s2elem, ypred, sstype)
        self.assertEqual(opred.shape, ypred.shape)
        
    def test_sstype_2(self):
        PI = PredictionIntervals()
        s2elem = np.array([[2.0]])
        ypred = np.linspace(2.0, 3.0, num = 10)
        ypred = ypred.reshape(5,2)
        sstype = 2
        opred = PI._observation_sample(s2elem, ypred, sstype)
        self.assertEqual(opred.shape, ypred.shape)
        
# --------------------------------------------
class AnalyzeDataStructure(unittest.TestCase):
    def test_basic_ds(self):
        PI = PredictionIntervals()
        DS = basic_data_structure()
        ndatabatches, nrows, ncols = PI._analyze_data_structure(data = DS)
        self.assertEqual(ndatabatches, 1, msg = 'Expect 1 batch')
        self.assertEqual(nrows, [100], msg = 'Expect [100]')
        self.assertEqual(ncols, [1], msg = 'Expect [1]')
        
    def test_ds_2_nbatch(self):
        PI = PredictionIntervals()
        DS = non_basic_data_structure()
        ndatabatches, nrows, ncols = PI._analyze_data_structure(data = DS)
        self.assertEqual(ndatabatches, 2, msg = 'Expect 2 batches')
        self.assertEqual(nrows, [100, 100], msg = 'Expect [100, 100]')
        self.assertEqual(ncols, [1, 2], msg = 'Expect [1, 1]')
        
    def test_basic_ds_shape_0(self):
        PI = PredictionIntervals()
        DS = basic_data_structure()
        DS.shape = [(100,)] # remove column - code should add it back
        ndatabatches, nrows, ncols = PI._analyze_data_structure(data = DS)
        self.assertEqual(ndatabatches, 1, msg = 'Expect 1 batch')
        self.assertEqual(nrows, [100], msg = 'Expect [100]')
        self.assertEqual(ncols, [1], msg = 'Expect [1]')
# --------------------------------------------
class SetupDataStructureForPrediction(unittest.TestCase):
    def test_basic_datapred(self):
        PI = PredictionIntervals()
        DS = basic_data_structure()
        datapred = PI._setup_data_structure_for_prediction(data = DS, ndatabatches = 1)
        self.assertTrue(np.array_equal(datapred[0].xdata[0], DS.xdata[0]), msg = 'Arrays should match')
        self.assertTrue(np.array_equal(datapred[0].ydata[0], DS.ydata[0]), msg = 'Arrays should match')
        
    def test_non_basic_datapred(self):
        PI = PredictionIntervals()
        DS = non_basic_data_structure()
        datapred = PI._setup_data_structure_for_prediction(data = DS, ndatabatches = 2)
        self.assertTrue(np.array_equal(datapred[0].xdata[0], DS.xdata[0]), msg = 'Arrays should match')
        self.assertTrue(np.array_equal(datapred[0].ydata[0], DS.ydata[0]), msg = 'Arrays should match')
        self.assertTrue(np.array_equal(datapred[1].xdata[0], DS.xdata[1]), msg = 'Arrays should match')
        self.assertTrue(np.array_equal(datapred[1].ydata[0], DS.ydata[1]), msg = 'Arrays should match')
# --------------------------------------------
class AssignFeaturesFromResultsStructure(unittest.TestCase):
    def test_feature_assignment(self):
        PI = PredictionIntervals()
        results = setup_pseudo_results()
        PI._assign_features_from_results_structure(results = results)
        PID = PI.__dict__
        check_these = ['chain', 's2chain', 'parind', 'local', 'theta', 'sstype']
        for ct in check_these:
            self.assertTrue(np.array_equal(PID[str('_PredictionIntervals__{}'.format(ct))], results[ct]), msg = str('Arrays should match: {}'.format(ct)))
        
        self.assertTrue(np.array_equal(PID['_PredictionIntervals__nbatch'], results['model_settings']['nbatch']), msg = 'Arrays should match: nbatch')
        
    def test_second_feature_assignment(self):
        PI = PredictionIntervals()
        results = setup_pseudo_results()
        results = removekey(results, 'sstype')
        PI._assign_features_from_results_structure(results = results)
        PID = PI.__dict__
        check_these = ['chain', 's2chain', 'parind', 'local', 'theta']
        for ct in check_these:
            self.assertTrue(np.array_equal(PID[str('_PredictionIntervals__{}'.format(ct))], results[ct]), msg = str('Arrays should match: {}'.format(ct)))
        
        self.assertTrue(np.array_equal(PID['_PredictionIntervals__nbatch'], results['model_settings']['nbatch']), msg = 'Arrays should match: nbatch')
        self.assertEqual(PID['_PredictionIntervals__sstype'], 0, msg = 'Should default to 0')
        
# --------------------------------------------
class DetermineShapeOfReponse(unittest.TestCase):
    def test_basic_modelfunction(self):
        PI = PredictionIntervals()
        DS = basic_data_structure()
        datapred = PI._setup_data_structure_for_prediction(data = DS, ndatabatches = 1)
        nrow, ncol = PI._determine_shape_of_response(modelfunction = predmodelfun, ndatabatches = 1, datapred = datapred, theta = [3.0, 5.0])
        self.assertEqual(nrow, [100], msg = 'Expect [100]')
        self.assertEqual(ncol, [1], msg = 'Expect [1]')
        
    def test_basic_modelfunction_list_nbatch_2(self):
        PI = PredictionIntervals()
        DS = non_basic_data_structure()
        datapred = PI._setup_data_structure_for_prediction(data = DS, ndatabatches = 2)
        nrow, ncol = PI._determine_shape_of_response(modelfunction = [predmodelfun, predmodelfun], ndatabatches = 2, datapred = datapred, theta = [3.0, 5.0])
        self.assertEqual(nrow, [100, 100], msg = 'Expect [100, 100]')
        self.assertEqual(ncol, [1, 1], msg = 'Expect [1, 1]')
        
    def test_non_basic_modelfunction(self):
        PI = PredictionIntervals()
        DS = non_basic_data_structure()
        datapred = PI._setup_data_structure_for_prediction(data = DS, ndatabatches = 2)
        nrow, ncol = PI._determine_shape_of_response(modelfunction = predmodelfun, ndatabatches = 2, datapred = datapred, theta = [3.0, 5.0])
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
                         
# --------------------------------------------
class AddBatchColumnTitle(unittest.TestCase):
    def test_add_batch_column_title_nbatch_gt_1(self):
        PI = PredictionIntervals()
        htmp, ax = PI._initialize_plot_features(ii = 0, jj = 0, ny = 1, figsizeinches = [10, 11])
        PI._add_batch_column_title(nbatch = 2, ny = 1, ii = 0, jj = 0)
        self.assertEqual(ax.get_title(), 'Batch #0, Column #0', msg = 'Strings should match')
                         
    def test_add_batch_column_title_ny_gt_1(self):
        PI = PredictionIntervals()
        htmp, ax = PI._initialize_plot_features(ii = 0, jj = 0, ny = 1, figsizeinches = [10, 11])
        PI._add_batch_column_title(nbatch = 1, ny = 2, ii = 0, jj = 0)
        self.assertEqual(ax.get_title(), 'Column #0', msg = 'Strings should match')
                         
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
        ci = setup_pseudo_ci()
        nbatch, nn, nlines = PI._setup_counting_metrics(credible_intervals = ci)
        self.assertEqual(nbatch, 1, msg = 'Expect nbatch = 1')
        self.assertEqual(nn, 2, msg = 'Expect nn = 2')
        self.assertEqual(nlines, 1, msg = 'Expect nlines = 1')

# --------------------------------------------
class SetupIntervalPlotting(unittest.TestCase):
    def test_setup_interval_plotting(self):
        PI = PredictionIntervals()
        ci = setup_pseudo_ci()
        prediction_intervals, figsizeinches, nbatch, nn, clabels, plabels = PI._setup_interval_plotting(plot_pred_int = True, prediction_intervals = None, credible_intervals = ci, figsizeinches = [10,11])
        self.assertEqual(prediction_intervals, None, msg = 'Expect None')
        self.assertEqual(figsizeinches, [10, 11], msg = 'Expect [10,11]')
        self.assertEqual(nbatch, 1, msg = 'Expect nbatch = 1')
        self.assertEqual(nn, 2, msg = 'Expect nn = 2')
        self.assertEqual(clabels, ['95% CI'], msg = 'String should match')
        self.assertEqual(plabels, ['95% PI'], msg = 'String should match')
        
    def test_setup_interval_plotting_default_figsize(self):
        PI = PredictionIntervals()
        ci = setup_pseudo_ci()
        prediction_intervals, figsizeinches, nbatch, nn, clabels, plabels = PI._setup_interval_plotting(plot_pred_int = True, prediction_intervals = None, credible_intervals = ci, figsizeinches = None)
        self.assertEqual(prediction_intervals, None, msg = 'Expect None')
        self.assertEqual(figsizeinches, [7, 5], msg = 'Expect [7,5]')
        self.assertEqual(nbatch, 1, msg = 'Expect nbatch = 1')
        self.assertEqual(nn, 2, msg = 'Expect nn = 2')
        self.assertEqual(clabels, ['95% CI'], msg = 'String should match')
        self.assertEqual(plabels, ['95% PI'], msg = 'String should match')

# --------------------------------------------
class SetupGenerationRequirements(unittest.TestCase):
    @patch('numpy.random.rand')
    def test_setup_generation(self, mock_rand):
        aa = np.random.rand([400,1])
        mock_rand.return_value = aa
        PI = PredictionIntervals()
        results = setup_pseudo_results()
        PI._assign_features_from_results_structure(results = results)
        chain, s2chain, nsimu, lims, sstype, nsample, iisample = PI._setup_generation_requirements(nsample = 400, calc_pred_int = False, sstype = 0)
        
        self.assertTrue(np.array_equal(chain, results['chain']), msg = str('Arrays should match: {}'.format(chain)))
        self.assertEqual(s2chain, None, msg = 'Expect None')
        self.assertEqual(nsimu, chain.shape[0], msg = 'Expect nsimu = chain.shape[0]')
        self.assertTrue(np.array_equal(lims, np.array([0.005,0.025,0.05,0.25,0.5,0.75,0.9,0.975,0.995])), msg = str('Arrays should match: {}'.format(lims)))
        self.assertEqual(sstype, 0, msg = 'Expect 0')
        self.assertEqual(nsample, 100, msg = 'Expect nsample to go to 100')
        self.assertTrue(np.array_equal(iisample, range(100)), msg = str('Arrays should match: {}'.format(iisample)))
        
    @patch('numpy.random.rand')
    def test_setup_generation_with_pi(self, mock_rand):
        aa = np.random.rand([400,1])
        mock_rand.return_value = aa
        PI = PredictionIntervals()
        results = setup_pseudo_results()
        PI._assign_features_from_results_structure(results = results)
        chain, s2chain, nsimu, lims, sstype, nsample, iisample = PI._setup_generation_requirements(nsample = 400, calc_pred_int = True, sstype = 0)
        
        self.assertTrue(np.array_equal(chain, results['chain']), msg = str('Arrays should match: {}'.format(chain)))
        self.assertTrue(np.array_equal(s2chain, results['s2chain']), msg = str('Arrays should match: {}'.format(s2chain)))
        self.assertEqual(nsimu, chain.shape[0], msg = 'Expect nsimu = chain.shape[0]')
        self.assertTrue(np.array_equal(lims, np.array([0.025, 0.5, 0.975])), msg = str('Arrays should match: {}'.format(lims)))
        self.assertEqual(sstype, 0, msg = 'Expect 0')
        self.assertEqual(nsample, 100, msg = 'Expect nsample to go to 100')
        self.assertTrue(np.array_equal(iisample, range(100)), msg = str('Arrays should match: {}'.format(iisample)))
        
# --------------------------------------------
class SetupPredii(unittest.TestCase):
    def test_setup_predii_nonlist_mdlfun(self):
        PI = PredictionIntervals()
        DS = basic_data_structure()
        datapred = PI._setup_data_structure_for_prediction(data = DS, ndatabatches = 1)
        
        datapredii, nrow, ncol, modelfun, test = PI._setup_predii(ii = 0, datapred = datapred, nrow = [100], ncol = [1], modelfunction = predmodelfun, local = np.array([0, 0]))
        self.assertTrue(np.array_equal(datapredii.xdata[0], datapred[0].xdata[0]), msg = 'Arrays should match')
        self.assertTrue(np.array_equal(datapredii.ydata[0], datapred[0].ydata[0]), msg = 'Arrays should match')
        self.assertEqual(nrow, 100, msg = 'Expect nrow = 100')
        self.assertEqual(ncol, 1, msg = 'Expect ncol = 1')
        self.assertEqual(modelfun, predmodelfun, msg = 'Functions should match')
        self.assertTrue(np.array_equal(test, np.array([True, True])), msg = str('Arrays should match: {}'.format(test)))
        
    def test_setup_predii_list_mdlfun(self):
        PI = PredictionIntervals()
        DS = basic_data_structure()
        datapred = PI._setup_data_structure_for_prediction(data = DS, ndatabatches = 1)
        
        datapredii, nrow, ncol, modelfun, test = PI._setup_predii(ii = 0, datapred = datapred, nrow = [100], ncol = [1], modelfunction = [predmodelfun], local = np.array([0, 0]))
        self.assertTrue(np.array_equal(datapredii.xdata[0], datapred[0].xdata[0]), msg = 'Arrays should match')
        self.assertTrue(np.array_equal(datapredii.ydata[0], datapred[0].ydata[0]), msg = 'Arrays should match')
        self.assertEqual(nrow, 100, msg = 'Expect nrow = 100')
        self.assertEqual(ncol, 1, msg = 'Expect ncol = 1')
        self.assertEqual(modelfun, predmodelfun, msg = 'Functions should match')
        self.assertTrue(np.array_equal(test, np.array([True, True])), msg = str('Arrays should match: {}'.format(test)))
        
# --------------------------------------------
class GenerateQuantiles(unittest.TestCase):
    def test_generate_quantiles(self):
        PI = PredictionIntervals()
        nsample = 500
        nrow = 100
        ncol = 1
        ysave = np.zeros([nsample, nrow, ncol])
        lims = np.array([0.025, 0.5, 0.975])
        cq = PI._generate_quantiles(response = ysave, lims = lims, ncol = ncol)
        self.assertTrue(isinstance(cq, list), msg = 'Expect list')
        self.assertEqual(cq[0].shape, (3,100), msg = 'Expect shape = (3, 100)')
        
    def test_generate_quantiles_s2chain_not_none(self):
        PI = PredictionIntervals()
        nsample = 500
        nrow = 100
        ncol = 1
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
class RunCredii(unittest.TestCase):
    def test_run_credii(self):
        PI = PredictionIntervals()
        results = setup_pseudo_results()
        PI._assign_features_from_results_structure(results = results)
        chain, s2chain, nsimu, lims, sstype, nsample, iisample = PI._setup_generation_requirements(nsample = 400, calc_pred_int = True, sstype = 0)
        testchain = np.random.random_sample(size = (100,2))
        DS = basic_data_structure()
        datapred = PI._setup_data_structure_for_prediction(data = DS, ndatabatches = 1)
        
        ysave = PI._run_credii(testchain = testchain, nrow = 100, ncol = 1, waitbar = False, test = np.array([True, True]), modelfun = predmodelfun, datapredii = datapred[0])
        self.assertTrue(isinstance(ysave, np.ndarray), msg = 'Expect array')
        self.assertEqual(ysave.shape[0], 100, msg = 'Expect 1st dim = 100')
        
# --------------------------------------------
class RunPredii(unittest.TestCase):
    def test_run_predii(self):
        PI = PredictionIntervals()
        results = setup_pseudo_results()
        PI._assign_features_from_results_structure(results = results)
        chain, s2chain, nsimu, lims, sstype, nsample, iisample = PI._setup_generation_requirements(nsample = 400, calc_pred_int = True, sstype = 0)
        testchain = np.random.random_sample(size = (100,2))
        tests2chain = np.random.random_sample(size = (100,1))
        DS = basic_data_structure()
        datapred = PI._setup_data_structure_for_prediction(data = DS, ndatabatches = 1)
        
        ysave, osave = PI._run_credii_and_predii(testchain = testchain, tests2chain = tests2chain, nrow = 100, ncol = 1, waitbar = False, sstype = 0, test = np.array([True, True]), modelfun = predmodelfun, datapredii = datapred[0])
        self.assertTrue(isinstance(ysave, np.ndarray), msg = 'Expect array')
        self.assertTrue(isinstance(osave, np.ndarray), msg = 'Expect array')
        self.assertEqual(ysave.shape[0], 100, msg = 'Expect 1st dim = 100')
        self.assertEqual(osave.shape[0], 100, msg = 'Expect 1st dim = 100')
        
    def test_run_predii_s2chain_none(self):
        PI = PredictionIntervals()
        results = setup_pseudo_results()
        PI._assign_features_from_results_structure(results = results)
        chain, s2chain, nsimu, lims, sstype, nsample, iisample = PI._setup_generation_requirements(nsample = 400, calc_pred_int = False, sstype = 0)
        testchain = np.random.random_sample(size = (100,2))
        tests2chain = None
        DS = basic_data_structure()
        datapred = PI._setup_data_structure_for_prediction(data = DS, ndatabatches = 1)
        
        with self.assertRaises(TypeError, msg = 'This function should not be called in tests2chain is None'):
            PI._run_credii_and_predii(testchain = testchain, tests2chain = tests2chain, nrow = 100, ncol = 1, waitbar = False, sstype = 0, test = np.array([True, True]), modelfun = predmodelfun, datapredii = datapred[0])
                   
    def test_run_predii_s2chain_tran(self):
        PI = PredictionIntervals()
        results = setup_pseudo_results()
        PI._assign_features_from_results_structure(results = results)
        chain, s2chain, nsimu, lims, sstype, nsample, iisample = PI._setup_generation_requirements(nsample = 400, calc_pred_int = False, sstype = 0)
        testchain = np.random.random_sample(size = (100,2))
        tests2chain = np.random.random_sample(size = (1,100))
        DS = basic_data_structure()
        datapred = PI._setup_data_structure_for_prediction(data = DS, ndatabatches = 1)
        
        with self.assertRaises(SystemExit, msg = 'Unknown structure'):
            PI._run_credii_and_predii(testchain = testchain, tests2chain = tests2chain, nrow = 100, ncol = 1, waitbar = False, sstype = 0, test = np.array([True, True]), modelfun = predmodelfun, datapredii = datapred[0])
            
# --------------------------------------------
class SetupPredictionIntervalCalculation(unittest.TestCase):
    def test_setup_pi_calc(self):
        PI = PredictionIntervals()
        DS = basic_data_structure()
        results = setup_pseudo_results()
        PI.setup_prediction_interval_calculation(results = results, data = DS, modelfunction = predmodelfun)
        self.assertEqual(PI._PredictionIntervals__ndatabatches, 1, msg = 'Expect ndatabatches = 1')
        self.assertEqual(PI.modelfunction, predmodelfun, msg = 'Functions should match')
        self.assertTrue(np.array_equal(PI._PredictionIntervals__s2chain_index, np.array([[0,1]])), msg = str('Arrays should match: {}'.format(PI._PredictionIntervals__s2chain_index)))
        self.assertTrue(np.array_equal(PI._PredictionIntervals__chain, results['chain']), msg = 'Arrays should match')
        
    def test_setup_pi_calc_with_s2chain_none(self):
        PI = PredictionIntervals()
        DS = basic_data_structure()
        results = setup_pseudo_results()
        results['s2chain'] = None
        PI.setup_prediction_interval_calculation(results = results, data = DS, modelfunction = predmodelfun)
        self.assertEqual(PI._PredictionIntervals__ndatabatches, 1, msg = 'Expect ndatabatches = 1')
        self.assertEqual(PI.modelfunction, predmodelfun, msg = 'Functions should match')
        self.assertFalse(hasattr(PI, '_PredictionIntervals__s2chain_index'), msg = 'Expect False')
        self.assertTrue(np.array_equal(PI._PredictionIntervals__chain, results['chain']), msg = 'Arrays should match')

# --------------------------------------------
#class SetupPredictionIntervalCalculation(unittest.TestCase):
    
#    # --------------------------------------------
#    def generate_prediction_intervals(self, sstype = None, nsample = 500, calc_pred_int = True, waitbar = False):
#        '''
#        Generate prediction/credible interval.
#        
#        :Args:
#            * **sstype** (:py:class:`int`): Sum-of-squares type
#            * **nsample** (:py:class:`int`): Number of samples to use in generating intervals.
#            * **calc_pred_int** (:py:class:`bool`): Flag to turn on prediction interval calculation.
#            * **waitbar** (:py:class:`bool`): Flag to turn on progress bar.
#        '''
#        
#        chain, s2chain, nsimu, lims, sstype, nsample, iisample = self._setup_generation_requirements(sstype = sstype, nsample = nsample, calc_pred_int = calc_pred_int)
#        
#        # setup progress bar
#        print('Generating credible/prediction intervals:\n')
#        if waitbar is True:
#            self.__wbarstatus = progress_bar(iters = int(nsample))
#            
#        # extract chain elements
#        testchain = chain[iisample,:]
#            
#        # loop through data sets
#        credible_intervals = []
#        prediction_intervals = []
#        for ii in range(len(self.datapred)):
#            datapredii, nrow, ncol, modelfun, test = self._setup_predii(ii = ii, datapred = self.datapred, nrow = self.__nrow, ncol = self.__ncol, modelfunction = self.modelfunction, local = self.__local)
#            if s2chain is not None:
#                s2ci = [self.__s2chain_index[ii][0], self.__s2chain_index[ii][1]]
#                tests2chain = s2chain[iisample, s2ci[0]:s2ci[1]]
#            else:
#                tests2chain = None
#            
#            # Run interval generation on set ii
#            ysave, osave = self._run_predii(testchain, tests2chain, nrow, ncol, waitbar, sstype, test, modelfun, datapredii)
#                
#            # generate quantiles
#            plim, olim = self._generate_quantiles(ysave, osave, lims, ncol, s2chain)
#                
#            credible_intervals.append(plim)
#            prediction_intervals.append(olim)
#            
#        if s2chain is None:
#            prediction_intervals = None
#            
#        # generate output dictionary
#        self.intervals = {'credible_intervals': credible_intervals,
#               'prediction_intervals': prediction_intervals}
#    
#        print('\nInterval generation complete\n')