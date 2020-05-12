#!/usr/bin/env python3
"""
script including
class objects called in main
"""

import pyximport

pyximport.install()

import numpy as np
import time
import os
from multiprocessing import Pool
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, auc
from functools import partial

from configuration import CONFIG
from src.MetaSeg.functions.metrics import compute_metrics_components, entropy
from src.MetaSeg.functions.helper import concatenate_metrics, metrics_to_dataset, get_lambdas, metrics_to_nparray
from src.MetaSeg.functions.in_out import get_save_path_input_i, get_save_path_metrics_i, probs_gt_load, metrics_dump, \
    components_dump, stats_dump, metrics_load, components_load, probs_gt_load_all, get_indices
from src.MetaSeg.functions.plot import visualize_regression_prediction_i, plot_roc_curve, plot_regression, plot_classif, \
    plot_scatter, plot_classif_hist
from src.MetaSeg.functions.calculate import regression_fit_and_predict, classification_l1_fit_and_predict, \
    classification_fit_and_predict, compute_correlations, compute_metrics_from_heatmap, meta_nn_predict


# NOTE: Couldn't use the python logging module here because of clash with the multiprocessing Pool


class ComputeMetrics(object):

    def __init__(self, num_cores=1, num_imgs=len(os.listdir(CONFIG.INPUT_DIR)), rewrite=True):
        """
    object initialization
    :param num_cores: (int) number of cores used for parallelization
    :param num_imgs:  (int) number of images to be processed
    :param rewrite:   (boolean) overwrite existing files if True
    """
        self.num_cores = num_cores if not hasattr(CONFIG, 'NUM_CORES') else CONFIG.NUM_CORES
        self.rewrite = rewrite
        if isinstance(num_imgs, int):
            num_imgs = get_indices(CONFIG.INPUT_DIR) \
                if num_imgs == 0 else get_indices(CONFIG.INPUT_DIR)[:num_imgs]
        elif not isinstance(num_imgs, (list, tuple)):
            raise ValueError('num_imgs should be of type int, list or tuple but received {}'.format(type(num_imgs)))
        self.num_imgs = num_imgs if not hasattr(CONFIG, 'NUM_IMAGES') \
            else get_indices(CONFIG.INPUT_DIR)[:CONFIG.NUM_IMAGES]

    def compute_metrics_per_image(self):
        """
    perform metrics computation
    """
        print("Calculating metrics for \'{}\'".format(CONFIG.DATASET.name))
        with Pool(self.num_cores) as p:
            p.map(partial(self.compute_metrics_i,
                          input_dir=CONFIG.INPUT_DIR,
                          metrics_dir=CONFIG.METRICS_DIR,
                          components_dir=CONFIG.COMPONENTS_DIR), self.num_imgs)

    def compute_metrics_i(self, i, input_dir, metrics_dir, components_dir):
        """
    perform metrics computation for one image
    :param i: (int) id of the image to be processed
    """
        if os.path.isfile(get_save_path_input_i(i, input_dir=input_dir)) and self.rewrite:
            start = time.time()
            probs, gt, _ = probs_gt_load(i, input_dir=input_dir)
            metrics, components = compute_metrics_components(probs, gt)
            metrics_dump(metrics, i, metrics_dir=metrics_dir)
            components_dump(components, i, components_dir=components_dir)
            print('image {} processed in {}s'.format(i, round(time.time() - start)))

    def add_heatmaps_as_metric(self, heat_dir, key):
        """
    add another dispersion heatmap as metric/input for meta model
    :param heat_dir:  (str) directory with heatmaps as numpy arrays
    :param key:       (str) new key to access added metric
    """
        print('Add {} to metrics'.format(key))
        p_args = [(heat_dir, key, k) for k in self.num_imgs]
        with Pool(self.num_cores) as p:
            p.starmap(self.add_heatmap_as_metric_i, p_args)

    @staticmethod
    def add_heatmap_as_metric_i(heat_dir, key, i):
        """
    derive aggregated metrics per image and add to metrics dictionary
    :param heat_dir:  (str) directory with heatmaps as numpy arrays
    :param key:       (str) new key to access added metric
    :param i:         (int) id of the image to be processed
    """
        _, _, path = probs_gt_load(i)
        heat_name = os.path.basename(path)[:-4] + ".npy"
        heatmap = np.load(heat_dir + heat_name)
        metrics = metrics_load(i, metrics_dir=CONFIG.METRICS_DIR)
        components = components_load(i, components_dir=CONFIG.COMPONENTS_DIR)
        keys = [key, key + "_in", key + "_bd", key + "_rel", key + "_rel_in"]
        heat_metric = {k: [] for k in keys}
        for comp_id in range(1, abs(np.min(components)) + 1):
            values = compute_metrics_from_heatmap(heatmap, components, comp_id)
            for j, k in enumerate(keys):
                heat_metric[k].append(values[j])
        metrics.update(heat_metric)
        metrics_dump(metrics, i, metrics_dir=CONFIG.METRICS_DIR)


class VisualizeMetaPrediction(object):

    def __init__(self, num_cores=1, num_imgs=len(os.listdir(CONFIG.INPUT_DIR)), **kwargs):
        """
    object initialization
    :param num_cores: (int) number of cores used for parallelization
    :param num_imgs:  (int) number of images to be processed
    """
        self.num_cores = num_cores if not hasattr(CONFIG, 'NUM_CORES') else CONFIG.NUM_CORES

        if isinstance(num_imgs, int):
            num_imgs = get_indices(CONFIG.INPUT_DIR) \
                if num_imgs == 0 else get_indices(CONFIG.INPUT_DIR)[:num_imgs]
        elif not isinstance(num_imgs, (list, tuple)):
            raise ValueError('num_imgs should be of type int, list or tuple but received {}'.format(type(num_imgs)))
        self.num_imgs = num_imgs if not hasattr(CONFIG, 'NUM_IMAGES') \
            else get_indices(CONFIG.INPUT_DIR)[:CONFIG.NUM_IMAGES]

    def visualize_regression_per_image(self):
        """
    perform metrics visualization
    """
        print("Visualization for {} running".format(CONFIG.DATASET.name))
        metrics, start = concatenate_metrics(self.num_imgs, save=False, metrics_dir=CONFIG.METRICS_DIR)
        nclasses = np.max(metrics["class"]) + 1

        xa, classes, ya, _, _, _, xa_mean, xa_std, classes_mean, classes_std = metrics_to_dataset(metrics,
                                                                                                  nclasses,
                                                                                                  non_empty=False)
        xa = np.concatenate((xa, classes), axis=-1)

        if CONFIG.META_MODEL == 'neural':
            ya_pred = meta_nn_predict(CONFIG.meta_nn_weights, xa, CONFIG.GPU_ID)
        elif CONFIG.META_MODEL == 'linear':
            ya_pred, _ = regression_fit_and_predict(xa, ya, xa)
        else:
            raise ValueError('Unknown meta model \'{}\''.format(CONFIG.META_MODEL))
        print("Model R2 score: {:.2%}\n".format(r2_score(ya, ya_pred)))
        p_args = [(ya_pred[start[i]:start[i + 1]], j)
                  for i, j in enumerate(self.num_imgs)]

        with Pool(self.num_cores) as p:
            p.starmap(visualize_regression_prediction_i, p_args)


class AnalyzeMetrics(object):

    def __init__(self, num_cores=1, num_imgs=len(os.listdir(CONFIG.INPUT_DIR)), n_av=CONFIG.NUM_AVERAGES):
        """
    object initialization
    :param num_cores: (int) number of cores used for parallelization
    :param num_imgs:  (int) number of images to be processed
    :param n_av:      (int) number of model fitting runs with random data splits
    """
        self.num_cores = num_cores if not hasattr(CONFIG, 'NUM_CORES') else CONFIG.NUM_CORES
        if isinstance(num_imgs, int):
            num_imgs = list(range(len(os.listdir(CONFIG.INPUT_DIR)))) if num_imgs == 0 else list(range(num_imgs))
        elif isinstance(num_imgs, (list, tuple)):
            pass
        else:
            raise ValueError('num_imgs should be of type int, list or tuple but received {}'.format(type(num_imgs)))
        self.num_imgs = num_imgs if not hasattr(CONFIG, 'NUM_IMAGES') else list(range(CONFIG.NUM_IMAGES))
        self.n_av = n_av

    def prepare_analysis(self):
        """
    prepare metrics analysis
    create dataframes storing results of meta model per run
    :return:
    """
        metrics, start = concatenate_metrics(self.num_imgs, save=False)
        nclasses = np.max(metrics["class"]) + 1

        xa, classes, ya, y0a, x_names, class_names = metrics_to_dataset(metrics, nclasses)
        xa = np.concatenate((xa, classes), axis=-1)
        self.X_names = x_names + class_names

        self.lambdas = get_lambdas(CONFIG.NUM_LASSO_LAMBDAS, min_pow=-4.2, max_pow=0.8)
        stats = self.init_stats()
        single_run_stats = self.init_stats()

        p_args = [(xa, ya, y0a, single_run_stats, run) for run in range(self.n_av)]
        with Pool(self.num_cores) as p:
            single_run_stats = p.starmap(self.fit_model_run, p_args)

        # single_run_stats = self.fit_model_run( Xa, ya, y0a, single_run_stats, 0 )
        df_all, df_full = compute_correlations(metrics)
        plot_scatter(df_full)
        y0a = metrics_to_nparray(metrics, ["iou0"], normalize=False, non_empty=True)
        print("IoU=0: {} of {}".format(np.sum(y0a == 1), y0a.shape[0]))
        print("IoU>0: {} of {}".format(np.sum(y0a == 0), y0a.shape[0]))
        stats = self.merge_stats(stats, single_run_stats)

        mean_stats, _ = stats_dump(stats, df_all, y0a)
        plot_classif(stats, mean_stats, x_names, class_names)

    def init_stats(self):
        """
    initialize dataframe for storing results
    """
        stats = dict({})
        per_alphas_av_stats = ['penalized_val_acc', 'penalized_val_auroc', 'penalized_train_acc',
                               'penalized_train_auroc',
                               'plain_val_acc', 'plain_val_auroc', 'plain_train_acc', 'plain_train_auroc', 'coefs']
        per_av_stats = ['entropy_val_acc', 'entropy_val_auroc', 'entropy_train_acc', 'entropy_train_auroc',
                        'regr_val_mse', 'regr_val_r2', 'regr_train_mse', 'regr_train_r2',
                        'entropy_regr_val_mse', 'entropy_regr_val_r2', 'entropy_regr_train_mse',
                        'entropy_regr_train_r2',
                        'iou0_found', 'iou0_not_found', 'not_iou0_found', 'not_iou0_not_found']

        for s in per_alphas_av_stats:
            stats[s] = 0.5 * np.ones((self.n_av, len(self.lambdas)))

        for s in per_av_stats:
            stats[s] = np.zeros((self.n_av,))

        stats["coefs"] = np.zeros((self.n_av, len(self.lambdas), len(self.X_names)))
        stats["lambdas"] = self.lambdas
        stats["n_av"] = self.n_av
        stats["n_metrics"] = len(self.X_names)
        stats["metric_names"] = self.X_names

        return stats

    def fit_model_run(self, xa, ya, y0a, single_run_stats, run):
        """
    fit meta model for one random data split and store results in dataframe
    :param xa:  (np array) dispersion metrics as input for meta model
    :param ya:  (np array) meta regression label of segment, i.e. IoU value
    :param y0a: (np array) meta classification label of segment, i.e. intersection with ground truth or not
    :param single_run_stats:  (dict) empty dataframe where results are stored into
    :param run: (int) run id
    :return: dict dataframe with stored results of meta model
    """
        print("Run {}".format(run))
        xa_val, ya_val, y0a_val, xa_train, ya_train, y0a_train = self.split_data_randomly(xa, ya, y0a, seed=run)
        coefs = np.zeros((len(self.lambdas), xa.shape[1]))
        max_acc = 0
        best_l1_results = []

        for i in range(len(self.lambdas)):

            y0a_val_pred, y0a_train_pred, lm_coefs = classification_l1_fit_and_predict(xa_train, y0a_train,
                                                                                       self.lambdas[i], xa_val)

            single_run_stats['penalized_val_acc'][run, i] = np.mean(np.argmax(y0a_val_pred, axis=-1) == y0a_val)
            single_run_stats['penalized_train_acc'][run, i] = np.mean(np.argmax(y0a_train_pred, axis=-1) == y0a_train)

            if single_run_stats['penalized_val_acc'][run, i] > max_acc:
                max_acc = single_run_stats['penalized_val_acc'][run, i]
                best_l1_results = [y0a_val_pred.copy(), y0a_train_pred.copy()]

            print('Step {}, alpha={:.2E}, val. acc.: {:.2%}'.format(
                i,
                self.lambdas[i],
                single_run_stats['penalized_val_acc'][run, i]))

            fpr, tpr, _ = roc_curve(y0a_val, y0a_val_pred[:, 1])
            single_run_stats['penalized_val_auroc'][run, i] = auc(fpr, tpr)
            fpr, tpr, _ = roc_curve(y0a_train, y0a_train_pred[:, 1])
            single_run_stats['penalized_train_auroc'][run, i] = auc(fpr, tpr)
            coefs[i] = lm_coefs

            if np.sum(np.abs(coefs[i]) > 1e-6) > 0:
                y0a_val_pred, y0a_train_pred = classification_fit_and_predict(xa_train[:, np.abs(coefs[i]) > 1e-6],
                                                                              y0a_train,
                                                                              xa_val[:, np.abs(coefs[i]) > 1e-6])

                single_run_stats['plain_val_acc'][run, i] = np.mean(np.argmax(y0a_val_pred, axis=-1) == y0a_val)
                single_run_stats['plain_train_acc'][run, i] = np.mean(np.argmax(y0a_train_pred, axis=-1) == y0a_train)
                fpr, tpr, _ = roc_curve(y0a_val, y0a_val_pred[:, 1])
                single_run_stats['plain_val_auroc'][run, i] = auc(fpr, tpr)
                fpr, tpr, _ = roc_curve(y0a_train, y0a_train_pred[:, 1])
                single_run_stats['plain_train_auroc'][run, i] = auc(fpr, tpr)
            else:
                single_run_stats['plain_val_acc'][run, i] = single_run_stats['penalized_val_acc'][run, i]
                single_run_stats['plain_train_acc'][run, i] = single_run_stats['penalized_train_acc'][run, i]

                single_run_stats['plain_val_auroc'][run, i] = single_run_stats['penalized_val_auroc'][run, i]
                single_run_stats['plain_train_auroc'][run, i] = single_run_stats['penalized_train_auroc'][run, i]

        ypred = np.argmax(best_l1_results[0], axis=-1)
        ypred_t = np.argmax(best_l1_results[1], axis=-1)

        e_ind = 0
        for e_ind in range(len(self.X_names)):
            if self.X_names[e_ind] == "E":
                break

        y0a_val_pred, y0a_train_pred = classification_fit_and_predict(
            xa_train[:, e_ind].reshape((xa_train.shape[0], 1)),
            y0a_train, xa_val[:, e_ind].reshape((xa_val.shape[0], 1)))

        single_run_stats['entropy_val_acc'][run] = np.mean(np.argmax(y0a_val_pred, axis=-1) == y0a_val)
        single_run_stats['entropy_train_acc'][run] = np.mean(np.argmax(y0a_val_pred, axis=-1) == y0a_val)
        fpr, tpr, _ = roc_curve(y0a_val, y0a_val_pred[:, 1])
        single_run_stats['entropy_val_auroc'][run] = auc(fpr, tpr)
        fpr, tpr, _ = roc_curve(y0a_train, y0a_train_pred[:, 1])
        single_run_stats['entropy_train_auroc'][run] = auc(fpr, tpr)

        single_run_stats['iou0_found'][run] = np.sum(np.logical_and(ypred == 1, y0a_val == 1)) + np.sum(
            np.logical_and(ypred_t == 1, y0a_train == 1))
        single_run_stats['iou0_not_found'][run] = np.sum(np.logical_and(ypred == 0, y0a_val == 1)) + np.sum(
            np.logical_and(ypred_t == 0, y0a_train == 1))
        single_run_stats['not_iou0_found'][run] = np.sum(np.logical_and(ypred == 0, y0a_val == 0)) + np.sum(
            np.logical_and(ypred_t == 0, y0a_train == 0))
        single_run_stats['not_iou0_not_found'][run] = np.sum(np.logical_and(ypred == 1, y0a_val == 0)) + np.sum(
            np.logical_and(ypred_t == 1, y0a_train == 0))

        x2_train = xa_val.copy()
        y2_train = ya_val.copy()
        x2_val = xa_train.copy()
        y2_val = ya_train.copy()

        y2_val_pred, y2_train_pred = regression_fit_and_predict(x2_train, y2_train, x2_val)

        single_run_stats['regr_val_mse'][run] = np.sqrt(mean_squared_error(y2_val, y2_val_pred))
        single_run_stats['regr_val_r2'][run] = r2_score(y2_val, y2_val_pred)
        single_run_stats['regr_train_mse'][run] = np.sqrt(mean_squared_error(y2_train, y2_train_pred))
        single_run_stats['regr_train_r2'][run] = r2_score(y2_train, y2_train_pred)

        y2e_val_pred, y2e_train_pred = regression_fit_and_predict(x2_train[:, e_ind].reshape((x2_train.shape[0], 1)),
                                                                  y2_train,
                                                                  x2_val[:, e_ind].reshape((x2_val.shape[0], 1)))

        single_run_stats['entropy_regr_val_mse'][run] = np.sqrt(mean_squared_error(y2_val, y2e_val_pred))
        single_run_stats['entropy_regr_val_r2'][run] = r2_score(y2_val, y2e_val_pred)
        single_run_stats['entropy_regr_train_mse'][run] = np.sqrt(mean_squared_error(y2_train, y2e_train_pred))
        single_run_stats['entropy_regr_train_r2'][run] = r2_score(y2_train, y2e_train_pred)

        single_run_stats['coefs'][run] = np.asarray(coefs)

        if run == 0:
            plot_roc_curve(y0a_val, best_l1_results[0][:, 1], CONFIG.RESULTS_DIR + 'roccurve.pdf')
            plot_regression(x2_val, y2_val, y2_val_pred, self.X_names)
            plot_classif_hist(ya_val, ypred)

        return single_run_stats

    @staticmethod
    def split_data_randomly(xa, ya, y0a, seed):
        """
    create random data split 50/50 for training and validation
    :param xa:  (np array) dispersion metrics
    :param ya:  (np array) meta regression label
    :param y0a: (np array) meta classification label
    :param seed: (int) number used to initialize random generator
    :return: splitted np arrays for training and validation
    """
        np.random.seed(seed)
        val_mask = np.random.rand(len(ya)) < 3.0 / 6.0

        xa_val = xa[val_mask]
        ya_val = ya[val_mask]
        y0a_val = y0a[val_mask]

        xa_train = xa[np.logical_not(val_mask)]
        ya_train = ya[np.logical_not(val_mask)]
        y0a_train = y0a[np.logical_not(val_mask)]

        return xa_val, ya_val, y0a_val, xa_train, ya_train, y0a_train

    def merge_stats(self, stats, single_run_stats):
        """
    combine results for every one single dataframe
    :param stats: (dict) the single dataframe
    :param single_run_stats: (dict) dataframe from one run, note: strange format due to parallelization
    :return: dict dataframe with stored results
    """
        for run in range(self.n_av):
            for s in stats:
                if s not in ["alphas", "n_av", "n_metrics", "metric_names"]:
                    stats[s][run] = single_run_stats[run][s][run]

        return stats


"""
TODOS: the functions "fit_model_run" and "init_stats" must be improved in terms of readability
"""
