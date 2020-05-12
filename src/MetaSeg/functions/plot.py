#!/usr/bin/env python3
"""
script including
functions for visualizations
"""

import pyximport

pyximport.install()

import os
from os.path import join
import numpy as np
from PIL import Image
import matplotlib

matplotlib.use('Agg')
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.stats import pearsonr, kde
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import importlib

from configuration import CONFIG
from src.MetaSeg.functions.in_out import get_save_path_input_i, probs_gt_load, get_img_path_fname, components_load
from src.MetaSeg.functions import labels
from src.MetaSeg.functions.utils import estimate_kernel_density, get_grid
from src.MetaSeg.functions.metrics import entropy

if CONFIG.DATASET.name == 'cityscapes':
    trainId2color = {label.trainId: label.color for label in reversed(labels.cs_labels)}
    ood_trainId2color = {label.trainId: label.color for label in reversed(labels.cs_labels)}
elif CONFIG.DATASET.name == 'ds20k':
    trainId2color = {label.trainId: label.color for label in reversed(labels.ds20k_labels)}
    ood_trainId2color = {label.trainId: label.color for label in reversed(labels.ds20k_labels)}
elif CONFIG.DATASET.name == 'a2d2':
    trainId2color = {label.trainId: label.color for label in reversed(labels.a2d2_labels)}
    ood_trainId2color = {label.trainId: label.color for label in reversed(labels.a2d2_labels)}

os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/texbin'  # for tex in matplotlib
plt.rc('font', size=10, family='serif')
plt.rc('axes', titlesize=10)
plt.rc('figure', titlesize=10)
plt.rc('text', usetex=False)


def visualize_segments(comp, metric):
    r = np.asarray(metric)
    r = 1 - 0.5 * r
    g = np.asarray(metric)
    b = 0.3 + 0.35 * np.asarray(metric)

    r = np.concatenate((r, np.asarray([0, 1])))
    g = np.concatenate((g, np.asarray([0, 1])))
    b = np.concatenate((b, np.asarray([0, 1])))

    components = np.asarray(comp.copy(), dtype='int16')
    components[components < 0] = len(r) - 1
    components[components == 0] = len(r)

    img = np.zeros(components.shape + (3,))
    x, y = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
    x = x.reshape(-1)
    y = y.reshape(-1)

    img[x, y, 0] = r[components[x, y] - 1]
    img[x, y, 1] = g[components[x, y] - 1]
    img[x, y, 2] = b[components[x, y] - 1]

    img = np.asarray(255 * img).astype('uint8')

    return img


def visualize_regression_prediction_i(iou_pred,
                                      i,
                                      input_dir=CONFIG.INPUT_DIR,
                                      seg_dir=CONFIG.IOU_SEG_VIS_DIR,
                                      components_dir=CONFIG.COMPONENTS_DIR):

    if os.path.isfile(get_save_path_input_i(i, input_dir=input_dir)):
        label_mapping = getattr(
            importlib.import_module(CONFIG.DATASET.module_name),
            CONFIG.DATASET.class_name)(
            **CONFIG.DATASET.kwargs,
        ).label_mapping
        pred_mapping = getattr(
            importlib.import_module(CONFIG.TRAIN_DATASET.module_name),
            CONFIG.TRAIN_DATASET.class_name)(
            **CONFIG.TRAIN_DATASET.kwargs,
        ).label_mapping

        probs, gt, path = probs_gt_load(i, input_dir=input_dir)
        input_image = Image.open(path).convert("RGB")
        input_image = np.asarray(input_image.resize(probs.shape[:2][::-1]))
        components = components_load(i, components_dir=components_dir)

        e = entropy(probs)
        pred = np.asarray(np.argmax(probs, axis=-1), dtype='int')
        gt[gt == 255] = 0
        predc = np.asarray([pred_mapping[pred[p, q]][1] for p in range(pred.shape[0]) for q in range(pred.shape[1])])
        gtc = np.asarray([label_mapping[gt[p, q]][1] for p in range(gt.shape[0]) for q in range(gt.shape[1])])
        predc = predc.reshape(input_image.shape)
        gtc = gtc.reshape(input_image.shape)

        overlay_factor = [1.0, 0.5, 1.0]
        img_predc, img_gtc, img_entropy = [
            Image.fromarray(np.uint8(arr * overlay_factor[i] + input_image * (1 - overlay_factor[i])))
            for i, arr in enumerate([predc,
                                     gtc,
                                     cm.jet(e)[:, :, :3] * 255.0])]

        img_ioupred = Image.fromarray(visualize_segments(components, iou_pred))

        images = [img_gtc, img_predc, img_entropy, img_ioupred]

        img_top = np.concatenate(images[2:], axis=1)
        img_bottom = np.concatenate(images[:2], axis=1)

        img_total = np.concatenate((img_top, img_bottom), axis=0)
        image = Image.fromarray(img_total.astype('uint8'), 'RGB')

        if not os.path.exists(seg_dir):
            os.makedirs(seg_dir)

        image.save(join(seg_dir, "image{}.png".format(i)))
        plt.close()

        print("stored: {}".format(join(seg_dir, "image{}.png".format(i))))


def plot_roc_curve(y, probs, roc_path):
    fpr, tpr, _ = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)
    print("auc", roc_auc)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='red', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve of meta classification performance')
    plt.legend(loc="lower right")

    roc_dir = os.path.dirname(roc_path)
    if not os.path.exists(roc_dir):
        os.makedirs(roc_dir)

    plt.savefig(roc_path)
    print("roc curve saved to " + roc_path)
    plt.close()

    return roc_auc


def plot_regression(x2_val, y2_val, y2_pred, x_names):
    cmap = plt.get_cmap('tab20')
    plt.figure(figsize=(3, 3), dpi=300)
    plt.clf()
    s_ind = 0
    for s_ind in range(len(x_names)):
        if x_names[s_ind] == "S":
            break

    sizes = np.squeeze(x2_val[:, s_ind] * np.std(x2_val[:, s_ind]))
    sizes = sizes - np.min(sizes)
    sizes = sizes / np.max(sizes) * 50  # + 1.5
    x = np.arange(0., 1, .01)
    plt.plot(x, x, color='black', alpha=0.5, linestyle='dashed')
    plt.scatter(y2_val, np.clip(y2_pred, 0, 1), s=sizes, linewidth=.5, c=cmap(0), edgecolors=cmap(1), alpha=0.25)
    plt.xlabel('$\mathit{IoU}_\mathrm{adj}$')
    plt.ylabel('predicted $\mathit{IoU}_\mathrm{adj}$')
    plt.savefig(CONFIG.RESULTS_DIR + 'regression.png', bbox_inches='tight')

    plt.clf()


def plot_classif_hist(ya_val, ypred):
    figsize = (8.75, 5.25)
    plt.clf()

    density1 = gaussian_kde(ya_val[ypred == 1])
    density2 = gaussian_kde(ya_val[ypred == 0])

    density1.set_bandwidth(bw_method=density1.factor / 2.)
    density2.set_bandwidth(bw_method=density2.factor / 2.)

    x = np.arange(0., 1, .01)

    plt.clf()
    plt.figure(figsize=figsize)
    plt.plot(x, density1(x), color='red', alpha=0.66, label="pred. $IoU = 0$")
    plt.plot(x, density2(x), color='blue', alpha=0.66, label="pred. $IoU > 0$")
    plt.hist(ya_val[ypred == 1], bins=20, color='red', alpha=0.1, density=True)
    plt.hist(ya_val[ypred == 0], bins=20, color='blue', alpha=0.1, density=True)
    plt.legend(loc='upper right')
    plt.xlabel('$\mathit{IoU}_\mathrm{adj}$')
    plt.savefig(CONFIG.RESULTS_DIR + 'classif_hist.pdf', bbox_inches='tight')

    plt.clf()


def plot_classif(stats, mean_stats, x_names, class_names):
    nc = len(x_names) - len(class_names)
    coefs = np.squeeze(stats['coefs'][0, :, :])
    coefs = np.concatenate([coefs[:, 0:nc], np.max(np.abs(coefs[:, nc:]), axis=1).reshape((coefs.shape[0], 1))], axis=1)
    max_acc = np.argmax(stats['penalized_val_acc'][0], axis=-1)
    lambdas = stats["lambdas"]

    cmap = plt.get_cmap('tab20')
    figsize = (8.75, 5.25)

    plt.clf()
    plt.semilogx(lambdas, stats['plain_val_acc'][0], label="unpenalized model", color=cmap(2))
    plt.semilogx(lambdas, stats['penalized_val_acc'][0], label="penalized model", color=cmap(0))
    plt.semilogx(lambdas, mean_stats['entropy_val_acc'] * np.ones((len(lambdas),)), label="entropy baseline",
                 color='black', linestyle='dashed')
    ymin, ymax = plt.ylim()
    plt.vlines(lambdas[max_acc], ymin, ymax, linestyle='dashed', linewidth=0.5, color='grey')
    legend = plt.legend(loc='lower right')
    plt.xlabel('$\lambda^{-1}$')
    plt.ylabel('classification accuracy')
    plt.axis('tight')
    plt.savefig(CONFIG.RESULTS_DIR + 'classif_perf.pdf', bbox_inches='tight')

    plt.clf()
    plt.semilogx(lambdas, stats['plain_val_auroc'][0], label="unpenalized model", color=cmap(2))
    plt.semilogx(lambdas, stats['penalized_val_auroc'][0], label="penalized model", color=cmap(0))
    plt.semilogx(lambdas, mean_stats['entropy_val_auroc'] * np.ones((len(lambdas),)), label="entropy baseline",
                 color='black', linestyle='dashed')
    ymin, ymax = plt.ylim()
    plt.vlines(lambdas[max_acc], ymin, ymax, linestyle='dashed', linewidth=0.5, color='grey')
    legend = plt.legend(loc='lower right')
    plt.xlabel('$\lambda^{-1}$')
    plt.ylabel('AUROC')
    plt.axis('tight')
    plt.savefig(CONFIG.RESULTS_DIR + 'classif_auroc.pdf', bbox_inches='tight')
    plt.close()


def add_scatterplot_vs_iou(ious, sizes, dataset, shortname, size_fac, scale, setylim=True):
    cmap = plt.get_cmap('tab20')
    rho = pearsonr(ious, dataset)
    plt.title(r"$\rho = {:.05f}$".format(rho[0]))
    plt.scatter(ious, dataset, s=sizes / np.max(sizes) * size_fac, linewidth=.5, c=cmap(0), edgecolors=cmap(1),
                alpha=.25)
    plt.xlabel('$\mathit{IoU}_\mathrm{adj}$', labelpad=-10)
    plt.ylabel(shortname, labelpad=-8)
    plt.ylim(-.05, 1.05)
    plt.xticks((0, 1), fontsize=10 * scale)
    plt.yticks((0, 1), fontsize=10 * scale)


def plot_scatter(df_full, m='E'):
    print("")
    print("making iou scatterplot ...")
    scale = .75
    size_fac = 50 * scale

    plt.figure(figsize=(3, 3), dpi=300)
    add_scatterplot_vs_iou(df_full['iou'], df_full['S'], df_full[m], m, size_fac, scale)

    plt.tight_layout(pad=1.0 * scale, w_pad=0.5 * scale, h_pad=1.5 * scale)
    save_path = os.path.join(CONFIG.RESULTS_DIR, 'iou_vs_' + m + '.png')
    plt.savefig(save_path, bbox_inches='tight')
    print("scatterplots saved to " + save_path)
    plt.close()


# noinspection PyArgumentList
def plot_metaseg_component_densities(xa, filename, n_components=1, model=None, method='TSNE'):
    colors = ['g', 'r', 'b', 'y', 'm']
    #     colors=['r', 'b', 'y', 'm']
    if not isinstance(xa, (list, tuple)):
        xa = [xa]

    print('Computing {}...'.format(method))
    if method == 'TSNE':
        pcs = PCA(n_components=50).fit_transform(np.concatenate(xa))
        pcs = TSNE(n_components=n_components).fit_transform(pcs)
    elif method == 'PCA':
        if model is None:
            pca = PCA(n_components=n_components)
            model = pca.fit(np.concatenate(xa))
        pcs = model.transform(np.concatenate(xa))
    else:
        raise ValueError('method should be one of [\'PCA\', \'TSNE\']')

    lengths = [0]
    for i in xa:
        lengths.append(i.shape[0])
    lengths = [0] + [lengths[i]+lengths[i-1] for i in range(1, len(lengths))]
    pcs = [pcs[lengths[i-1]:lengths[i]] for i in range(1, len(lengths))]

    plt.figure(figsize=(24, 15))
    ax = plt.axes(projection='3d' if n_components == 2 else None)

    for i, pc in enumerate(pcs):
        print('Transforming...')
        xx, yy, positions = get_grid(pc, n_components=n_components)
        kernel = estimate_kernel_density(pc)
        f = np.reshape(kernel(positions).T, xx.shape)

        if n_components == 1:
            ax.plot(xx,
                    f,
                    color=colors[i],
                    linestyle='-')
        elif n_components == 2:
            ax.plot_surface(xx,
                            yy,
                            f,
                            rstride=1,
                            cstride=1,
                            color=colors[i],
                            edgecolor='none')

    if n_components == 1:
        ax.set_ylabel('Density')
    elif n_components == 2:
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Density')

    ax.set_xlabel('Component 1')
    #     ax.set_title('MetaSeg Component Metric Density Estimate')
    #     ax.set(xlim=(0,100), ylim=(-50,50), zlim=(0,1))
    #     ax.view_init(60,35)
    #     ax.set(xlim=(ax_ranges[0], ax_ranges[1]), ylim=(ax_ranges[2], ax_ranges[3]), zlim=(ax_ranges[4], ax_ranges[5]))
    #     plt.show()
    plt.savefig(filename)
    return model
