#!/usr/bin/env python3
"""
script including
functions for easy usage in main scripts
"""

import logging
from os.path import join

import numpy as np
from scipy.interpolate import interp1d
import tqdm

from configuration import CONFIG

from .in_out import get_indices, metrics_dump, metrics_load


def concatenate_metrics(num_imgs, metrics_dir, save=False):
    log = logging.getLogger("concatenate_metrics")
    if isinstance(num_imgs, int):
        metrics = metrics_load(0, metrics_dir=metrics_dir)
        num_imgs = list(range(1, num_imgs))
    elif isinstance(num_imgs, (list, tuple)):
        metrics = metrics_load(num_imgs[0], metrics_dir=metrics_dir)
        num_imgs = num_imgs[1:]
    elif num_imgs is None:
        num_imgs = get_indices(metrics_dir)
        metrics = metrics_load(num_imgs[0], metrics_dir=metrics_dir)
        num_imgs = num_imgs[1:]
    else:
        raise ValueError(
            "num_imgs should either be of type int, list or tuple "
            "but received {}".format(type(num_imgs))
        )

    start = list([0, len(metrics["S"])])
    for i, k in enumerate(tqdm.tqdm(num_imgs, total=len(num_imgs) + 1, initial=1)):
        m = metrics_load(k, metrics_dir=metrics_dir)
        start += [start[-1] + len(m["S"])]
        for j in metrics:
            metrics[j] += m[j]
        if save:
            metrics_dump(metrics, "_all", metrics_dir=metrics_dir)
            metrics_dump(start, "_start", metrics_dir=metrics_dir)
    log.debug("Loaded {} segments.".format(max(start)))
    return metrics, start


def metrics_to_nparray(
    metrics, names, normalize=False, non_empty=False, all_metrics=(), **kwargs
):
    In = range(len(metrics["S_in"]))
    if non_empty:
        In = np.asarray(metrics["S_in"]) > 0
    M = np.asarray([np.asarray(metrics[m])[In] for m in names])
    MM = []
    if not all_metrics:
        MM = M.copy()
    else:
        MM = np.asarray([np.asarray(all_metrics[m])[In] for m in names])
    if normalize:
        mu = np.mean(MM, axis=-1) if "mean" not in kwargs else kwargs["mean"]
        std = np.std(MM, axis=-1) if "std" not in kwargs else kwargs["std"]
        for i in range(M.shape[0]):
            if names[i] != "class":
                M[i] = (np.asarray(M[i]) - mu[i]) / (std[i] + 1e-10)
        M = np.squeeze(M.T)
        return M, mu, std
    M = np.squeeze(M.T)
    return M


def label_as_onehot(label, num_classes, shift_range=0):
    y = np.zeros((num_classes, label.shape[0], label.shape[1]))
    for c in range(shift_range, num_classes + shift_range):
        y[c - shift_range][label == c] = 1
    y = np.transpose(y, (1, 2, 0))  # shape is (height, width, num_classes)
    return y.astype("uint8")


def classes_to_categorical(classes, nc=None):
    classes = np.squeeze(np.asarray(classes))
    if nc is None:
        nc = np.max(classes)
    classes = label_as_onehot(classes.reshape((classes.shape[0], 1)), nc).reshape(
        (classes.shape[0], nc)
    )
    names = ["C_" + str(i) for i in range(nc)]
    return classes, names


def metrics_to_dataset(
    metrics,
    nclasses,
    non_empty=True,
    all_metrics=(),
    class_dtype=CONFIG.CLASS_DTYPE,
    **kwargs
):
    x_names = sorted(
        [m for m in metrics if m not in ["class", "iou", "iou0"] and "cprob" not in m]
    )
    if class_dtype == "probs":
        class_names = [
            "cprob" + str(i) for i in range(nclasses) if "cprob" + str(i) in metrics
        ]
    elif class_dtype == "one_hot_classes":
        class_names = ["class"]
    else:
        raise ValueError(
            "class_dtype should be one of ['one_hot_classes', 'probs'] "
            "but got {}.".format(class_dtype)
        )

    if "xa_mean" in kwargs and "xa_std" in kwargs:
        xa, xa_mean, xa_std = metrics_to_nparray(
            metrics,
            x_names,
            normalize=True,
            non_empty=non_empty,
            all_metrics=all_metrics,
            mean=kwargs["xa_mean"],
            std=kwargs["xa_std"],
        )
    else:
        xa, xa_mean, xa_std = metrics_to_nparray(
            metrics,
            x_names,
            normalize=True,
            non_empty=non_empty,
            all_metrics=all_metrics,
        )

    if "classes_mean" in kwargs and "classes_std" in kwargs:
        classes, classes_mean, classes_std = metrics_to_nparray(
            metrics,
            class_names,
            normalize=True,
            non_empty=non_empty,
            all_metrics=all_metrics,
            mean=kwargs["classes_mean"],
            std=kwargs["classes_std"],
        )
    else:
        classes, classes_mean, classes_std = metrics_to_nparray(
            metrics,
            class_names,
            normalize=True,
            non_empty=non_empty,
            all_metrics=all_metrics,
        )

    ya = metrics_to_nparray(metrics, ["iou"], normalize=False, non_empty=non_empty)
    y0a = metrics_to_nparray(metrics, ["iou0"], normalize=False, non_empty=non_empty)

    if class_dtype == "one_hot_classes":
        classes, class_names = classes_to_categorical(classes, nclasses)
    return (
        xa,
        classes,
        ya,
        y0a,
        x_names,
        class_names,
        xa_mean,
        xa_std,
        classes_mean,
        classes_std,
    )


def load_data(
    dataset="cityscapes", num_imgs=None, model_name=CONFIG.MODEL_NAME, **kwargs
):
    # loads all data that can be found in the metaseg input folder
    metrics_dir = join(CONFIG.metaseg_io_path, "metrics", model_name, dataset)
    num_imgs = get_indices(metrics_dir) if num_imgs is None else num_imgs
    metrics, start = concatenate_metrics(num_imgs, save=False, metrics_dir=metrics_dir)
    nclasses = np.max(metrics["class"]) + 1

    (
        xa,
        classes,
        ya,
        _,
        x_names,
        class_names,
        xa_mean,
        xa_std,
        classes_mean,
        classes_std,
    ) = metrics_to_dataset(metrics, nclasses, non_empty=False, **kwargs)
    xa = np.concatenate((xa, classes), axis=-1)

    return (
        xa,
        ya,
        x_names,
        class_names,
        xa_mean,
        xa_std,
        classes_mean,
        classes_std,
        start,
        metrics["class"],
    )


def get_lambdas(n_steps, min_pow, max_pow):
    m = interp1d([0, n_steps - 1], [min_pow, max_pow])
    lambdas = [10 ** m(i).item() for i in range(n_steps)]
    return lambdas
