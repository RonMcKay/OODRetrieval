#!/usr/bin/env python3
"""
script including
functions that do calculations
"""

import importlib

import numpy as np
import pandas as pd
from sklearn import linear_model
import torch

from configuration import CONFIG, meta_models


def regression_fit_and_predict(x_train, y_train, x_test):
    model = linear_model.LinearRegression()
    model.fit(x_train, y_train)
    y_test_pred = np.clip(model.predict(x_test), 0, 1)
    y_train_pred = np.clip(model.predict(x_train), 0, 1)
    return y_test_pred, y_train_pred


def classification_l1_fit_and_predict(x_train, y_train, lambdas, x_test):
    if CONFIG.META_MODEL_TYPE == "linear":
        model = linear_model.LogisticRegression(
            C=lambdas, penalty="l1", solver="saga", max_iter=1000, tol=1e-3
        )
    model.fit(x_train, y_train)
    y_test_pred = model.predict_proba(x_test)
    y_train_pred = model.predict_proba(x_train)
    return y_test_pred, y_train_pred, np.asarray(model.coef_[0])


def classification_fit_and_predict(x_train, y_train, x_test):
    if CONFIG.META_MODEL_TYPE == "linear":
        model = linear_model.LogisticRegression(solver="saga", max_iter=1000, tol=1e-3)
    else:
        raise ValueError(
            "meta segmentation model '{}' not supported by this function.".format(
                CONFIG.META_MODEL_TYPE
            )
        )
    model.fit(x_train, y_train)
    y_test_pred = model.predict_proba(x_test)
    y_train_pred = model.predict_proba(x_train)

    return y_test_pred, y_train_pred


def meta_nn_predict(pretrained_model_path, x_test, gpu=0, batch_size=64):
    net = getattr(
        importlib.import_module(meta_models[CONFIG.META_MODEL_NAME].module_name),
        meta_models[CONFIG.META_MODEL_NAME].class_name,
    )(x_test.shape[1], **meta_models[CONFIG.META_MODEL_NAME].kwargs).cuda(gpu)
    net.load_state_dict(torch.load(pretrained_model_path)["state_dict"])
    net.eval()

    with torch.no_grad():
        out = []
        for b in torch.split(torch.from_numpy(x_test).float(), batch_size):
            b = b.cuda(gpu)
            out.append(torch.sigmoid(net(b).data.cpu()))
    return torch.cat(out).squeeze().numpy()


def compute_correlations(metrics):
    pd.options.display.float_format = "{:,.5f}".format
    df_full = pd.DataFrame(data=metrics)
    df_full = df_full.copy().drop(["class", "iou0"], axis=1)
    df_all = df_full.copy()
    df_full = df_full.copy().loc[df_full["S_in"].nonzero()[0]]
    return df_all, df_full


def compute_metrics_from_heatmap(heatmap, components, comp_id):
    n_in = np.count_nonzero(components == comp_id)
    n_bd = np.count_nonzero(components == -comp_id)
    value = np.sum(heatmap[abs(components) == comp_id]) / (n_in + n_bd)
    value_in = np.sum(heatmap[components == comp_id]) / n_in if n_in > 0 else 0
    value_bd = np.sum(heatmap[components == -comp_id]) / n_bd
    value_rel = value * (n_in + n_bd) / n_bd
    value_rel_in = value_in * n_in / n_bd
    return [value, value_in, value_bd, value_rel, value_rel_in]
