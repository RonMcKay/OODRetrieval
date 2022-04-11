import logging
from multiprocessing import Pool
import os
from os.path import join
import pickle as pkl
import sys

import matplotlib.pyplot as plt
import numpy as np
from sacred import Experiment
import tqdm

from configuration import CONFIG
from src.MetaSeg.functions.calculate import meta_nn_predict
from src.MetaSeg.functions.helper import load_data
from src.MetaSeg.functions.in_out import components_load, get_indices, probs_gt_load
from src.datasets.a2d2 import a2d2_to_cityscapes
from src.datasets.cityscapes import id_to_catid, num_categories, trainid_to_catid
from src.eval_utils import iou_numpy as iou
from src.log_utils import log_config

ex = Experiment("eval_iou")
log = logging.getLogger()
log.handlers = []

log_format = logging.Formatter(
    "%(asctime)s || %(name)s - [%(levelname)s] - %(message)s"
)

streamhandler = logging.StreamHandler(sys.stdout)
streamhandler.setFormatter(log_format)
log.addHandler(streamhandler)

log.setLevel("INFO")
ex.logger = log

# build mapping from a2d2 full ID set to cityscapes training ID set
label_mappings = dict(
    a2d2={k: id_to_catid[a2d2_to_cityscapes[k]] for k in a2d2_to_cityscapes.keys()},
    cityscapes_val=trainid_to_catid,
)


def pool_wrapper(inputs):
    return get_ious_for_image(*inputs)


@ex.capture
def get_ious_for_image(image_index, iou_pred, thresholds, args):
    confusion_matrices_pos = {
        t: np.zeros((num_categories, num_categories)) for t in thresholds
    }
    confusion_matrices_neg = {
        t: np.zeros((num_categories, num_categories)) for t in thresholds
    }

    pred, gt, _ = probs_gt_load(
        image_index,
        input_dir=join(
            CONFIG.metaseg_io_path, "input", "deeplabv3plus", args["dataset"]
        ),
        preds=True,
    )

    # transform a2d2 labels to cityscapes category ids
    gt = np.vectorize(label_mappings[args["dataset"]].get)(gt)

    # transform predictions to cityscapes category ids
    pred = np.vectorize(trainid_to_catid.get)(pred)

    # load components for constructing the iou mask based on different IoU thresholds
    components = components_load(
        image_index,
        components_dir=join(
            CONFIG.metaseg_io_path, "components", "deeplabv3plus", args["dataset"]
        ),
    )

    # border of components have been labeled with the negative index of the
    # main component itself we want however to include the border of the segment in
    # the evaluation which is why we have to make it also positive
    components = np.absolute(components)

    # -1 because component indices start with 1
    components = iou_pred[components - 1]

    for t in thresholds:
        # confusion_matrices_pos[t] = iou(pred,
        #                                 gt,
        #                                 n_classes=num_categories,
        #                                 update_matrix=confusion_matrices_pos[t],
        #                                 ignore_index=0,
        #                                 mask=(components >= t))[1]
        confusion_matrices_neg[t] = iou(
            pred,
            gt,
            n_classes=num_categories,
            update_matrix=confusion_matrices_neg[t],
            ignore_index=0,
            mask=(components < t),
        )[1]

    return confusion_matrices_pos, confusion_matrices_neg


@ex.config
def config():
    args = dict(
        meta_nn_path=join(".", "src", "meta_nn.pth"),
        save_dir=CONFIG.metaseg_io_path,
        load_file=None,
        plot_dir=join(".", "plots"),
        steps=51,
        gpu=CONFIG.GPU_ID,
        dpi=400,
        n_workers=CONFIG.NUM_CORES,
        max_t=0.75,
        dataset="a2d2",
    )

    if not os.path.exists(args["plot_dir"]):
        os.makedirs(os.path.abspath(args["plot_dir"]), exist_ok=True)


def iou_wrapper(inputs):
    return iou(*inputs)


@ex.automain
def main(args, _run, _log):
    log_config(_run, _log)

    if args["load_file"] is None:
        _log.info("Loading data...")
        _log.info("Cityscapes...")

        # load cityscapes train data for normalization of out of domain data
        _, _, _, _, xa_mean, xa_std, classes_mean, classes_std, *_ = load_data(
            "cityscapes"
        )

        _log.info("{}...".format(args["dataset"]))
        xa, *_, start, _ = load_data(
            args["dataset"],
            xa_mean=xa_mean,
            xa_std=xa_std,
            classes_mean=classes_mean,
            classes_std=classes_std,
        )

        # predict iou using MetaSeg metrics
        iou_pred = meta_nn_predict(args["meta_nn_path"], xa, gpu=args["gpu"])

        # get all available input file IDs
        inds = get_indices(
            join(CONFIG.metaseg_io_path, "metrics", "deeplabv3plus", args["dataset"])
        )

        # construct thresholds and dictionary for saving
        thresholds = np.linspace(0, 1, args["steps"])
        confusion_matrices = dict(
            pos={t: np.zeros((num_categories, num_categories)) for t in thresholds},
            neg={t: np.zeros((num_categories, num_categories)) for t in thresholds},
        )

        with open("/data/poberdie/metaseg/confusion_matrices.p", "rb") as f:
            confusion_matrices["pos"] = pkl.load(f)

        _log.info("Calculating IoUs...")
        inputs = [
            (ind, iou_pred[start[i] : start[i + 1]], thresholds)
            for i, ind in enumerate(inds)
        ]
        with Pool(args["n_workers"]) as p:
            res = list(tqdm.tqdm(p.imap(pool_wrapper, inputs), total=len(inputs)))
            for r in res:
                # for t, v in r[0].items():
                #     confusion_matrices['pos'][t] += v
                for t, v in r[1].items():
                    confusion_matrices["neg"][t] += v

        with open(
            join(args["save_dir"], "confusion_matrices_{}.p".format(args["dataset"])),
            "wb",
        ) as f:
            pkl.dump(confusion_matrices, f)
    else:
        thresholds = np.linspace(0, 1, args["steps"])
        with open(args["load_file"], "rb") as f:
            confusion_matrices = pkl.load(f)

    _log.info("Start plotting")
    ious_pos = []
    for t, v in confusion_matrices["pos"].items():
        tp = np.diag(v)[1:]
        fn = v.sum(0)[1:] - tp
        fp = v.sum(1)[1:] - tp
        ious_pos.append((t, (tp / (tp + fp + fn + 1e-6)).mean() * 100))

    ious_pos.sort(key=lambda x: x[0])
    ious_pos = [i[1] for i in ious_pos if i[0] < args["max_t"]]

    ious_neg = []
    for t, v in confusion_matrices["neg"].items():
        tp = np.diag(v)[1:]
        fn = v.sum(0)[1:] - tp
        fp = v.sum(1)[1:] - tp
        ious_neg.append((t, (tp / (tp + fp + fn + 1e-6)).mean() * 100))

    ious_neg.sort(key=lambda x: x[0])
    ious_neg = [i[1] for i in ious_neg if i[0] < args["max_t"]]

    colmap = plt.get_cmap("tab20c")

    fig = plt.figure("IoU under different thresholds", dpi=args["dpi"])

    ax = fig.add_subplot(111)
    ax.set_axisbelow(True)
    ax.grid(linestyle="--")
    ax.set_xlabel("Threshold t on predicted IoU")
    ax.set_ylabel("mIoU in % for segments below t", color=colmap(4))
    ax.tick_params(axis="y", labelcolor=colmap(4))

    ax2 = ax.twinx()
    ax2.set_ylabel("mIoU in % for segments above t", color=colmap(8))
    ax2.tick_params(axis="y", labelcolor=colmap(8))

    ax.plot(
        thresholds[thresholds < args["max_t"]],
        np.array(ious_neg),
        color=colmap(4),
        linestyle="-",
    )
    ax2.plot(
        thresholds[thresholds < args["max_t"]],
        np.array(ious_pos),
        color=colmap(8),
        linestyle="-",
    )

    plt.savefig(
        join(args["plot_dir"], "iou_plot_{}.png".format(args["dataset"])),
        dpi=args["dpi"],
        bbox_inches="tight",
    )
    _log.info(
        "Saved plot to '{}'".format(
            join(args["plot_dir"], "iou_plot_{}.png".format(args["dataset"]))
        )
    )
