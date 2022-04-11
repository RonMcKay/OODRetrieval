import logging
import math
import os
from os.path import join
import pickle as pkl
import sys

import matplotlib.pyplot as plt
import numpy as np
from sacred import Experiment
from scipy.ndimage import label
import tqdm

from configuration import CONFIG
from src.MetaSeg.functions.in_out import get_indices, probs_gt_load
from src.datasets.a2d2 import id_to_trainid, trainid_to_name
from src.log_utils import log_config

ex = Experiment("eval_selection")

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


def get_gt(segment_indices, gt):
    cls, cls_counts = np.unique(
        gt[segment_indices[:, 0], segment_indices[:, 1]], return_counts=True
    )
    return cls[np.argsort(cls_counts)[-1]]


def return_and_update_instances(components_gt, box):
    found_instances, instance_size = np.unique(
        components_gt[box[1] : box[3], box[0] : box[2]], return_counts=True
    )
    rel_instance_size = (
        instance_size
        / np.unique(
            components_gt[np.isin(components_gt, found_instances)], return_counts=True
        )[1]
    )
    found_instances = found_instances[rel_instance_size >= 0.5]
    found_instances = found_instances[found_instances > 0]
    components_gt[np.isin(components_gt, found_instances)] = 0
    return components_gt, found_instances.shape[0]


@ex.config
def config():
    args = dict(
        embeddings_file=join(
            CONFIG.metaseg_io_path, "embeddings_128_128_densenet201.p"
        ),
        file_total_count="total_count_eval_128_128_a2d2.py",
        save_file_filtered=join(
            CONFIG.metaseg_io_path, "filtering_count_eval_128_128_a2d2.p"
        ),
        save_file_total=join(CONFIG.metaseg_io_path, "total_count_eval_128_128_a2d2.p"),
        plot_dir=join(".", "plots"),
        dpi=500,
        only_plot=False,
        min_height=128,
        min_width=128,
        plot_filetype="png",
    )

    if not os.path.exists(args["plot_dir"]):
        os.makedirs(os.path.abspath(args["plot_dir"]), exist_ok=True)


@ex.automain
def main(args, _run, _log):
    log_config(_run, _log)

    if not args["only_plot"]:
        with open(args["embeddings_file"], "rb") as f:
            data = pkl.load(f)

        image_indices = np.array(data["image_index"])
        image_level_index = np.array(data["image_level_index"])
        gt_segments = np.array(data["gt"])
        boxes = np.array(data["box"])

        inds = get_indices(
            join(CONFIG.metaseg_io_path, "input", "deeplabv3plus", "a2d2")
        )

        if args["file_total_count"] is None:
            total_num_instances = {cl: 0 for cl in id_to_trainid.keys()}
        else:
            with open(args["file_total_count"], "rb") as f:
                total_num_instances = pkl.load(f)
        filtered_num_instances = {cl: 0 for cl in id_to_trainid.keys()}

        for ind in tqdm.tqdm(inds):
            pred, gt, img_path = probs_gt_load(
                ind,
                join(CONFIG.metaseg_io_path, "input", "deeplabv3plus", "a2d2"),
                preds=True,
            )

            # count number of instances of each class of the minimum size in
            # ground truth and prediction
            for cl in np.unique(gt):
                components_gt, counts_gt = label(gt == cl)
                if args["file_total_count"] is None:
                    for c in range(1, counts_gt + 1):
                        segment_indices = np.argwhere(components_gt == c)
                        top, left = segment_indices.min(0)
                        bottom, right = segment_indices.max(0)
                        if (bottom - top) < args["min_height"] or (right - left) < args[
                            "min_width"
                        ]:
                            continue
                        else:
                            total_num_instances[cl] += 1

                if ind in image_indices:
                    for b in boxes[
                        (gt_segments == cl)
                        & (
                            image_level_index
                            == np.argwhere(image_indices == ind).squeeze()
                        ),
                        :,
                    ]:
                        components_gt, instance_counts = return_and_update_instances(
                            components_gt, b
                        )
                        filtered_num_instances[cl] += instance_counts

        _log.info("Saving file with total counts...")
        if args["file_total_count"] is None:
            with open(args["save_file_total"], "wb") as f:
                pkl.dump(total_num_instances, f)

        _log.info("Saving file with filtered counts...")
        with open(args["save_file_filtered"], "wb") as f:
            pkl.dump(filtered_num_instances, f)
    else:
        with open(args["save_file_total"], "rb") as f:
            total_num_instances = pkl.load(f)
        with open(args["save_file_filtered"], "rb") as f:
            filtered_num_instances = pkl.load(f)

    _log.info("Start plotting")

    # aggregate over training ids:
    num_instances = {k: 0 for k in trainid_to_name.keys()}
    f_num_instances = {k: 0 for k in trainid_to_name.keys()}
    for k, v in total_num_instances.items():
        num_instances[id_to_trainid[k]] += v
    for k, v in filtered_num_instances.items():
        f_num_instances[id_to_trainid[k]] += v

    sel_classes = None
    # sel_classes = [31, 22, 12, 34, 3, 35]  # classes with many extracted instances
    # sel_classes = [1, 4, 17, 24, 16, 18]  # classes with few extracted instances
    # start_angles = [45, 0, 10, 0, 0, 0]
    start_angles = [0] * 6
    fontsize = 8  # noqa: F841

    fig = plt.figure(
        "Class occurances filtered and not filtered",
        figsize=(3.3, 2.5) if sel_classes is not None else (10, 10),
        dpi=args["dpi"],
    )
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams["font.size"] = 6.0

    def label_autopct(pct, allvals):
        absolute = int(pct / 100.0 * np.sum(allvals))
        return "{:.1f}%\n({:d})".format(pct, absolute) if pct > 10 else ""

    n = math.ceil(math.sqrt(len([1 for v in num_instances.values() if v > 0])))
    cmap = plt.get_cmap("tab20c")
    for i, k in enumerate(
        [key for key, v in num_instances.items() if v > 0]
        if sel_classes is None
        else sel_classes
    ):
        if num_instances[k] > 0:
            ax = fig.add_subplot(
                n if sel_classes is None else 2, n if sel_classes is None else 3, i + 1
            )
            ax.text(
                0.5,
                1.0,
                "{}".format(
                    trainid_to_name[k]
                    if not trainid_to_name[k][-1].isdigit()
                    else trainid_to_name[k][:-2]
                ),
                horizontalalignment="center",
                transform=ax.transAxes,
                fontdict=dict(size=8),
            )
            ax.pie(
                [num_instances[k] - f_num_instances[k], f_num_instances[k]],
                radius=1.2,
                colors=cmap(np.array([10, 5])),
                startangle=start_angles[i] if sel_classes is not None else 0,
                # autopct=lambda pct: '{:1.0f}%'.format(pct) if pct > 10 else '',
                autopct=lambda pct: label_autopct(
                    pct, [num_instances[k] - f_num_instances[k], f_num_instances[k]]
                ),
                pctdistance=0.65,
                wedgeprops=dict(
                    width=1.0,
                    edgecolor="w",
                    linewidth=2,
                ),
                textprops=dict(
                    # size=fontsize,
                ),
            )
            ax.set(aspect="equal")
    fig.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.6, rect=(0.0, 0.0, 1.0, 1.0))
    plt.savefig(
        join(
            args["plot_dir"],
            "instance_counts{}.{}".format(
                "" if sel_classes is None else "_selected", args["plot_filetype"]
            ),
        ),
        dpi=args["dpi"],
    )
    _log.info(
        "Saved instance counts plot to '{}'".format(
            join(
                args["plot_dir"],
                "instance_counts{}.{}".format(
                    "" if sel_classes is None else "_selected", args["plot_filetype"]
                ),
            )
        )
    )
