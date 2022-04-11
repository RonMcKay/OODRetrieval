#!/usr/bin/env python3
"""
script including
functions for handling input/output like loading/saving
"""

import os
from os.path import join, splitext
import pickle
import re

import h5py
import numpy as np

from configuration import CONFIG


def get_save_path_input_i(i, input_dir):
    if os.path.isfile(join(input_dir, "probs_{}.hdf5".format(i))):
        file_path = join(input_dir, "probs_{}.hdf5".format(i))
    else:
        file_path = join(input_dir, "input{}.hdf5".format(i))
    return file_path


def get_save_path_metrics_i(i, metrics_dir):
    return join(metrics_dir, "metrics{}.p".format(i))


def get_save_path_components_i(i, components_dir):
    return join(components_dir, "components{}.p".format(i))


def metrics_load(i, metrics_dir):
    read_path = get_save_path_metrics_i(i, metrics_dir=metrics_dir)
    metrics = pickle.load(open(read_path, "rb"))
    return metrics


def components_load(i, components_dir):
    read_path = get_save_path_components_i(i, components_dir=components_dir)
    components = pickle.load(open(read_path, "rb"))
    return components


def components_load_all(num_imgs, components_dir=CONFIG.COMPONENTS_DIR):
    components = []
    for ind in num_imgs:
        comp = components_load(ind, components_dir=components_dir)
        components.append(comp)
    return np.stack(components).squeeze()


def metrics_dump(metrics, i, metrics_dir):
    dump_path = get_save_path_metrics_i(i, metrics_dir=metrics_dir)
    dump_dir = os.path.dirname(dump_path)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir, exist_ok=True)

    with open(dump_path, "wb") as f:
        pickle.dump(metrics, f)


def components_dump(components, i, components_dir=CONFIG.COMPONENTS_DIR):
    dump_path = get_save_path_components_i(i, components_dir=components_dir)
    dump_dir = os.path.dirname(dump_path)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir, exist_ok=True)

    with open(dump_path, "wb") as f:
        pickle.dump(components, f)


def get_img_path_fname(filename, img_dir=CONFIG.IMG_DIR):
    path = []
    for root, dirnames, filenames in os.walk(img_dir):
        for fn in filenames:
            if filename in fn:
                path = os.path.join(root, fn)
                break
    if path == []:
        print("file {} not found.".format(filename))
    return path


def image_path_load(i, input_dir=CONFIG.INPUT_DIR):
    with h5py.File(get_save_path_input_i(i, input_dir=input_dir), "r") as f_probs:
        if "file_names" in list(f_probs.keys()):
            image_path = get_img_path_fname(f_probs["file_names"][0].decode("utf8"))
        else:
            image_path = f_probs["image_path"][0].decode("utf8")
    return image_path


def probs_gt_load(i, input_dir=CONFIG.INPUT_DIR, preds=False):
    with h5py.File(get_save_path_input_i(i, input_dir=input_dir), "r") as f_probs:
        if preds:
            probs = f_probs["prediction"][()]
        else:
            probs = f_probs["probabilities"][()]
        gt = f_probs["ground_truths"][()]
        probs = np.squeeze(probs)
        gt = np.squeeze(gt)
        if "file_names" in list(f_probs.keys()):  # compatibility with metaseg 1.0
            image_path = get_img_path_fname(f_probs["file_names"][0].decode("utf8"))
        else:
            image_path = f_probs["image_path"][0].decode("utf8")
    return probs, gt, image_path


def probs_gt_load_all(num_imgs, input_dir=CONFIG.INPUT_DIR):
    if isinstance(num_imgs, int):
        num_imgs = get_indices(input_dir)[:num_imgs]
    elif isinstance(num_imgs, (list, tuple)):
        pass
    else:
        raise ValueError(
            "num_imgs should either be of type int, list or tuple "
            "but received {}".format(type(num_imgs))
        )

    probs = []
    gt = []
    image_paths = []
    for i in num_imgs:
        p, g, img_path = probs_gt_load(i, input_dir=input_dir)
        probs.append(p)
        gt.append(g)
        image_paths.append(img_path)
    return np.stack(probs), np.stack(gt), tuple(image_paths)


def stats_dump(stats, df_all, y0a):
    df_full = df_all.copy().loc[df_all["S_in"].nonzero()[0]]
    iou_corrs = df_full.corr()["iou"]
    mean_stats = dict({})
    std_stats = dict({})
    for s in stats:
        if s not in ["alphas", "n_av", "n_metrics", "metric_names"]:
            mean_stats[s] = np.mean(stats[s], axis=0)
            std_stats[s] = np.std(stats[s], axis=0)
    best_pen_ind = np.argmax(mean_stats["penalized_val_acc"])

    # dump stats latex ready
    with open(join(CONFIG.RESULTS_DIR, "av_results.txt"), "wt") as f:

        print(iou_corrs, file=f)
        print(" ", file=f)

        print("classification", file=f)
        print(
            "        & train           &  val             &    \\\\ ",
            file=f,
        )
        M = sorted([s for s in mean_stats if "penalized" in s and "acc" in s])
        print("ACC penalized               ", end=" & ", file=f)
        for s in M:
            print(
                "${:.2f}\%".format(100 * mean_stats[s][best_pen_ind])  # noqa: W605
                + "(\pm{:.2f}\%)$".format(  # noqa: W605
                    100 * std_stats[s][best_pen_ind]
                ),
                end=" & ",
                file=f,
            )
        print("   \\\\ ", file=f)
        M = sorted([s for s in mean_stats if "plain" in s and "acc" in s])
        print("ACC unpenalized             ", end=" & ", file=f)
        for s in M:
            print(
                "${:.2f}\%".format(100 * mean_stats[s][best_pen_ind])  # noqa: W605
                + "(\pm{:.2f}\%)$".format(  # noqa: W605
                    100 * std_stats[s][best_pen_ind]
                ),
                end=" & ",
                file=f,
            )
        print("   \\\\ ", file=f)
        M = sorted([s for s in mean_stats if "entropy" in s and "acc" in s])
        print("ACC entropy baseline        ", end=" & ", file=f)
        for s in M:
            print(
                "${:.2f}\%".format(100 * mean_stats[s])  # noqa: W605
                + "(\pm{:.2f}\%)$".format(100 * std_stats[s]),  # noqa: W605
                end=" & ",
                file=f,
            )
        print("   \\\\ ", file=f)

        M = sorted([s for s in mean_stats if "penalized" in s and "auroc" in s])
        print("AUROC penalized             ", end=" & ", file=f)
        for s in M:
            print(
                "${:.2f}\%".format(100 * mean_stats[s][best_pen_ind])  # noqa: W605
                + "(\pm{:.2f}\%)$".format(  # noqa: W605
                    100 * std_stats[s][best_pen_ind]
                ),
                end=" & ",
                file=f,
            )
        print("   \\\\ ", file=f)
        M = sorted([s for s in mean_stats if "plain" in s and "auroc" in s])
        print("AUROC unpenalized           ", end=" & ", file=f)
        for s in M:
            print(
                "${:.2f}\%".format(100 * mean_stats[s][best_pen_ind])  # noqa: W605
                + "(\pm{:.2f}\%)$".format(  # noqa: W605
                    100 * std_stats[s][best_pen_ind]
                ),
                end=" & ",
                file=f,
            )
        print("   \\\\ ", file=f)
        M = sorted([s for s in mean_stats if "entropy" in s and "auroc" in s])
        print("AUROC entropy baseline      ", end=" & ", file=f)
        for s in M:
            print(
                "${:.2f}\%".format(100 * mean_stats[s])  # noqa: W605
                + "(\pm{:.2f}\%)$".format(100 * std_stats[s]),  # noqa: W605
                end=" & ",
                file=f,
            )
        print("   \\\\ ", file=f)

        print(" ", file=f)
        print("regression", file=f)

        M = sorted(
            [s for s in mean_stats if "regr" in s and "mse" in s and "entropy" not in s]
        )
        print("$\sigma$, all metrics       ", end=" & ", file=f)  # noqa: W605
        for s in M:
            print(
                "${:.3f}".format(mean_stats[s])
                + "(\pm{:.3f})$".format(std_stats[s]),  # noqa: W605
                end="    & ",
                file=f,
            )
        print("   \\\\ ", file=f)
        M = sorted(
            [s for s in mean_stats if "regr" in s and "mse" in s and "entropy" in s]
        )
        print("$\sigma$, entropy baseline  ", end=" & ", file=f)  # noqa: W605
        for s in M:
            print(
                "${:.3f}".format(mean_stats[s])
                + "(\pm{:.3f})$".format(std_stats[s]),  # noqa: W605
                end="    & ",
                file=f,
            )
        print("   \\\\ ", file=f)

        M = sorted(
            [s for s in mean_stats if "regr" in s and "r2" in s and "entropy" not in s]
        )
        print("$R^2$, all metrics          ", end=" & ", file=f)
        for s in M:
            print(
                "${:.2f}\%".format(100 * mean_stats[s])  # noqa: W605
                + "(\pm{:.2f}\%)$".format(100 * std_stats[s]),  # noqa: W605
                end=" & ",
                file=f,
            )
        print("   \\\\ ", file=f)
        M = sorted(
            [s for s in mean_stats if "regr" in s and "r2" in s and "entropy" in s]
        )
        print("$R^2$, entropy baseline     ", end=" & ", file=f)
        for s in M:
            print(
                "${:.2f}\%".format(100 * mean_stats[s])  # noqa: W605
                + "(\pm{:.2f}\%)$".format(100 * std_stats[s]),  # noqa: W605
                end=" & ",
                file=f,
            )
        print("   \\\\ ", file=f)

        print(" ", file=f)
        M = sorted([s for s in mean_stats if "iou" in s])
        for s in M:
            print(
                s,
                ": {:.0f}".format(mean_stats[s])
                + "($\pm${:.0f})".format(std_stats[s]),  # noqa: W605
                file=f,
            )
        print(
            "IoU=0:",
            np.sum(y0a == 1),
            "of",
            y0a.shape[0],
            "non-empty components",
            file=f,
        )
        print(
            "IoU>0:",
            np.sum(y0a == 0),
            "of",
            y0a.shape[0],
            "non-empty components",
            file=f,
        )
        print("total number of components: ", len(df_all), file=f)
        print(" ", file=f)

        dump_dir = CONFIG.STATS_DIR
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir, exist_ok=True)

        with open(join(dump_dir, "stats.p"), "wb") as f:
            pickle.dump(stats, f)
    return mean_stats, std_stats


def get_indices(directory):
    files = [f for f in os.listdir(directory) if splitext(f)[-1] in (".hdf5", ".p")]
    inds = [int(re.match(".*?([0-9]+)$", splitext(f)[0]).group(1)) for f in files]
    inds.sort()
    return inds
