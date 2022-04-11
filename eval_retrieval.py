import logging
from multiprocessing import Pool
from os.path import join
import pickle as pkl
import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from sacred import Experiment
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap

from configuration import CONFIG
from src.datasets.a2d2 import id_to_trainid, trainid_to_name
from src.log_utils import log_config

ex = Experiment("eval_retrieval")

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


def lp_dist(point, all_points, d=2):
    return ((all_points - point) ** d).sum(1) ** (1.0 / d)


def cos_dist(point, all_points):
    return 1 - ((point * all_points).sum(1) / (norm(point) * norm(all_points, axis=1)))


def ap_wrapper(arguments):
    return ap(*arguments)


def ap(gt, retrieval_list):
    result = 0
    relevant = retrieval_list == gt
    for k in range(retrieval_list.shape[0]):
        result += relevant[: (k + 1)].mean() * relevant[k]
    return result / relevant.sum()


def meanaverageprecision(
    query_indices,
    gt,
    embeddings,
    distance_metric="euclid",
    gt_annotation=None,
    void=-1,
    n_jobs=4,
):
    if gt_annotation is not None:
        gt[gt_annotation == 0] = void
    if distance_metric == "euclid":
        args = [
            (gt[q], gt[np.argsort(lp_dist(embeddings[q], embeddings))][1:])
            for q in query_indices
        ]
    elif distance_metric == "cos":
        args = [
            (gt[q], gt[np.argsort(cos_dist(embeddings[q], embeddings))][1:])
            for q in query_indices
        ]
    with Pool(n_jobs) as p:
        average_precisions = list(p.imap(ap_wrapper, args))

    return sum(average_precisions) / len(average_precisions)


@ex.config
def config():
    args = dict(  # noqa: F841
        embeddings_file=join(
            CONFIG.metaseg_io_path, "embeddings_128_128_densenet201.p"
        ),
        embedding_size=2,
        overwrite_embeddings=False,
        plot_dir=None,
        method="TSNE",
        distance_metric="euclid",
        n_jobs=10,
        min_query_count=2,
    )

    tsne = dict(  # noqa: F841
        perplexity=30,
        learning_rate=200.0,
        early_exaggeration=12.0,
        verbose=3,
    )


@ex.automain
def main(args, tsne, _run, _log):
    log_config(_run, _log)
    with open(args["embeddings_file"], "rb") as f:
        data = pkl.load(f)

    gt = np.array(data["gt"]).squeeze()
    _log.debug("Number of segments: {}".format(gt.shape[0]))

    gt = np.vectorize(id_to_trainid.get)(gt)

    if (
        data["nn_embeddings"].shape[1] != args["embedding_size"]
        if "nn_embeddings" in data.keys()
        else True
    ) or args["overwrite_embeddings"]:
        embeddings = np.stack(data["embeddings"])

        # _log.info('Standardizing embeddings...')
        # embeddings = (embeddings - embeddings.mean()) / embeddings.std()

        if (
            args["embedding_size"] < embeddings.shape[1]
            if args["embedding_size"] is not None
            else False
        ):
            _log.info("Computing embeddings for nearest neighbor search...")
            if args["method"] == "TSNE":
                _log.info(
                    "Using t-SNE with method '{}' and dimensionality {}".format(
                        "barnes_hut" if args["embedding_size"] < 4 else "exact",
                        args["embedding_size"],
                    )
                )
                embeddings = PCA(
                    n_components=50 if args["embedding_size"] < 50 else 100
                ).fit_transform(embeddings)
                embeddings = TSNE(
                    n_components=args["embedding_size"],
                    n_jobs=args["n_jobs"],
                    method="barnes_hut" if args["embedding_size"] < 4 else "exact",
                    **tsne
                ).fit_transform(embeddings)
            elif args["method"] == "Isomap":
                _log.info("Using Isomap method.")
                embeddings = PCA(
                    n_components=50 if args["embedding_size"] < 50 else 100
                ).fit_transform(embeddings)
                embeddings = Isomap(
                    n_components=args["embedding_size"],
                    n_jobs=args["n_jobs"],
                ).fit_transform(embeddings)
            elif args["method"] == "PCA":
                _log.info("Using PCA method.")
                embeddings = PCA(n_components=args["embedding_size"]).fit_transform(
                    embeddings
                )

            data["nn_embeddings"] = embeddings
            _log.debug("Saving computed manifold to embeddings file.")
            with open(args["embeddings_file"], "wb") as f:
                pkl.dump(data, f)
        else:
            _log.info("Leaving data as it is.")
    else:
        embeddings = data["nn_embeddings"]
        _log.info(
            (
                "Using precomputed embeddings "
                "({} dimensions) for nearest neighbor search...".format(
                    embeddings.shape[1]
                )
            )
        )

    embeddings = embeddings[gt != 255]
    gt = gt[gt != 255]

    if "annotated" in data:
        annotated_gt = data["annotated"]

    results = {}
    n_queries = {}
    # sel_classes = [12, 22, 3, 34]
    sel_classes = list(range(37))
    for cl in sel_classes:
        query_list = np.argwhere(gt == cl).flatten()
        if "annotated" in data and query_list.size >= args["min_query_count"]:
            query_list = np.array([q for q in query_list if annotated_gt[q] != 0])
        n_queries[cl] = (
            len(query_list) if len(query_list) >= args["min_query_count"] else 0
        )
        if query_list.size >= args["min_query_count"]:
            results[cl] = meanaverageprecision(
                query_list,
                gt,
                embeddings,
                distance_metric=args["distance_metric"],
                gt_annotation=annotated_gt if "annotated" in data else None,
                n_jobs=args["n_jobs"],
            )
            _log.info(
                "{:>{width}s} ({:>4d}): {:>7.2%}".format(
                    trainid_to_name[cl],
                    len(query_list),
                    results[cl],
                    width=max([len(str(v)) for v in trainid_to_name.values()]),
                )
            )

    _log.info("Average: {:.2%}".format(sum(results.values()) / len(results.values())))
    _log.info(
        "Weighted Average: {:.2%}".format(
            sum([v * n_queries[k] for k, v in results.items()])
            / sum(n_queries.values())
        )
    )

    if args["plot_dir"] is not None:
        _log.info("Start plotting...")
        fig = plt.figure("mAP values in % for retrieval in the embedding space")
        ax = fig.add_subplot(111)
        rects = ax.bar(
            x=np.arange(len(results) + 2),
            height=(
                [v * 100 for k, v in results.items()]
                + [sum(results.values()) / len(results.values()) * 100]
                + [
                    sum([v * n_queries[k] for k, v in results.items()])
                    / sum(n_queries.values())
                    * 100
                ]
            ),
        )
        ax.set_xticks(np.arange(len(results) + 2))
        ax.set_xticklabels(
            labels=[trainid_to_name[k] for k in results.keys()]
            + ["Average"]
            + ["Weighted Average"]
        )
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                "{:.1f}".format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )
        # ax.title.set_text('Retrieval results in the embedding space')
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.yaxis.grid(True)
        ax.xaxis.set_tick_params(rotation=50)
        ax.set_ylabel("mAP in %")
        ax.set_axisbelow(True)
        plt.savefig(
            join(args["plot_dir"], "map_plot.eps"), dpi=300, bbox_inches="tight"
        )
        _log.info(
            "Saved plot of mAP results to '{}'".format(
                join(args["plot_dir"], "map_plot.eps")
            )
        )
