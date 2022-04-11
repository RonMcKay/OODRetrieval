import pyximport

pyximport.install()

from collections import OrderedDict
import importlib
import logging
import math
import os
from os.path import expanduser, join
import pickle as pkl

from PIL import Image, ImageDraw
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from tqdm import tqdm

from configuration import CONFIG, datasets
from src.MetaSeg.functions.in_out import components_load, probs_gt_load
from src.MetaSeg.functions.metrics import entropy
from src.MetaSeg.functions.utils import estimate_kernel_density


# noinspection DuplicatedCode
class Discovery(object):
    def __init__(
        self,
        embeddings_file="embeddings.p",
        distance_metric="euclid",
        method="TSNE",
        embedding_size=2,
        overwrite_embeddings=False,
        n_jobs=10,
        dpi=300,
        main_plot_args={},
        tsne_args={},
        save_dir=join(CONFIG.metaseg_io_path, "vis_embeddings"),
    ):
        """Loads the embedding files, computes the dimensionality reductions and calls
        the initilization of the main plot.

        Args:
            embeddings_file (str): Path to the file where all data of segments
                including feature embeddings is saved.
            distance_metric (str): Distance metric to use for nearest neighbor
                computation.
            method (str): Method to use for dimensionality reduction of nearest
                neighbor embeddings. For plotting the points are always reduced in
                dimensionality using PCA to 50 dimensions followed by t-SNE to two
                dimensions.
            embedding_size (int): Dimensionality of the feature embeddings used for
                nearest neighbor search.
            overwrite_embeddings (bool): If True, precomputed nearest neighbor and
                plotting embeddings from previous runs are overwritten with freshly
                computed ones. Otherwise precomputed embeddings are used if requested
                embedding_size is matching.
            n_jobs (int): Number of processes to use for t-SNE computation.
            dpi (int): Dots per inch for graphics that are saved to disk.
            main_plot_args (dict): Keyword arguments for the creation of the main plot.
            tsne_args (dict): Keyword arguments for the t-SNE algorithm.
            save_dir (str): Path to the directory where saved images are placed in.
        """
        self.log = logging.getLogger("Discovery")
        self.embeddings_file = embeddings_file
        self.distance_metrics = ["euclid", "cos"]
        self.dm = (
            0
            if distance_metric not in self.distance_metrics
            else self.distance_metrics.index(distance_metric)
        )

        self.dpi = dpi
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.cluster_methods = OrderedDict()
        self.cluster_methods["kmeans"] = {"main": KMeans, "kwargs": {}}
        self.cluster_methods["spectral"] = {"main": SpectralClustering, "kwargs": {}}
        self.cluster_methods["agglo"] = {
            "main": AgglomerativeClustering,
            "kwargs": {"linkage": "ward"},
        }

        self.methods_with_ncluster_param = ["kmeans", "spectral", "agglo"]
        self.cme = 0
        self.clustering = None
        self.n_clusters = 25

        # colors:
        self.standard_color = (0, 0, 1, 1)
        self.current_color = (1, 0, 0, 1)
        self.nn_color = (1, 0, 1, 1)

        self.log.info("Loading data...")
        with open(self.embeddings_file, "rb") as f:
            self.data = pkl.load(f)

        self.iou_preds = self.data["iou_pred"]
        self.gt = np.array(self.data["gt"]).flatten()
        self.pred = np.array(self.data["pred"]).flatten()
        self.gi = self.data[
            "image_level_index"
        ]  # global indices (on image level and not on component level)

        self.log.info("Loaded {} segment embeddings.".format(self.pred.shape[0]))

        self.nearest_neighbors = None

        if len(self.data["embeddings"]) == 1:
            self.data["plot_embeddings"] = np.array(
                [self.data["embeddings"][0][0], self.data["embeddings"][0][1]]
            ).reshape((1, 2))
            self.data["nn_embeddings"] = self.data["plot_embeddings"]
        else:
            if (
                "nn_embeddings" not in self.data.keys()
                or overwrite_embeddings
                or "plot_embeddings" not in self.data.keys()
            ) and embedding_size < self.data["embeddings"][0].shape[0]:
                self.log.info("Computing PCA...")
                n_comp = (
                    50
                    if 50
                    < min(
                        len(self.data["embeddings"]),
                        self.data["embeddings"][0].shape[0],
                    )
                    else min(
                        len(self.data["embeddings"]),
                        self.data["embeddings"][0].shape[0],
                    )
                )
                embeddings = PCA(n_components=n_comp).fit_transform(
                    np.stack(self.data["embeddings"]).reshape(
                        (-1, self.data["embeddings"][0].shape[0])
                    )
                )
                rewrite = True
            else:
                rewrite = False

            if "plot_embeddings" not in self.data.keys() or overwrite_embeddings:
                self.log.info("Computing t-SNE for plotting")
                self.data["plot_embeddings"] = TSNE(
                    n_components=2, **tsne_args
                ).fit_transform(embeddings)
                new_plot_embeddings = True
            else:
                new_plot_embeddings = False

            if (
                embedding_size >= self.data["embeddings"][0].shape[0]
                or embedding_size is None
            ):
                self.embeddings = np.stack(self.data["embeddings"]).reshape(
                    (-1, self.data["embeddings"][0].shape[0])
                )
                self.log.debug(
                    (
                        "Requested embedding size of {} was greater or equal "
                        "to data dimensionality of {}. "
                        "Data has thus not been reduced in dimensionality."
                    ).format(embedding_size, self.data["embeddings"].shape[1])
                )
            elif (
                self.data["nn_embeddings"].shape[1] == embedding_size
                if "nn_embeddings" in self.data.keys()
                else False
            ) and not overwrite_embeddings:
                self.embeddings = self.data["nn_embeddings"]
                self.log.info(
                    (
                        "Loaded reduced embeddings "
                        "({} dimensions) from precomputed file "
                        + "for nearest neighbor search."
                    ).format(self.embeddings.shape[1])
                )
            elif rewrite:
                if method == "TSNE":
                    if (
                        "plot_embeddings" in self.data.keys()
                        and embedding_size == 2
                        and new_plot_embeddings
                    ):
                        self.embeddings = self.data["plot_embeddings"]
                        self.log.info(
                            (
                                "Reused the precomputed manifold for plotting for "
                                "nearest neighbor search."
                            )
                        )
                    else:
                        self.log.info(
                            (
                                "Computing t-SNE of dimension "
                                "{} for nearest neighbor search..."
                            ).format(embedding_size)
                        )
                        self.embeddings = TSNE(
                            n_components=embedding_size, n_jobs=n_jobs, **tsne_args
                        ).fit_transform(embeddings)
                else:
                    self.log.info(
                        (
                            "Computing Isomap of dimension "
                            "{} for nearest neighbor search..."
                        ).format(embedding_size)
                    )
                    self.embeddings = Isomap(
                        n_components=embedding_size,
                        n_jobs=n_jobs,
                    ).fit_transform(embeddings)
                self.data["nn_embeddings"] = self.embeddings
            else:
                raise ValueError(
                    (
                        "Please specify a valid combination of arguments.\n"
                        "Loading fails if 'overwrite_embeddings' is False and "
                        "saved 'embedding_size' does not match the requested one."
                    )
                )

            # Write added data into pickle file
            if rewrite:
                with open(self.embeddings_file, "wb") as f:
                    pkl.dump(self.data, f)

        self.x = self.data["plot_embeddings"][:, 0]
        self.y = self.data["plot_embeddings"][:, 1]

        self.label_mapping = dict()
        for d in np.unique(self.data["dataset"]).flatten():
            try:
                self.label_mapping[d] = getattr(
                    importlib.import_module(datasets[d].module_name),
                    datasets[d].class_name,
                )(
                    **datasets[d].kwargs,
                ).label_mapping
            except AttributeError:
                self.label_mapping[d] = None

        train_dat = self.label_mapping[CONFIG.TRAIN_DATASET.name] = getattr(
            importlib.import_module(CONFIG.TRAIN_DATASET.module_name),
            CONFIG.TRAIN_DATASET.class_name,
        )(
            **CONFIG.TRAIN_DATASET.kwargs,
        )
        self.pred_mapping = train_dat.pred_mapping
        if CONFIG.TRAIN_DATASET.name not in self.label_mapping:
            self.label_mapping[CONFIG.TRAIN_DATASET.name] = train_dat.label_mapping

        self.tnsize = (50, 50)
        self.fig_nn = None
        self.fig_main = None
        self.line_main = None
        self.im = None
        self.xybox = None
        self.ab = None
        self.basecolors = np.stack(
            [self.standard_color for _ in range(self.x.shape[0])]
        )
        self.n_neighbors = 49
        self.current_pressed_key = None

        self.plot_main(**main_plot_args)

    def plot_main(self, **plot_args):
        """Initializes the main plot.

        Only 'legend' (bool) is currently supported as keyword argument.
        """
        self.fig_main = plt.figure(num=1)
        self.fig_main.canvas.set_window_title("Embedding space")
        ax = self.fig_main.add_subplot(111)
        ax.set_axis_off()
        self.line_main = ax.scatter(
            self.x, self.y, marker="o", color=self.basecolors, zorder=2
        )
        self.line_main.set_picker(True)

        if (
            (
                plot_args["legend"]
                and all(lm is not None for lm in self.label_mapping.values())
            )
            if "legend" in plot_args
            else False
        ):
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
            legend_elements = []
            for d in np.unique(self.data["dataset"]).flatten():
                cls = np.unique(self.gt[np.array(self.data["dataset"])[self.gi] == d])
                cls = list(
                    {
                        (self.label_mapping[d][cl][0], self.label_mapping[d][cl][1])
                        for cl in cls
                    }
                )
                names = np.array([i[0] for i in cls])
                cols = np.array([i[1] for i in cls])
                legend_elements += [
                    Patch(
                        color=tuple(i / 255.0 for i in cols[i]) + (1.0,),
                        label=names[i]
                        if not names[i][-1].isdigit()
                        else names[i][: names[i].rfind(" ")],
                    )
                    for i in range(names.shape[0])
                ]
            ax.legend(
                loc="upper left",
                handles=legend_elements,
                ncol=8,
                bbox_to_anchor=(0, 1.2),
            )
        self.basecolors = self.line_main.get_facecolor()

        tmp = (
            Image.open(self.data["image_path"][self.gi[0]])
            .convert("RGB")
            .crop(self.data["box"][0])
        )
        tmp.thumbnail(self.tnsize, Image.ANTIALIAS)
        self.im = OffsetImage(tmp, zoom=2)
        self.xybox = (50.0, 50.0)
        self.ab = AnnotationBbox(
            self.im,
            (0, 0),
            xybox=self.xybox,
            xycoords="data",
            boxcoords="offset points",
            pad=0.3,
            arrowprops=dict(arrowstyle="->"),
        )
        ax.add_artist(self.ab)
        self.ab.set_visible(False)

        if plot_args["save_path"] is not None if "save_path" in plot_args else False:
            plt.savefig(
                expanduser(plot_args["save_path"]), dpi=300, bbox_inches="tight"
            )

        else:
            self.fig_main.canvas.mpl_connect("motion_notify_event", self.hover_main)
            self.fig_main.canvas.mpl_connect("button_press_event", self.click_main)
            self.fig_main.canvas.mpl_connect("scroll_event", self.scroll)
            self.fig_main.canvas.mpl_connect("key_press_event", self.key_press)
            self.fig_main.canvas.mpl_connect("key_release_event", self.key_release)
            plt.show()

    def hover_main(self, event):
        """Action handler for the main plot.

        This function shows a thumbnail of the underlying image when a scatter point
        is hovered with the mouse.
        """
        # if the mouse is over the scatter points
        if self.line_main.contains(event)[0]:
            # find out the index within the array from the event
            ind, *_ = self.line_main.contains(event)[1]["ind"]

            # get the figure size
            w, h = self.fig_main.get_size_inches() * self.fig_main.dpi
            ws = (event.x > w / 2.0) * -1 + (event.x <= w / 2.0)
            hs = (event.y > h / 2.0) * -1 + (event.y <= h / 2.0)

            # if event occurs in the top or right quadrant of the figure,
            # change the annotation box position relative to mouse.
            self.ab.xybox = (self.xybox[0] * ws, self.xybox[1] * hs)

            # make annotation box visible
            self.ab.set_visible(True)

            # place it at the position of the hovered scatter point
            self.ab.xy = (self.x[ind], self.y[ind])

            # set the image corresponding to that point
            tmp = (
                Image.open(self.data["image_path"][self.gi[ind]])
                .convert("RGB")
                .crop(self.data["box"][ind])
            )
            tmp.thumbnail(self.tnsize, Image.ANTIALIAS)
            self.im.set_data(tmp)
            tmp.close()
        else:
            # if the mouse is not over a scatter point
            self.ab.set_visible(False)
        self.fig_main.canvas.draw_idle()

    def click_main(self, event):
        """Action handler for the main plot.

        This function shows a single or full image or displays nearest neighbors based
        on the button that has been pressed and which scatter point was pressed.
        """
        if self.line_main.contains(event)[0]:
            ind, *_ = self.line_main.contains(event)[1]["ind"]

            if self.current_pressed_key == "t" and event.button == 1:
                self.store_thumbnail(ind)
            elif self.current_pressed_key == "control" and event.button == 1:
                self.show_single_image(ind, save=True)
            elif self.current_pressed_key == "control" and event.button == 2:
                self.show_full_image(ind, save=True)
            elif event.button == 1:  # left mouse button
                self.show_single_image(ind)
            elif event.button == 2:  # middle mouse button
                self.show_full_image(ind)
            elif event.button == 3:  # right mouse button
                if not plt.fignum_exists(2):
                    # nearest neighbor figure is not open anymore or has not been
                    # opened yet
                    self.log.info("Loading nearest neighbors...")
                    self.nearest_neighbors = self.get_nearest_neighbors(
                        ind, metric=self.distance_metrics[self.dm]
                    )
                    thumbnails = []
                    for neighbor_ind in self.nearest_neighbors:
                        thumbnails.append(
                            Image.open(
                                self.data["image_path"][self.gi[neighbor_ind]]
                            ).crop(self.data["box"][neighbor_ind])
                        )
                    columns = math.ceil(math.sqrt(self.n_neighbors))
                    rows = math.ceil(self.n_neighbors / columns)

                    self.fig_nn = plt.figure(num=2, dpi=self.dpi)
                    self.fig_nn.canvas.set_window_title(
                        "{} nearest neighbors to selected image".format(
                            self.n_neighbors
                        )
                    )
                    for p in range(columns * rows):
                        ax = self.fig_nn.add_subplot(rows, columns, p + 1)
                        ax.set_axis_off()
                        if p < len(thumbnails):
                            ax.imshow(np.asarray(thumbnails[p]))
                    self.fig_nn.canvas.mpl_connect("button_press_event", self.click_nn)
                    self.fig_nn.canvas.mpl_connect("key_press_event", self.key_press)
                    self.fig_nn.canvas.mpl_connect(
                        "key_release_event", self.key_release
                    )
                    self.fig_nn.canvas.mpl_connect("scroll_event", self.scroll)
                    self.fig_nn.show()
                else:
                    # nearest neighbor figure is already open. Update the figure with
                    # new nearest neighbor
                    self.update_nearest_neighbors(ind)
                    return

            self.set_color(ind, self.current_color)
            self.flush_colors()

    def click_nn(self, event):
        """Action handler for the nearest neighbor window.

        When clicking a cropped segment in the nearest neighbor window the same actions
        are taken as in the click handler for the main plot.
        """
        if event.inaxes in self.fig_nn.axes:
            ind = self.get_ind_nn(event)

            if self.current_pressed_key == "t" and event.button == 1:
                self.store_thumbnail(self.nearest_neighbors[ind])
            elif self.current_pressed_key == "control" and event.button == 1:
                self.show_single_image(self.nearest_neighbors[ind], save=True)
            elif self.current_pressed_key == "control" and event.button == 2:
                self.show_full_image(self.nearest_neighbors[ind], save=True)
            elif event.button == 1:  # left mouse button
                self.show_single_image(self.nearest_neighbors[ind])
            elif event.button == 2:  # middle mouse button
                self.show_full_image(self.nearest_neighbors[ind])
            elif event.button == 3:  # right mouse button
                self.update_nearest_neighbors(self.nearest_neighbors[ind])

    def key_press(self, event):
        """Performs different actions based on pressed keys."""
        self.log.debug("Key '{}' pressed.".format(event.key))
        if event.key == "m":
            self.dm += 1
            self.dm = self.dm % len(self.distance_metrics)
            self.log.info(
                "Changed distance metric to {}".format(self.distance_metrics[self.dm])
            )
        elif event.key == "#":
            self.cme += 1
            self.cme = self.cme % len(self.cluster_methods)
            self.log.info(
                "Changed clustering method to {}".format(
                    list(self.cluster_methods.keys())[self.cme]
                )
            )
        elif event.key == "c":
            self.log.info(
                "Started clustering with {}...".format(
                    list(self.cluster_methods.keys())[self.cme]
                )
            )
            self.cluster(method=list(self.cluster_methods.keys())[self.cme])
            if self.fig_main.axes[0].get_legend() is not None:
                self.fig_main.axes[0].get_legend().remove()
            self.basecolors = cm.get_cmap("viridis", (max(self.clustering) + 1))(
                self.clustering
            )
            self.flush_colors()
        elif event.key == "g":
            self.color_gt()
        elif event.key == "h":
            self.color_pred()
        elif event.key == "b":
            self.set_color(list(range(self.basecolors.shape[0])), self.standard_color)
            if self.fig_main.axes[0].get_legend() is not None:
                self.fig_main.axes[0].get_legend().remove()
            self.flush_colors()
        elif event.key == "d":
            self.show_density()

        self.current_pressed_key = event.key

    def key_release(self, event):
        """Clears the variable where the last pressed key is saved."""
        self.current_pressed_key = None
        self.log.debug("Key '{}' released.".format(event.key))

    def scroll(self, event):
        """Increases or decreases number of nearest neighbors when scrolling on
        the main or nearest neighbor plot."""
        if event.button == "up":
            self.n_neighbors += 1
            self.log.info(
                "Increased number of nearest neighbors to {}".format(self.n_neighbors)
            )
        elif event.button == "down":
            if self.n_neighbors > 0:
                self.n_neighbors -= 1
                self.log.info(
                    "Decreased number of nearest neighbors to {}".format(
                        self.n_neighbors
                    )
                )

    def show_single_image(self, ind, save=False):
        """Displays the full image belonging to a segment. The segment is marked with
        a red bounding box."""
        self.log.info("{} image...".format("Saving" if save else "Loading"))
        img_box = self.draw_box_on_image(ind)
        fig_tmp = plt.figure(max(3, max(plt.get_fignums()) + 1), dpi=self.dpi)
        ax = fig_tmp.add_subplot(111)
        ax.set_axis_off()
        ax.imshow(np.asarray(img_box), interpolation="nearest")
        if save:
            fig_tmp.subplots_adjust(
                bottom=0, left=0, right=1, top=1, hspace=0, wspace=0
            )
            ax.margins(0.05, 0.05)
            fig_tmp.gca().xaxis.set_major_locator(plt.NullLocator())
            fig_tmp.gca().yaxis.set_major_locator(plt.NullLocator())
            fig_tmp.savefig(
                join(self.save_dir, "image_{}.jpg".format(ind)),
                bbox_inches="tight",
                pad_inches=0.0,
            )
            self.log.debug(
                "Saved image to {}".format(
                    join(self.save_dir, "image_{}.jpg".format(ind))
                )
            )
        else:
            fig_tmp.canvas.set_window_title(
                "Dataset: {}, Image index: {}".format(
                    self.data["dataset"][self.gi[ind]],
                    self.data["image_index"][self.gi[ind]],
                )
            )
            fig_tmp.tight_layout(pad=0.0)
            fig_tmp.show()

    def show_full_image(self, ind, save=False):
        """Displays four panels of the full image belonging to a segment.

        Top left: Entropy heatmap of prediction.
        Top right: Predicted IoU of each segment.
        Bottom left: Source image with ground truth overlay.
        Bottom right: Predicted semantic segmentation.
        """
        self.log.info("{} detailed image...".format("Saving" if save else "Loading"))
        box = self.data["box"][ind]
        image = np.asarray(
            Image.open(self.data["image_path"][self.gi[ind]]).convert("RGB")
        )
        image_index = self.data["image_index"][self.gi[ind]]
        iou_pred = self.data["iou_pred"][self.gi[ind]]
        dataset = self.data["dataset"][self.gi[ind]]
        model_name = self.data["model_name"][self.gi[ind]]

        pred, gt, image_path = probs_gt_load(
            image_index,
            input_dir=join(CONFIG.metaseg_io_path, "input", model_name, dataset),
        )
        components = components_load(
            image_index,
            components_dir=join(
                CONFIG.metaseg_io_path, "components", model_name, dataset
            ),
        )
        e = entropy(pred)
        pred = pred.argmax(2)
        predc = np.asarray(
            [
                self.pred_mapping[pred[ind_i, ind_j]][1]
                for ind_i in range(pred.shape[0])
                for ind_j in range(pred.shape[1])
            ]
        ).reshape(image.shape)
        overlay_factor = [1.0, 0.5, 1.0]

        if self.label_mapping[dataset] is not None:
            gtc = np.asarray(
                [
                    self.label_mapping[dataset][gt[ind_i, ind_j]][1]
                    for ind_i in range(gt.shape[0])
                    for ind_j in range(gt.shape[1])
                ]
            ).reshape(image.shape)
        else:
            gtc = np.zeros_like(image)
            overlay_factor[1] = 0.0

        img_predc, img_gtc, img_entropy = [
            Image.fromarray(
                np.uint8(arr * overlay_factor[i] + image * (1 - overlay_factor[i]))
            )
            for i, arr in enumerate([predc, gtc, cm.jet(e)[:, :, :3] * 255.0])
        ]

        img_ioupred = Image.fromarray(self.visualize_segments(components, iou_pred))

        images = [img_gtc, img_predc, img_entropy, img_ioupred]

        box_line_width = 5
        left, upper = max(0, box[0] - box_line_width), max(0, box[1] - box_line_width)
        right, lower = min(pred.shape[1], box[2] + box_line_width), min(
            pred.shape[0], box[3] + box_line_width
        )

        for k in images:
            draw = ImageDraw.Draw(k)
            draw.rectangle(
                [left, upper, right, lower], outline=(255, 0, 0), width=box_line_width
            )
            del draw

        for k in range(len(images)):
            images[k] = np.asarray(images[k]).astype("uint8")

        img_top = np.concatenate(images[2:], axis=1)
        img_bottom = np.concatenate(images[:2], axis=1)

        img_total = np.concatenate((img_top, img_bottom), axis=0)
        fig_tmp = plt.figure(max(3, max(plt.get_fignums()) + 1), dpi=self.dpi)
        fig_tmp.canvas.set_window_title(
            "Dataset: {}, Image index: {}".format(dataset, image_index)
        )
        ax = fig_tmp.add_subplot(111)
        ax.set_axis_off()
        ax.imshow(img_total, interpolation="nearest")

        if save:
            fig_tmp.subplots_adjust(
                bottom=0, left=0, right=1, top=1, hspace=0, wspace=0
            )
            ax.margins(0.05, 0.05)
            fig_tmp.gca().xaxis.set_major_locator(plt.NullLocator())
            fig_tmp.gca().yaxis.set_major_locator(plt.NullLocator())
            fig_tmp.savefig(
                join(self.save_dir, "detailed_image_{}.jpg".format(ind)),
                bbox_inches="tight",
                pad_inches=0.0,
            )
            self.log.debug(
                "Saved image to {}".format(
                    join(self.save_dir, "detailed_image_{}.jpg".format(ind))
                )
            )
        else:
            fig_tmp.tight_layout(pad=0.0)
            fig_tmp.show()

    def store_thumbnail(self, ind):
        """Stores a thumbnail of a segment if requested. Thus is not saving the whole
        image but only the cropped part."""
        image = Image.open(self.data["image_path"][self.gi[ind]]).convert("RGB")
        image = image.crop(self.data["box"][ind])

        if self.label_mapping[self.data["dataset"][self.gi[ind]]] is None:
            name = "None"
        else:
            name = self.label_mapping[self.data["dataset"][self.gi[ind]]][self.gt[ind]][
                0
            ]
        if name[-1].isdigit():
            name = name[:-2]

        name = name.replace(" ", "_")

        image.save(
            join(
                self.save_dir,
                "thumbnail_{}_{:0>2.1f}_{:0>2.1f}.jpg".format(
                    name, self.x[ind], self.y[ind]
                ),
            )
        )
        self.log.debug(
            "Saved thumbnail to {}".format(
                join(
                    self.save_dir,
                    "thumbnail_{}_{:0>2.1f}_{:0>2.1f}.jpg".format(
                        name, self.x[ind], self.y[ind]
                    ),
                )
            )
        )

    def get_nearest_neighbors(self, ind, metric="cos"):
        """Computes nearest neighbors to the specified index in the collection of
        segment crops."""
        if metric == "euclid":
            dists = self.lp_dist(self.embeddings[ind], self.embeddings, d=2)
        else:
            dists = self.cos_dist(self.embeddings[ind], self.embeddings)
        return np.argsort(dists)[1 : (self.n_neighbors + 1)]

    def update_nearest_neighbors(self, ind):
        """If requesting nearest neighbors of a segment within the nearest neighbor
        plot window the nearest neighbors are updated according to the newly
        selected segment.
        """
        self.log.info("Loading nearest neighbors...")
        self.nearest_neighbors = self.get_nearest_neighbors(
            ind, metric=self.distance_metrics[self.dm]
        )
        thumbnails = []
        for neighbor_ind in self.nearest_neighbors:
            b = self.data["box"][neighbor_ind]
            thumbnails.append(
                plt.imread(self.data["image_path"][self.gi[neighbor_ind]])[
                    b[1] : b[3], b[0] : b[2], :
                ]
            )
        n = math.ceil(math.sqrt(len(self.nearest_neighbors)))
        if len(self.fig_nn.axes) != (n**2):
            # number of nearest neighbors has been changed
            # redefine number of subplots in fig_nn
            self.rearrange_axes(n, n)

        for p in range(n**2):
            if p < self.n_neighbors:
                self.fig_nn.axes[p].imshow(thumbnails[p])
            else:
                self.fig_nn.axes[p].clear()
                self.fig_nn.axes[p].set_axis_off()

        self.fig_nn.canvas.draw()
        self.set_color(ind, self.current_color)
        self.flush_colors()

    def cluster(self, method="kmeans"):
        if method in self.methods_with_ncluster_param:
            n_clusters = self.n_cluster_prompt()
            if n_clusters == "elbow" and method == "kmeans":
                n_clusters = self.elbow()

            self.clustering = self.cluster_methods[method]["main"](
                n_clusters=n_clusters, **self.cluster_methods[method]["kwargs"]
            ).fit_predict(self.embeddings)

    def elbow(self):
        low = int(input("Enter the minimum number of clusters: "))
        high = int(input("Enter the maximum number of clusters: "))
        km = [KMeans(n_clusters=i) for i in range(low, high + 1)]

        km = [k.fit(self.embeddings) for k in tqdm(km, total=len(km))]
        score = [k.inertia_ for k in km]

        fig_elbow = plt.figure(max(3, max(plt.get_fignums()) + 1))
        ax = fig_elbow.add_subplot(111)
        ax.plot(range(low, high + 1), score)
        fig_elbow.show()
        return int(input("Enter number of clusters: "))

    def n_cluster_prompt(self):
        inp = input("Enter the number of clusters: ")
        if inp == "elbow":
            return inp
        else:
            try:
                inp = int(inp)
            except ValueError:
                self.log.error(
                    "Input should be an int or 'elbow' but received {}!".format(inp)
                )
                return "error"
            if inp <= 1:
                raise ValueError("Number of clusters should be greater than 1!")
            else:
                return inp

    def rearrange_axes(self, nrows, ncols):
        """Helper function for the nearest neighbor plot window. If number of nearest
        neighbors has been changed and a new query segment has been chosen the
        arrangement of subplots within the window has to be changed.
        """
        n = len(self.fig_nn.axes)
        if n <= (nrows * ncols):
            # we need to add more axes
            for i, ax in enumerate(self.fig_nn.axes):
                ax.change_geometry(nrows, ncols, i + 1)
            for p in range(n, nrows * ncols):
                ax = self.fig_nn.add_subplot(nrows, ncols, p + 1)
                ax.set_axis_off()
        else:
            # we need to remove some axes
            for p in range(n - 1, (nrows * ncols) - 1, -1):
                self.fig_nn.delaxes(self.fig_nn.axes[p])
            for i, ax in enumerate(self.fig_nn.axes):
                ax.change_geometry(nrows, ncols, i + 1)

    def draw_box_on_image(self, ind):
        """Draws the red bounding of a selected segment onto the source image."""
        box_line_width = 5
        img_box = Image.open(self.data["image_path"][self.gi[ind]]).convert("RGB")
        draw = ImageDraw.Draw(img_box)
        left, upper, right, lower = self.data["box"][ind]
        left, upper = max(0, left - box_line_width), max(0, upper - box_line_width)
        right, lower = min(img_box.size[0], right + box_line_width), min(
            img_box.size[1], lower + box_line_width
        )
        draw.rectangle(
            [left, upper, right, lower], outline=(255, 0, 0), width=box_line_width
        )
        del draw
        return img_box

    @staticmethod
    def visualize_segments(comp, metric):
        """Helper function for generation of the four panels in the detailed
        image function."""
        r = np.asarray(metric)
        r = 1 - 0.5 * r
        g = np.asarray(metric)
        b = 0.3 + 0.35 * np.asarray(metric)

        r = np.concatenate((r, np.asarray([0, 1])))
        g = np.concatenate((g, np.asarray([0, 1])))
        b = np.concatenate((b, np.asarray([0, 1])))

        components = np.asarray(comp.copy(), dtype="int16")
        components[components < 0] = len(r) - 1
        components[components == 0] = len(r)

        img = np.zeros(components.shape + (3,))
        x, y = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
        x = x.reshape(-1)
        y = y.reshape(-1)

        img[x, y, 0] = r[components[x, y] - 1]
        img[x, y, 1] = g[components[x, y] - 1]
        img[x, y, 2] = b[components[x, y] - 1]

        img = np.asarray(255 * img).astype("uint8")

        return img

    @staticmethod
    def lp_dist(point, all_points, d=2):
        """Calculates the L_p distance from a point to a collection of points.
        Used for retrieval."""
        return ((all_points - point) ** d).sum(1) ** (1.0 / d)

    @staticmethod
    def cos_dist(point, all_points):
        """Calculates the cosine distance from a point to a collection of points.
        Used for retrieval."""
        return 1 - (
            (point * all_points).sum(1) / (norm(point) * norm(all_points, axis=1))
        )

    @staticmethod
    def get_gridsize(fig):
        """Helper function for the nearest neighbor plot."""
        return fig.axes[0].get_subplotspec().get_gridspec().get_geometry()

    def get_ind_nn(self, event):
        """Helper function for the nearest neighbor plot."""
        _, ncols = self.get_gridsize(self.fig_nn)
        eventrow = event.inaxes.rowNum
        eventcol = event.inaxes.colNum
        return (eventrow * ncols) + eventcol

    def color_gt(self):
        """When called colors the scatter in the main plot according to the ground
        truth colors."""
        if all(
            self.label_mapping[self.data["dataset"][self.gi[ind]]] is not None
            for ind in range(self.basecolors.shape[0])
        ):
            self.basecolors = np.stack(
                [
                    tuple(
                        i / 255.0
                        for i in self.label_mapping[self.data["dataset"][self.gi[ind]]][
                            self.gt[ind]
                        ][1]
                    )
                    + (1.0,)
                    for ind in range(self.basecolors.shape[0])
                ]
            )
            legend_elements = []
            for d in np.unique(self.data["dataset"]).flatten():
                cls = np.unique(self.gt[np.array(self.data["dataset"])[self.gi] == d])
                cls = list(
                    {
                        (self.label_mapping[d][cl][0], self.label_mapping[d][cl][1])
                        for cl in cls
                    }
                )
                names = np.array([i[0] for i in cls])
                cols = np.array([i[1] for i in cls])
                legend_elements += [
                    Patch(
                        color=tuple(i / 255.0 for i in cols[i]) + (1.0,),
                        label=names[i]
                        if not names[i][-1].isdigit()
                        else names[i][: names[i].rfind(" ")],
                    )
                    for i in range(names.shape[0])
                ]
            self.fig_main.axes[0].legend(
                loc="upper left",
                handles=legend_elements,
                ncol=8,
                bbox_to_anchor=(0, 1.1),
            )
            self.flush_colors()

    def color_pred(self):
        """When called colors the scatter in the main plot according to the predicted
        class color."""
        self.basecolors = np.stack(
            [
                tuple(i / 255.0 for i in self.pred_mapping[self.pred[ind]][1]) + (1.0,)
                for ind in range(self.basecolors.shape[0])
            ]
        )
        legend_elements = [
            Patch(
                color=tuple(i / 255.0 for i in self.pred_mapping[cl][1]) + (1.0,),
                label=self.pred_mapping[cl][0],
            )
            for cl in np.unique(self.pred).flatten()
        ]
        self.fig_main.axes[0].legend(
            loc="upper left", handles=legend_elements, ncol=8, bbox_to_anchor=(0, 1.1)
        )
        self.flush_colors()

    def show_density(self):
        embedding_kde = estimate_kernel_density(self.data["plot_embeddings"])
        xmin = self.x.min()
        xmin = xmin * 1.3 if xmin < 0 else xmin * 0.8
        xmax = self.x.max()
        xmax = xmax * 1.3 if xmax > 0 else xmax * 0.8

        ymin = self.y.min()
        ymin = ymin * 1.3 if ymin < 0 else ymin * 0.8
        ymax = self.y.max()
        ymax = ymax * 1.3 if ymax > 0 else ymax * 0.8

        grid_x, grid_y = np.mgrid[xmin:xmax, ymin:ymax]
        grid_z = embedding_kde(np.vstack([grid_x.flatten(), grid_y.flatten()]))
        colmap = plt.get_cmap("Greys")
        colmap = colors.LinearSegmentedColormap.from_list(
            "trunc({n},{a:.2f},{b:.2f})".format(n=colmap.name, a=0.0, b=0.75),
            colmap(np.linspace(0.0, 0.75, 256)),
        )
        grid_z[grid_z < np.quantile(grid_z, 0.55)] = np.NaN
        colmap.set_bad("white")
        self.fig_main.axes[0].pcolormesh(
            grid_x,
            grid_y,
            grid_z.reshape(grid_x.shape),
            cmap=colmap,
            shading="gouraud",
            zorder=1,
        )
        self.flush_colors()

    def set_color(self, ind, color):
        """Helper function to set a color of a segment with index ind."""
        self.basecolors[ind, :] = color

    def change_color(self, old_color, new_color):
        """Helper function to change a specific color to a different one."""
        self.basecolors[(self.basecolors == old_color).all(axis=1)] = new_color

    def flush_colors(self):
        """Flushes all color changes onto the main scatter plot."""
        self.line_main.set_color(self.basecolors)
        self.fig_main.canvas.draw()
