import importlib
import logging
from multiprocessing import Pool
from os.path import join
import pickle as pkl
import sys

import PIL.Image as Image
import numpy as np
from sacred import Experiment
import torch
import torchvision.transforms as trans
import tqdm

from configuration import CONFIG
from src.MetaSeg.functions.calculate import meta_nn_predict, regression_fit_and_predict
from src.MetaSeg.functions.helper import load_data
from src.MetaSeg.functions.in_out import components_load, get_indices, probs_gt_load
from src.embedding_networks import (
    feature_densenet201,
    feature_resnet18,
    feature_resnet101,
    feature_resnet152,
    feature_vgg16,
    feature_wide_resnet101,
)
from src.log_utils import log_config

ex = Experiment("compute_embeddings")

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

# this mean and stardard deviation have been used for all PyTorch models during
# training. Use them during prediction of feature embeddings
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)


def wrapper_cutout_components(args):
    """Wrapper for the multiprocessing pool."""
    return cutout_components(*args)


# noinspection PyArgumentList
def cutout_components(
    component_indices,
    image_index,
    iou_pred,
    dataset="a2d2",
    min_height=64,
    min_width=64,
    min_crop_height=128,
    min_crop_width=128,
    model_name="deeplabv3plus",
):
    """Cuts out all components of the image if they match the minimum size requirements.

    Args:
        component_indices (sequence): Sequence of local component numbers.
        image_index (int): Index of the image to process.
        iou_pred (numpy array): Array of iou predictions for each component
        dataset (str): Name of the dataset to process.
        min_height (int): Minimum height of the component to be processed. Useful if
            you want to pass the crop to a neural network.
        min_width (int): Minimum height of the component to be processed. Useful if you
            want to pass the crop to a neural network .
        min_crop_width (int): Minimum width the resulting bounding box should have.
            If the segment satisfies the min_width but is smaller then min_crop_width
            the bounding box is getting enlarged until the min_crop_width is satisfied.
        min_crop_height: Minimum height the resulting bounding box should have. If the
            segment satisfies the min_height but is smaller then min_crop_height the
            bounding box is getting enlarged until the min_crop_height is satisfied.
        model_name (str): Name of the model used.

    Returns: Dictionary with
        'dataset': Name of the dataset the image belongs to.
        'model_name': Name of the model used for the prediction
        'data': List of raw image crops containing the matching components
        'addresses':
            List of addresses where to find the crop (path to the image file
            and corner coordinates of the box (top, left, bottom, right)
    """
    components = components_load(
        image_index,
        components_dir=join(CONFIG.metaseg_io_path, "components", model_name, dataset),
    )
    crops = {
        "dataset": dataset,
        "model_name": model_name,
        "embeddings": [],
        "boxes": [],
        "image_index": image_index,
        "iou_pred": iou_pred,
        "component_indices": [],
        "segment_indices": [],
        "img_crops": [],
    }
    for cindex in component_indices:
        segment_indices = np.argwhere(components == cindex)
        if segment_indices.shape[0] > 0:
            upper, left = segment_indices.min(0)
            lower, right = segment_indices.max(0)
            if (lower - upper) < min_height or (right - left) < min_width:
                continue

            if (right - left) < min_crop_width:
                margin = min_crop_width - (right - left)
                if left - (margin // 2) < 0:
                    left = 0
                    right = left + min_crop_width
                elif right + (margin // 2) > components.shape[1]:
                    right = components.shape[1]
                    left = right - min_crop_width

                if right > components.shape[1] or left < 0:
                    raise IndexError(
                        "Image with shape {} is too small for a {} x {} crop".format(
                            components.shape, min_crop_height, min_crop_width
                        )
                    )
            if (lower - upper) < min_crop_height:
                margin = min_crop_height - (lower - upper)
                if upper - (margin // 2) < 0:
                    upper = 0
                    lower = upper + min_crop_height
                elif lower + (margin // 2) > components.shape[0]:
                    lower = components.shape[0]
                    upper = lower - min_crop_height

                if lower > components.shape[0] or upper < 0:
                    raise IndexError(
                        "Image with shape {} is too small for a {} x {} crop".format(
                            components.shape, min_crop_height, min_crop_width
                        )
                    )

            crops["boxes"].append((left, upper, right, lower))
            crops["component_indices"].append(cindex)
            crops["segment_indices"].append(segment_indices)
    return crops


def get_image_index_to_components(component_indices, start):
    """Maps global component indices and start values to their local component indices
    and image index.

    Args:
        component_indices (sequence): Sequence of component indices.
        start (sequence): Sequence of indices where components of each image start
    """
    out = {}

    for i in range(len(start) - 1):
        index = component_indices[
            np.logical_and(
                start[i] <= component_indices, component_indices < start[i + 1]
            )
        ]
        out[i] = [j - start[i] + 1 for j in index]
    return out


@ex.capture
def get_embedding(image, net, args):
    """Computes the output of the supplied neural network with respect to the supplied
    image.

    Args:
        image (tensor): Image tensor to be processed by the neural network.
        net (nn.Module): Neural Network to use.
        args: Arguments provided by sacred.

    Returns: Output tensor of the neural network moved to the cpu

    """
    image = image.cuda(args["gpu"])
    with torch.no_grad():
        out = net(image)
    return out.data.cpu().squeeze().numpy()


def get_component_gt(gt, segment_indices):
    """Computes the ground truth for the supplied gt labels and segment indices."""
    cls, cls_counts = np.unique(
        gt[segment_indices[:, 0], segment_indices[:, 1]], return_counts=True
    )
    # cls, cls_counts = np.unique(gt[box[1]:box[3], box[0]:box[2]], return_counts=True)
    return cls[np.argsort(cls_counts)[-1]]


def get_component_pred(pred, segment_indices):
    """Computes the prediction of a segment based on the supplied predictions and
    segment indices."""
    return pred[segment_indices[0, 0], segment_indices[0, 1]]


@ex.config
def config():
    args = dict(
        net="densenet201",  # Network architecture used for computing visual features
        datasets=(CONFIG.TRAIN_DATASET.name, CONFIG.DATASET.name),
        # First specified dataset will always be used as source domain
        load_file=None,  # File in which segments got already extracted. If specified
        # the file get's loaded and
        # the embeddings in there are overwritten.
        gpu=CONFIG.GPU_ID,  # GPU id to use for computation of features for the
        # embedding space
        n_jobs=CONFIG.NUM_CORES,  # Number of processes to use for the extraction of
        # all bounding boxes
        min_height=128,  # Minimum height of a predicted segment
        min_width=128,  # Minimum width of a predicted segment
        min_crop_height=128,  # Minimum height of the resulting bounding box, can be
        # larger than min_height
        min_crop_width=128,  # Minimum width of the resulting bounding box, can be
        # larger than min_width
        meta_nn_path="./src/meta_nn.pth",  # Path to the meta segmentation model
        iou_threshold=0.5,  # Threshold to use for extracting segments based on
        # predicted IoU
        meta_model=CONFIG.META_MODEL_TYPE,  # Model type to use for meta segmentation
    )

    if args["meta_model"] == "neural":
        args["meta_nn_path"] = "./src/meta_nn.pth"

    args["save_file"] = join(
        CONFIG.metaseg_io_path,
        "embeddings_{}_{}_{}.p".format(
            args["min_height"], args["min_width"], args["net"]
        ),
    )


@ex.automain
def main(args, _run, _log):
    log_config(_run, _log)
    # load a network architecture
    _log.info("Loading {}...".format(args["net"]))
    if args["net"] == "vgg16":
        net = feature_vgg16()
    elif args["net"] == "resnet18":
        net = feature_resnet18()
    elif args["net"] == "resnet101":
        net = feature_resnet101()
    elif args["net"] == "resnet152":
        net = feature_resnet152()
    elif args["net"] == "wide_resnet101":
        net = feature_wide_resnet101()
    elif args["net"] == "densenet201":
        net = feature_densenet201()
    else:
        raise ValueError
    net = net.cuda(args["gpu"])
    net.eval()

    # if no precomputed segments have been supplied, they have to be computed
    if args["load_file"] is None:
        _log.info("Loading Metrics...")
        xa_all = []
        start_others = []
        pred_test = []
        dataset_assignments = []
        image_indices = []

        # the first dataset of the 'datasets' configuration serves as source domain
        # dataset. Metric statistics of this dataset are used to normalize the target
        # domain metric statistics. This is why it has to get loaded too.
        if args["meta_model"] == "neural" and all(
            i in torch.load(args["meta_nn_path"]).keys()
            for i in [
                "train_xa_mean",
                "train_xa_std",
                "train_classes_mean",
                "train_classes_std",
            ]
        ):
            _log.info(
                "Loading values for normalization from saved model file '{}'".format(
                    args["meta_nn_path"]
                )
            )
            model_dict = torch.load(args["meta_nn_path"])
            xa_mean = model_dict["train_xa_mean"]
            xa_std = model_dict["train_xa_std"]
            classes_mean = model_dict["train_classes_mean"]
            classes_std = model_dict["train_classes_std"]
        else:
            _log.info("{}...".format(args["datasets"][0]))
            (
                xa,
                ya,
                x_names,
                class_names,
                xa_mean,
                xa_std,
                classes_mean,
                classes_std,
                *_,
                start,
                pred,
            ) = load_data(args["datasets"][0])

        # Now load all other metric statistics and normalize them using the source
        # domain mean and standard deviation
        for i, d in enumerate(args["datasets"][1:], start=1):
            _log.info("{} ...".format(d))
            num_imgs = get_indices(
                join(CONFIG.metaseg_io_path, "metrics", "deeplabv3plus", d)
            )
            xa_tmp, *_, start_tmp, pred_tmp = load_data(
                d,
                num_imgs=num_imgs,
                xa_mean=xa_mean,
                xa_std=xa_std,
                classes_mean=classes_mean,
                classes_std=classes_std,
            )
            xa_all.append(xa_tmp)
            pred_test.append(pred_tmp)
            dataset_assignments += [i] * len(num_imgs)
            image_indices += num_imgs
            start_others.append(start_tmp)

        # combine them into single arrays
        xa_all = np.concatenate(xa_all).squeeze()
        pred_test = np.concatenate(pred_test).squeeze()
        dataset_assignments = np.array(dataset_assignments).squeeze()
        image_indices = np.array(image_indices).squeeze()

        for starts in start_others[1:]:
            start_others[0] += [s + start_others[0][-1] for s in starts[1:]]
        start_all = start_others[0]
        del xa_tmp, start_tmp, pred_tmp, start_others

        _log.debug("Shape of metrics array: {}".format(xa_all.shape))

        # Using the normalized metric statistics use a meta segmentation network
        # pretrained on the source domain to predict IoU
        _log.info("Predicting IoU...")
        if args["meta_model"] == "neural":
            ya_pred_test = meta_nn_predict(
                pretrained_model_path=args["meta_nn_path"],
                x_test=xa_all,
                gpu=args["gpu"],
            )
        elif args["meta_model"] == "linear":
            ya_pred_test, _ = regression_fit_and_predict(
                x_train=xa, y_train=ya, x_test=xa_all
            )
        else:
            raise ValueError("Meta model {} not supported.".format(args["meta_model"]))

        # Now the different filters are getting applied to the segments
        _log.info("Filtering segments...")
        inds = np.zeros(pred_test.shape[0]).astype(np.bool)

        # Filter for the predicted IoU to be less than the supplied threshold
        inds = np.logical_or(inds, (ya_pred_test < args["iou_threshold"]))

        # Filter for extracting segments with predefined class predictions
        if hasattr(
            importlib.import_module(CONFIG.TRAIN_DATASET.module_name),
            "pred_class_selection",
        ):
            pred_class_selection = getattr(
                importlib.import_module(CONFIG.TRAIN_DATASET.module_name),
                "pred_class_selection",
            )
            inds = np.logical_and(inds, np.isin(pred_test, pred_class_selection))

        _log.info("Filtered components (not checked for minimum size):")
        train_dat = getattr(
            importlib.import_module(CONFIG.TRAIN_DATASET.module_name),
            CONFIG.TRAIN_DATASET.class_name,
        )(**CONFIG.TRAIN_DATASET.kwargs)
        _log.info(
            "\t{:^{width}s} | Filtered | Total".format(
                "Class name",
                width=max(
                    [len(v[0]) for v in train_dat.pred_mapping.values()]
                    + [len("Class name")]
                ),
            )
        )
        for cl in np.unique(pred_test).flatten():
            _log.info(
                "\t{:^{width}s} | {:>8d} | {:<8d}".format(
                    train_dat.pred_mapping[cl][0],
                    inds[pred_test == cl].sum(),
                    (pred_test == cl).sum(),
                    width=max(
                        [len(v[0]) for v in train_dat.pred_mapping.values()]
                        + [len("Class name")]
                    ),
                )
            )

        # Aggregating arguments for extraction of component information.
        inds = np.argwhere(inds).flatten()
        component_image_mapping = get_image_index_to_components(inds, start_all)
        p_args = [
            (
                v,
                image_indices[k],
                ya_pred_test[start_all[k] : start_all[k + 1]],
                args["datasets"][dataset_assignments[k]],
                args["min_height"],
                args["min_width"],
                args["min_crop_height"],
                args["min_crop_width"],
                "deeplabv3plus",
            )
            for k, v in component_image_mapping.items()
        ]

        # Extracting component information can be parallelized in a multiprocessing pool
        _log.info("Extracting component information...")
        with Pool(args["n_jobs"]) as p:
            r = list(
                tqdm.tqdm(p.imap(wrapper_cutout_components, p_args), total=len(p_args))
            )
        r = [c for c in r if len(c["component_indices"]) > 0]

        _log.info("Computing embeddings...")
        crops = {
            "embeddings": [],
            "image_path": [],
            "image_index": [],
            "component_index": [],
            "box": [],
            "gt": [],
            "pred": [],
            "dataset": [],
            "model_name": [],
            "image_level_index": [],
            "iou_pred": [],
        }
        # process all extracted crops and compute feature embeddings
        for c in tqdm.tqdm(r):
            # load image
            preds, gt, image_path = probs_gt_load(
                c["image_index"],
                input_dir=join(
                    CONFIG.metaseg_io_path, "input", c["model_name"], c["dataset"]
                ),
                preds=True,
            )

            crops["image_path"].append(image_path)
            crops["model_name"].append(c["model_name"])
            crops["dataset"].append(c["dataset"])
            crops["image_index"].append(c["image_index"])
            crops["iou_pred"].append(c["iou_pred"])

            image = Image.open(image_path).convert("RGB")
            for i, b in enumerate(c["boxes"]):
                img = trans.ToTensor()(image.crop(b))
                img = trans.Normalize(mean=imagenet_mean, std=imagenet_std)(img)
                crops["embeddings"].append(get_embedding(img.unsqueeze(0), net))
                crops["box"].append(b)
                crops["component_index"].append(c["component_indices"][i])
                crops["image_level_index"].append(len(crops["image_path"]) - 1)
                crops["gt"].append(get_component_gt(gt, c["segment_indices"][i]))
                crops["pred"].append(get_component_pred(preds, c["segment_indices"][i]))

        _log.info("Saving data...")
        with open(args["save_file"], "wb") as f:
            pkl.dump(crops, f)
    else:
        with open(args["load_file"], "rb") as f:
            crops = pkl.load(f)

        _log.info("Computing embeddings...")
        boxes = np.array(crops["box"]).squeeze()
        image_level_index = np.array(crops["image_level_index"]).squeeze()
        crops["embeddings"] = []
        for i, image_path in tqdm.tqdm(
            enumerate(crops["image_path"]), total=len(crops["image_path"])
        ):
            image = Image.open(image_path).convert("RGB")
            for j in np.argwhere(image_level_index == i).flatten():
                img = trans.ToTensor()(image.crop(boxes[j]))
                img = trans.Normalize(mean=imagenet_mean, std=imagenet_std)(img)
                crops["embeddings"].append(get_embedding(img.unsqueeze(0), net))

        if "plot_embeddings" in crops:
            del crops["plot_embeddings"]
        if "nn_embeddings" in crops:
            del crops["nn_embeddings"]

        _log.info("Saving data...")
        with open(args["save_file"], "wb") as f:
            pkl.dump(crops, f)
