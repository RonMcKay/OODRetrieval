from collections import namedtuple
import os

import PIL
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

Label = namedtuple(
    "Label",
    [
        "name",  # The identifier of this label, e.g. 'car', 'person', ... .
        # We use them to uniquely name a class
        "id",  # An integer ID that is associated with this label.
        # The IDs are used to represent the label in ground truth images
        # An ID of -1 means that this label does not have an ID and thus
        # is ignored when creating ground truth images (e.g. license plate).
        "trainId",  # An integer ID that overwrites the ID above, when creating ground
        # truth images for training.
        # For training, multiple labels might have the same ID. Then, these labels
        # are mapped to the same class in the ground truth images. For the inverse
        # mapping, we use the label that is defined first in the list below.
        # For example, mapping all void-type classes to the same ID in training,
        # might make sense for some approaches.
        "hasInstances",  # Distinguishes between single instances or not
        "ignoreInEval",  # Pixels having this class as ground truth label are ignored
        # during evaluations or not
        "color",  # The color of this label
    ],
)


num_classes = 44
void_ind = 255

# mean and sd rgb values of the cityscapes training dataset (for normalization)

labels = [
    #   name     id      trainId    hasInstances   ignoreInEval   color
    Label("unlabeled", 0, 0, False, True, (0, 0, 0)),
    Label("ego vehicle", 0, 0, False, True, (0, 0, 0)),
    Label("rectification border", 0, 0, False, True, (0, 0, 0)),
    Label("out of roi", 0, 0, False, True, (0, 0, 0)),
    Label("background", 0, 0, False, False, (0, 0, 0)),
    Label("free", 1, 1, False, False, (128, 64, 128)),
    Label("01", 2, 2, True, False, (0, 0, 142)),
    Label("02", 3, 2, True, False, (0, 0, 142)),
    Label("03", 4, 2, True, False, (0, 0, 142)),
    Label("04", 5, 2, True, False, (0, 0, 142)),
    Label("05", 6, 2, True, False, (0, 0, 142)),
    Label("06", 7, 2, True, False, (0, 0, 142)),
    Label("07", 8, 2, True, False, (0, 0, 142)),
    Label("08", 9, 2, True, False, (0, 0, 142)),
    Label("09", 10, 2, True, False, (0, 0, 142)),
    Label("10", 11, 2, True, False, (0, 0, 142)),
    Label("11", 12, 2, True, False, (0, 0, 142)),
    Label("12", 13, 2, True, False, (0, 0, 142)),
    Label("13", 14, 2, True, False, (0, 0, 142)),
    Label("14", 15, 2, True, False, (0, 0, 142)),
    Label("15", 16, 2, True, False, (0, 0, 142)),
    Label("16", 17, 2, True, False, (0, 0, 142)),
    Label("17", 18, 2, True, False, (0, 0, 142)),
    Label("18", 19, 2, True, False, (0, 0, 142)),
    Label("19", 20, 2, True, False, (0, 0, 142)),
    Label("20", 21, 2, True, False, (0, 0, 142)),
    Label("21", 22, 2, True, False, (0, 0, 142)),
    Label("22", 23, 2, True, False, (0, 0, 142)),
    Label("23", 24, 2, True, False, (0, 0, 142)),
    Label("24", 25, 2, True, False, (0, 0, 142)),
    Label("25", 26, 2, True, False, (0, 0, 142)),
    Label("26", 27, 2, True, False, (0, 0, 142)),
    Label("27", 28, 2, True, False, (0, 0, 142)),
    Label("28", 29, 2, True, False, (0, 0, 142)),
    Label("29", 30, 2, True, False, (0, 0, 142)),
    Label("30", 31, 0, True, False, (0, 0, 0)),
    Label("31", 32, 2, True, False, (0, 0, 142)),
    Label("32", 33, 0, True, False, (0, 0, 0)),
    Label("33", 34, 0, True, False, (0, 0, 0)),
    Label("34", 35, 2, True, False, (0, 0, 142)),
    Label("35", 36, 0, True, False, (0, 0, 0)),
    Label("36", 37, 0, True, False, (0, 0, 0)),
    Label("37", 38, 0, True, False, (0, 0, 0)),
    Label("38", 39, 0, True, False, (0, 0, 0)),
    Label("39", 40, 2, True, False, (0, 0, 142)),
    Label("40", 41, 2, True, False, (0, 0, 142)),
    Label("41", 42, 2, True, False, (0, 0, 142)),
    Label("42", 43, 2, True, False, (0, 0, 142)),
    Label("unlabeled", void_ind, void_ind, False, True, (0, 0, 0)),
]


id_to_trainid = {label.id: label.trainId for label in labels}
id_to_color = {label.id: label.color for label in labels}
id_to_name = {label.id: label.name for label in labels}
trainid_to_color = {label.trainId: label.color for label in labels}
trainid_to_name = {label.trainId: label.name for label in labels}

discover_mapping = {
    label.id: (trainid_to_name[label.trainid], trainid_to_color[label.trainid])
    for label in labels
}


def fulltotrain(target):
    """Transforms labels from full cityscapes labelset to training label set."""
    return np.vectorize(id_to_trainid.get)(target)


def trainidtocolor(target):
    if not isinstance(target, np.ndarray):
        target = np.array(target)
    target_col = np.empty((target.shape[0], 3, target.shape[1], target.shape[2]))
    cols = np.array([trainid_to_color[c] for c in target.flatten()]).reshape(
        (target.shape[0], -1)
    )

    target_col[target[0], target[1], target[2]] = cols
    return target_col


def fulltocolor(target):
    """Maps labels to their RGB colors in cityscapes."""
    colors = [(label.id, label.color) for label in labels if label.id != -1]
    colors.sort(key=lambda x: x[0])
    colors = np.array([x[1] for x in colors], dtype=np.uint8)

    target = target.numpy()
    if len(target.shape) == 2:
        b = 1
        h = target.shape[0]
        w = target.shape[1]
    elif len(target.shape) == 3:
        b, h, w = target.shape
    else:
        b, _, h, w = target.shape
    target = target.reshape(b, -1)

    rgb_target = np.concatenate(
        [np.expand_dims(colors[t].reshape(3, h, w), 0) for t in target]
    )
    return torch.from_numpy(rgb_target)


def onehot(target, n_classes=num_classes):
    """Transforms labels to one hot encoded."""
    if len(target.size()) == 2:
        # When no batch dimension is given
        h, w = target.size()
        onehot_target = torch.zeros(n_classes, h, w)

        for c in range(n_classes):
            onehot_target[c][target == c] = 1
    else:
        # when batch dimension is given
        b, h, w = target.size()
        onehot_target = torch.zeros(b, n_classes, h, w)

        for c in range(n_classes):
            onehot_target[:, c, :, :][target == c] = 1
    return onehot_target


class CityscapesLAF(Dataset):
    def __init__(
        self,
        split="train",
        root="/data/datasets/semseg/Cityscapes_lost_and_found",
        map_fun=None,
        transform=None,
        label_mapping=discover_mapping,
    ):
        """Load all filenames."""
        super(CityscapesLAF, self).__init__()
        self.root = root
        self.split = split
        if self.split not in ["train", "test"]:
            raise ValueError(
                "'split' should be one of [train, test] but received {}!".format(
                    self.split
                )
            )
        self.transform = transform
        self.label_mapping = label_mapping
        self.images = []
        self.targets = []
        self.map_fun = map_fun

        for root, _, filenames in os.walk(
            os.path.join(self.root, "leftImg8bit", split)
        ):
            for filename in filenames:
                if os.path.splitext(filename)[1] == ".png":
                    filename_base = "_".join(filename.split("_")[:-1])
                    self.images.append(
                        os.path.join(root, filename_base + "_leftImg8bit.png")
                    )

                    if self.split == "train":
                        target_root = os.path.join(
                            self.root, "gtCoarse", split, os.path.basename(root)
                        )
                        self.targets.append(
                            os.path.join(
                                target_root, filename_base + "_gtCoarse_labelIds.png"
                            )
                        )

    def __len__(self):
        """Return number of images in the dataset."""
        return len(self.images)

    def __getitem__(self, i):
        # Load images and perform augmentations with PIL
        image = Image.open(self.images[i]).convert("RGB")
        if self.split == "train":
            target = Image.open(self.targets[i]).convert("L")
        else:
            target = Image.fromarray(np.zeros(image.size[::-1]), mode="L")

        if self.transform is not None:
            image, target = self.transform(image, target)

        if isinstance(target, PIL.Image.Image):
            target = torch.tensor(np.array(target, dtype=np.uint8), dtype=torch.long)

        if self.map_fun is not None:
            target = self.map_fun(target)

        return image, target, self.images[i]


# EXPECTED FOLDER STRUCTURE:
#    root
#    | leftImg8bit
#        | train_extra
#        | test
#        | train
#        | val
#    | gtCoarse
#        | train_extra
#        | train
#        | val
