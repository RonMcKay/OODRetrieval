import os

import PIL
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class Label(object):
    def __init__(
        self, name, id, trainid, category, catid, hasinstances, ignoreineval, color
    ):
        self.name = name
        self.id = id
        self.trainid = trainid
        self.category = category
        self.catid = catid
        self.hasinstances = hasinstances
        self.ignoreineval = ignoreineval
        self.color = color

    def __call__(self):
        print(
            "name: %s\nid: %d\ntrainid: %d\ncategory: %s\ncatid:\
         %d\nhasinstances: %d\nignoreineval: %d\ncolor:%s"
            % (
                self.name,
                self.id,
                self.trainid,
                self.category,
                self.catid,
                self.hasinstances,
                self.ignoreineval,
                str(self.color),
            )
        )


num_classes = 19
num_categories = 8
void_ind = 255
a2d2_void = void_ind

# mean and sd rgb values of the cityscapes training dataset (for normalization)
# mean = (0.5389, 0.5758, 0.5377)    # original values
# std = (0.1524, 0.1464, 0.1495)
mean = (0.485, 0.456, 0.406)  # values from github repo where the model originates from
std = (0.229, 0.224, 0.225)


# class weights for training
# inverse class frequency: 1/fc
# class_weights = (3.0637286, # road
#                  18.563517, # sidewalk
#                  4.94911, # building
#                  172.35123, # wall
#                  128.76141, # fence
#                  92.035294, # pole
#                  543.60718, # traffic light
#                  204.90633, # traffic sign
#                  7.0915442, # vegetation
#                  97.561729, # terrain
#                  28.106302, # sky
#                  92.668327, # person
#                  174.40337, # rider
#                  16.149548, # car
#                  224.72762, # truck
#                  485.0007, # train
#                  8.7168007) # unlabeled

# median weighted class frequencies: median(fc)/fc
class_weights = (
    0.033061225,  # road
    0.20032212,  # sidewalk
    0.053406704,  # building
    1.859872,  # wall
    1.3894867,  # fence
    0.99316877,  # pole
    5.8661594,  # traffic light
    2.2111797,  # traffic sign
    0.076526083,  # vegetation
    1.0528055,  # terrain
    0.30329996,  # sky
    1.0,  # person
    1.8820169,  # rider
    0.17427257,  # car
    2.4250746,  # truck
    5.2337265,  # train
    0.094064504,
)  # unlabeled


labels = [
    #   name  id trainId  category catId  hasInstances   ignoreInEval   color
    Label("unlabeled", 0, void_ind, "void", 0, False, True, (0, 0, 0)),
    Label("ego vehicle", 1, void_ind, "void", 0, False, True, (0, 0, 0)),
    Label("rectification border", 2, void_ind, "void", 0, False, True, (0, 0, 0)),
    Label("out of roi", 3, void_ind, "void", 0, False, True, (0, 0, 0)),
    Label("static", 4, void_ind, "void", 0, False, True, (0, 0, 0)),
    Label("dynamic", 5, void_ind, "void", 0, False, True, (111, 74, 0)),
    Label("ground", 6, void_ind, "void", 0, False, True, (81, 0, 81)),
    Label("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),
    Label("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),
    Label("parking", 9, void_ind, "flat", 1, False, True, (250, 170, 160)),
    Label("rail track", 10, void_ind, "flat", 1, False, True, (230, 150, 140)),
    Label("building", 11, 2, "construction", 2, False, False, (70, 70, 70)),
    Label("wall", 12, 3, "construction", 2, False, False, (102, 102, 156)),
    Label("fence", 13, 4, "construction", 2, False, False, (190, 153, 153)),
    Label("guard rail", 14, void_ind, "construction", 2, False, True, (180, 165, 180)),
    Label("bridge", 15, void_ind, "construction", 2, False, True, (150, 100, 100)),
    Label("tunnel", 16, void_ind, "construction", 2, False, True, (150, 120, 90)),
    Label("pole", 17, 5, "object", 3, False, False, (153, 153, 153)),
    Label("polegroup", 18, void_ind, "object", 3, False, True, (153, 153, 153)),
    Label("traffic light", 19, 6, "object", 3, False, False, (250, 170, 30)),
    Label("traffic sign", 20, 7, "object", 3, False, False, (220, 220, 0)),
    Label("vegetation", 21, 8, "nature", 4, False, False, (107, 142, 35)),
    Label("terrain", 22, 9, "nature", 4, False, False, (152, 251, 152)),
    Label("sky", 23, 10, "sky", 5, False, False, (70, 130, 180)),
    Label("person", 24, 11, "human", 6, True, False, (220, 20, 60)),
    Label("rider", 25, 12, "vehicle", 7, True, False, (255, 0, 0)),
    Label("car", 26, 13, "vehicle", 7, True, False, (0, 0, 142)),
    Label("truck", 27, 14, "vehicle", 7, True, False, (0, 0, 70)),
    Label("bus", 28, 15, "vehicle", 7, True, False, (0, 60, 100)),
    Label("caravan", 29, void_ind, "vehicle", 7, True, True, (0, 0, 90)),
    Label("trailer", 30, void_ind, "vehicle", 7, True, True, (0, 0, 110)),
    Label("train", 31, 16, "vehicle", 7, True, False, (0, 80, 100)),
    Label("motorcycle", 32, 17, "vehicle", 7, True, False, (0, 0, 230)),
    Label("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
    Label("license plate", -1, -1, "vehicle", 7, False, True, (0, 0, 142)),
    Label("unlabeled", void_ind, void_ind, "void", 0, False, True, (0, 0, 0)),
]

# This list is used for filtering in 'compute_embeddings.py'. Only segments that have a
# prediction within the list are considered for further processing
pred_class_selection = [
    # 0,  # road
    # 1,  # sidewalk
    # 2,  # building
    3,  # wall
    4,  # fence
    6,  # traffic light
    7,  # traffic sign
    # 8,  # vegetation
    # 9,  # terrain
    # 10,  # sky
    11,  # person
    12,  # rider
    13,  # car
    14,  # truck
    15,  # bus
    16,  # train
    17,  # motorcycle
    18,  # bicycle
]

cityscapes_to_a2d2 = {
    0: a2d2_void,
    1: 14,
    2: 5,
    3: 5,
    4: a2d2_void,
    5: a2d2_void,
    6: a2d2_void,
    7: 27,
    8: 32,
    9: 22,
    10: 19,
    11: 6,
    12: a2d2_void,
    13: 16,
    14: a2d2_void,
    15: 6,
    16: a2d2_void,
    17: 26,
    18: 26,
    19: 46,
    20: 43,
    21: 18,
    22: 18,
    23: 34,
    24: 23,
    25: a2d2_void,
    26: 7,
    27: 49,
    28: 49,
    29: 7,
    30: 49,
    31: 52,
    32: 36,
    33: 1,
    -1: a2d2_void,
}

id_to_trainid = {label.id: label.trainid for label in labels}
id_to_color = {label.id: label.color for label in labels}
id_to_name = {label.id: label.name for label in labels}
trainid_to_id = {
    label.trainid: (label.id if (label.trainid != void_ind) else 0) for label in labels
}
trainid_to_name = {label.trainid: label.name for label in labels}
trainid_to_color = {label.trainid: label.color for label in labels}
name_to_rgba = {
    label.name: tuple(i / 255.0 for i in label.color) + (1.0,) for label in labels
}
id_to_catid = {label.id: label.catid for label in labels}
trainid_to_catid = {label.trainid: label.catid for label in labels}
id_to_categoryname = {label.id: label.category for label in labels}
trainid_to_categoryname = {label.trainid: label.category for label in labels}

discover_mapping = {
    label.id: (trainid_to_name[label.trainid], trainid_to_color[label.trainid])
    for label in labels
}
pred_mapping = {
    label.trainid: (trainid_to_name[label.trainid], trainid_to_color[label.trainid])
    for label in labels
}


def fulltotrain(target):
    """Transforms labels from full cityscapes labelset to training label set."""
    remapped_target = target.clone()
    for k, v in id_to_trainid.items():
        remapped_target[target == k] = v
    return remapped_target


def traintofull(target):
    """Transforms labels from training label set to the full label set."""
    remapped_target = target.clone()
    for k, v in trainid_to_id.items():
        remapped_target[target == k] = v
    return remapped_target


def traintocolor(target):
    return fulltocolor(traintofull(target))


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
        [
            np.expand_dims(colors[t].reshape(h, w, 3).transpose([2, 0, 1]), 0)
            for t in target
        ]
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


class Cityscapes(Dataset):
    def __init__(
        self,
        split="train",
        root="/data/datasets/semseg/Cityscapes",
        map_fun=fulltotrain,
        transform=None,
        label_mapping=discover_mapping,
        pred_mapping=pred_mapping,
    ):
        """Load all filenames."""
        super(Cityscapes, self).__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.label_mapping = label_mapping
        self.pred_mapping = pred_mapping
        self.images = []
        self.targets = []
        self.map_fun = map_fun

        for root, _, filenames in os.walk(
            os.path.join(self.root, "leftImg8bit", self.split)
        ):
            for filename in filenames:
                if os.path.splitext(filename)[1] == ".png":
                    filename_base = "_".join(filename.split("_")[:-1])
                    self.images.append(
                        os.path.join(root, filename_base + "_leftImg8bit.png")
                    )

                    if self.split in ["train", "val"]:
                        target_root = os.path.join(
                            self.root, "gtFine", self.split, os.path.basename(root)
                        )
                        self.targets.append(
                            os.path.join(
                                target_root, filename_base + "_gtFine_labelIds.png"
                            )
                        )

    def __len__(self):
        """Return number of images in the dataset."""
        return len(self.images)

    def __getitem__(self, i):
        # Load images and perform augmentations with PIL
        image = Image.open(self.images[i]).convert("RGB")
        if self.split in ["train", "val"]:
            target = Image.open(self.targets[i]).convert("L")
        else:
            target = Image.fromarray(
                np.full(image.size[:-1], void_ind).astype("uint8"), mode="L"
            )

        if self.transform is not None:
            image, target = self.transform(image, target)

        if isinstance(target, PIL.Image.Image):
            target = torch.tensor(np.array(target, dtype=np.uint8), dtype=torch.long)

        if self.map_fun is not None:
            target = self.map_fun(target)

        return image, target, self.images[i]


# EXPECTED FOLDER STRUCTURE:
# 	root
# 	| devkit
# 		| cityscapesscripts
# 		| .git
# 		| docs
# 	| leftImg8bit
# 		| train_extra
# 		| test
# 		| train
# 		| val
# 	| gtFine
# 		| test
# 		| train
# 		| val
# 	| gtCoarse
# 		| train_extra
# 		| train
# 		| val
