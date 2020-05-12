#!/usr/bin/env python3
"""
script including
class object with global settings
"""
import os
from os.path import join
from collections import namedtuple

DatasetConfig = namedtuple('Dataset', [
    'name',  # Name of the dataset
    'module_name',  # Name of the module the dataset class is defined in
    'class_name',  # Name of the dataset class within the module
    'kwargs',  # Dictionary of keyword arguments to supply to the dataset class during loading
])

datasets = dict(
    cityscapes=DatasetConfig('cityscapes',
                             'src.datasets.cityscapes',
                             'Cityscapes',
                             dict(root='/data/datasets/semseg/Cityscapes',
                                  split='train')),
    cityscapes_val=DatasetConfig('cityscapes_val',
                                 'src.datasets.cityscapes',
                                 'Cityscapes',
                                 dict(root='/data/datasets/semseg/Cityscapes',
                                      split='val')),
    cityscapes_test=DatasetConfig('cityscapes_test',
                                  'src.datasets.cityscapes',
                                  'Cityscapes',
                                  dict(root='/data/datasets/semseg/Cityscapes',
                                       split='test')),
    cityscapes_laf=DatasetConfig('cityscapes_laf',
                                 'src.datasets.cityscapes_laf',
                                 'CityscapesLAF',
                                 dict(root='/data/datasets/semseg/Cityscapes_lost_and_found',
                                      split='train')),
    a2d2=DatasetConfig('a2d2',
                       'src.datasets.a2d2',
                       'A2D2',
                       dict(root='/data/datasets/semseg/A2D2',
                            cam_positions=['front_center'])),
    custom_dataset=DatasetConfig('custom_dataset',
                                 'src.datasets.custom',
                                 'CustomDataset',
                                 dict(root='/path/to/your/image/directory',
                                      image_file_extension='.png'))
)


class CONFIG:
    # --------------------- #
    # set necessary paths   #
    # --------------------- #

    metaseg_io_path = "/data/poberdie/metaseg"  # directory with inputs and outputs, i.e. saving and loading data

    # ---------------------------- #
    # paths for data preparation   #
    # ---------------------------- #
    # The following path definitions are deprecated but for compatibility reasons still here
    IMG_DIR = "/data/ai_data_and_models/data/DS_20k/test/Input/"
    GT_DIR = "/data/ai_data_and_models/data/DS_20k/test/PNG_13cl/"
    PROBS_DIR = "/data/ai_data_and_models/inference_results/FRRNA_Softmax_Output/nparrays/softmax/predictions/"

    # ------------------ #
    # select or define   #
    # ------------------ #

    model_names = ["deeplabv3plus"]
    pretrained_model = "/data/poberdie/nvidia_pretrained_models/cityscapes_best.pth"  # path to pretrained model weights file
    meta_nn_weights = "./src/meta_nn.pth"
    meta_models = ["linear", "neural"]

    CLASS_DTYPE = 'probs'  # = "probs" ( one of ['one_hot_classes', 'probs']
    TRAIN_DATASET = datasets['cityscapes']  # The dataset the semantic segmentation model got trained on
    DATASET = datasets['cityscapes_val']  # used for input/output folder path
    MODEL_NAME = model_names[0]  # used for input/output folder path
    META_MODEL = meta_models[1]

    # --------------------------------------------------------------------#
    # select tasks to be executed by setting boolean variable True/False #
    # --------------------------------------------------------------------#

    COMPUTE_METRICS = True
    VISUALIZE_RATING = False
    ANALYZE_METRICS = False

    # ----------- #
    # optionals   #
    # ----------- #

    GPU_ID = 0
    NUM_CORES = 8
    NUM_IMAGES = 500
    NUM_AVERAGES = 10
    NUM_LASSO_LAMBDAS = 40
    CLASSINDEX = None

    INPUT_DIR = join(metaseg_io_path, "input", MODEL_NAME, DATASET.name) + "/"
    METRICS_DIR = join(metaseg_io_path, "metrics", MODEL_NAME, DATASET.name) + "/"
    COMPONENTS_DIR = join(metaseg_io_path, "components", MODEL_NAME, DATASET.name) + "/"
    IOU_SEG_VIS_DIR = join(metaseg_io_path, "iou_seg_vis", MODEL_NAME, DATASET.name) + "/"
    RESULTS_DIR = join(metaseg_io_path, "results", MODEL_NAME, DATASET.name) + "/"
    STATS_DIR = join(metaseg_io_path, "stats", MODEL_NAME, DATASET.name) + "/"
    LOG_FILE_PATH = join(metaseg_io_path, 'log.txt')

    for p in [INPUT_DIR, METRICS_DIR, COMPONENTS_DIR]:
        if not os.path.exists(p):
            os.makedirs(p, exist_ok=True)
