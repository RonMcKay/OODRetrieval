#!/usr/bin/env python3
"""
script including
class object with global settings
"""
import os
from os.path import join
from collections import namedtuple

ModelConfig = namedtuple('Model', [
    'name',  # Name of the model
    'module_name',  # Name of the module the model class is defined in
    'class_name',  # Name of the model class within the module
    'kwargs',  # Dictionary of keyword arguments to supply to the model class during loading
    'model_weights',  # Path to the pretrained model weights
])

DatasetConfig = namedtuple('Dataset', [
    'name',  # Name of the dataset
    'module_name',  # Name of the module the dataset class is defined in
    'class_name',  # Name of the dataset class within the module
    'kwargs',  # Dictionary of keyword arguments to supply to the dataset class during loading
])

models = dict(
    deeplabv3plus=ModelConfig('deeplabv3plus',
                              'src.model.deepv3',
                              'DeepWV3Plus',
                              dict(num_classes=19),
                              '/path/to/the/pretrained/deeplabv3plus/weights.pth')
)

# The constructor of your meta model needs to accept an integer (as first argument) to specify the number of input features
# The output of your meta model is expected to be linear and to have a single node
meta_models = dict(
    meta_nn=ModelConfig('meta_nn',
                        'src.MetaSeg.functions.meta_nn',
                        'MetaNN',
                        {},
                        './src/meta_nn.pth')
)

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

    metaseg_io_path = "/your/metaseg/input-output/path"  # directory with inputs and outputs, i.e. saving and loading data

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

    meta_model_types = ["linear", "neural"]

    CLASS_DTYPE = 'probs'  # = "probs" ( one of ['one_hot_classes', 'probs'], just leave it like that
    TRAIN_DATASET = datasets['cityscapes']  # The dataset the semantic segmentation model got trained on
    DATASET = datasets['cityscapes_val']  # used for input/output folder path
    MODEL_NAME = 'deeplabv3plus'  # used for input/output folder path
    META_MODEL_NAME = 'meta_nn'
    META_MODEL_TYPE = meta_model_types[1]

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
    # NUM_IMAGES = 500  # uncomment to only process the first 'NUM_IMAGES' images of the specified dataset
    NUM_AVERAGES = 10  # only used when meta model is 'linear'
    NUM_LASSO_LAMBDAS = 40  # only used when meta model is 'linear'
    CLASSINDEX = None  # if DATASET is set to 'a2d2' and CLASSINDEX is set to an valid integer only images
    # containing the specified class will be processed.

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
