#!/usr/bin/env python3
"""
script for metaseg input preparation
author: Chan, Robin (University of Wuppertal), email: robin.chan@uni-wuppertal.de
"""

import os
import h5py
import numpy as np
from PIL import Image
from glob import glob
from configuration import CONFIG
from . import labels

"""
This script was tested for the cityscapes and ds20k dataset. If you are on devcube ip:192.168.1.29 (and no data has been deleted), set the following paths in CONFIG class in "configuration.py" in order to reproduce:
  
  Cityscapes:
  IMG_DIR           = "/data/ai_data_and_models/data/cityscapes/leftImg8bit/"
  GT_DIR            = "/data/ai_data_and_models/data/cityscapes/gtFine/"
  PROBS_DIR         = "/data/ai_data_and_models/inference_results/mn.sscl.os16/"

  DS20k:
  IMG_DIR           = "/data/ai_data_and_models/data/DS_20k/test/Input/"
  GT_DIR            = "/data/ai_data_and_models/data/DS_20k/test/PNG_13cl/"
  PROBS_DIR         = "/data/ai_data_and_models/inference_results/FRRNA_Softmax_Output/nparrays/softmax/predictions/"

Function calls:

  prepare_data().cityscapes() or prepare_data().ds20k() , respectively

The produced MetaSeg inputs are stored as hdf5 files in CONFIG.INPUT_DIR.
Note, that this script assumes the softmax outputs to be available as numpy files (*.npy) per image. Moreover, pay attention to the filenames as this script assumes an unique (sub-)string for every image and its corresponding ground_truth as well as probabilties array.
"""

if not os.path.exists(CONFIG.INPUT_DIR):
    os.makedirs(CONFIG.INPUT_DIR)


class prepare_data(object):

    def __init__(self, num_cores=CONFIG.NUM_CORES):
        self.num_cores = num_cores
        self.trainId2label_ds20k = {label.trainId: label for label in reversed(labels.ds20k_labels)}
        self.label2trainId_cs = {label.Id: label for label in reversed(labels.cs_labels)}

    @staticmethod
    def probs_gt_save(probs, gt, img_path, i):
        file_name = CONFIG.INPUT_DIR + "input" + str(i) + ".hdf5"
        f = h5py.File(file_name, "w")
        f.create_dataset("probabilities", data=probs)
        f.create_dataset("ground_truths", data=gt)
        f.create_dataset("image_path", data=[img_path.encode('utf8')])
        print("file stored:", file_name)
        f.close()

    def color2trainId_ds20k(self, gtc):
        gt = np.zeros(shape=gtc.shape[:2], dtype=int)
        colors = [self.trainId2label_ds20k[i].color for i in range(len(labels.ds20k_labels))]
        for idx, rgb in enumerate(colors):
            gt[(gtc == rgb).all(2)] = idx
        return gt

    def labelId2TrainId_cityscapes(self, gt_label):
        gt_train = np.zeros(shape=gt_label.shape, dtype=int)
        for labelId in np.unique(gt_label):
            gt_train[gt_label == labelId] = self.label2trainId_cs[labelId].trainId
        return gt_train

    def ds20k(self):
        print("Process DS20k data")
        probs_list = sorted(glob(CONFIG.PROBS_DIR + '**.npy'))
        for i in range(len(probs_list)):
            image_name = os.path.basename(probs_list[i])[:-4]
            probs = np.load(probs_list[i])
            gtc = Image.open(CONFIG.GT_DIR + image_name + ".png")
            gtc = np.array(gtc.resize(probs.shape[:2][::-1]))
            gt = self.color2trainId_ds20k(gtc)
            image_path = CONFIG.IMG_DIR + image_name + ".png"
            self.probs_gt_save(probs, gt, image_path, i)

    def cityscapes(self):
        print("Process Cityscapes data")
        probs_list = sorted(glob(CONFIG.PROBS_DIR + '**/*.npy', recursive=True))
        for i in range(len(probs_list)):
            image_name = os.path.basename(probs_list[i])[:-4]
            probs = np.load(probs_list[i])
            gt_label_path = glob(CONFIG.GT_DIR + "**/*" + image_name + "_gtFine_labelIds.png", recursive=True)[0]
            gt_label = np.array(Image.open(gt_label_path))
            gt = self.labelId2TrainId_cityscapes(gt_label)
            image_path = glob(CONFIG.IMG_DIR + "**/*" + image_name + "_leftImg8bit.png", recursive=True)[0]
            self.probs_gt_save(probs, gt, image_path, i)
