## What is MetaSeg:

![MetaSeg](MetaSeg.jpg)

MetaSeg is a post-processing tool for quantifying the reliability of neural networks (NNs) for semantic segmentation which can be easily added on top every NN. MetaSeg is a method that treats NNs like blackboxes, i.e. only using the NNs' softmax output. Based on that output, different aggregated dispersion measures are derived at segment level in order to fit another, preferably low complex and interpretable, model that indicates the prediction uncertainty per segment. For each segment/connected component in the segmentation mask, MetaSeg, on the one hand, provides a method predicting whether this particular component intersects with the ground truth or not. Using the Intersection over Union (IoU) as performance measure, this latter task can be understood as "meta" classifying between the two classes {IoU=0} and {IoU>0}. On the other hand, MetaSeg ultimately also provides a method for quantifying the uncertainty of each predicted segment by predicting IoU values via regression, i.e. rating how well (accoding to IoU) each segment is predicted.

For further reading, please refer to http://arxiv.org/abs/1811.00648.

## Preparation:
We assume that the user is already using a neural network for semantic segmentation and a corresponding dataset. For each image from the segmentation dataset, MetaSeg requires a hdf5 file that contains the following data:

- a three-dimensional numpy array (height, width, classes) that contains the softmax probabilities computed for the current image
- the full filepath of the current input image
- a two-dimensional numpy array (height, width) that contains the ground truth class indices for the current image

MetaSeg provides a class object in "prepare_data.py" for generating these hdf5 files. Before running MetaSeg, please edit all necessary paths stored in "global_defs.py". MetaSeg is CPU based and parts of MetaSeg trivially parallize over the number of input images, adjust "NUM_CORES" in "global_defs.py" to make use of this. Also, in the same file, select the tasks to be executed by setting the corresponding boolean variable (True/False).

## Run Code:
```sh
./x.sh
```

## Deeplabv3+ and Cityscapes:
The results in http://arxiv.org/abs/1811.00648 have been obtained from two Deeplabv3+ networks (https://github.com/tensorflow/models/tree/master/research/deeplab) together with the Cityscapes dataset (https://www.cityscapes-dataset.com/). For using the latter you need to enroll at Cityscapes on the website. For details on using Deeplabv3+ networks in combination with cityscapes, we refer to the README page https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/cityscapes.md.

## Packages and their versions we used:
We used the Python 3.5.2. Please make sure to have the following packages installed:

- Cython==0.29.13
- h5py==2.10.0
- matplotlib==3.0.3
- numpy==1.17.2
- pandas==0.24.2
- Pillow==6.1.0
- scikit-learn==0.21.3

See also requirements.txt.

## Authors:
Matthias Rottmann (University of Wuppertal), Thomas Paul Hack (Leipzig University), Pascal Colling (University of Wuppertal), Robin Chan (University of Wuppertal), Kira Maag (University of Wuppertal)
