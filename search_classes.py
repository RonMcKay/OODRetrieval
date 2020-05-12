"""This script runs through the whole A2D2 Dataset and saves the image index for each class.

This way you can specify the 'CLASSINDEX' within the configuration.py and only process images that have this specific
class on them. A precomputed version is included in the repository.
"""

from src.datasets.a2d2 import A2D2
from configuration import CONFIG
from src.MetaSeg.functions.in_out import get_indices
import pickle as pkl
from src.imageaugmentations import ToTensor
from os.path import join
from tqdm import tqdm

dat = A2D2(transform=ToTensor())

inds = get_indices(join(CONFIG.metaseg_io_path, 'input', 'deeplabv3plus', 'a2d2'))
print('Counting data...')
selected_classes = {c: [] for c in range(55)}
for ind in tqdm(inds, total=len(inds)):
    _, y, _ = dat[ind]
    for c in list(y.unique().squeeze().numpy()):
        selected_classes[c].append(ind)

print('Filtering empty lists...')
keys = [k for k, v in selected_classes.items() if len(v) > 0]
selected_classes = {k: selected_classes[k] for k in keys}

print('Saving to file...')
with open(join(CONFIG.metaseg_io_path, 'a2d2_dataset_overview.p'), 'wb') as f:
    pkl.dump(selected_classes, f)
