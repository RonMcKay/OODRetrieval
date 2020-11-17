import torch.utils.data as data
import numpy as np
import os
from os.path import isfile, splitext, join
import logging
from PIL import Image


discover_mapping = {0: ('unlabeled', (0, 0, 0))}


class CustomDataset(data.Dataset):
    def __init__(self,
                 root='/path/to/your/image/directory',
                 image_file_extension='.png',
                 transform=None,
                 label_mapping=None):
        self.log = logging.getLogger('CustomDataset')
        self.root = root
        self.transform = transform
        self.label_mapping = label_mapping
        self.filenames = []

        if not os.path.exists(self.root):
            self.log.error('\'{}\' does not exist!'.format(self.root))
            raise ValueError('\'{}\' does not exist!'.format(self.root))

        for f in os.listdir(self.root):
            if isfile(join(self.root, f)) and splitext(f)[-1] == image_file_extension:
                self.filenames.append(join(self.root, f))

    def __getitem__(self, index):
        image = Image.open(self.filenames[index]).convert('RGB')
        target = Image.fromarray(np.zeros((image.size[1], image.size[0])))

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target, self.filenames[index]

    def __len__(self):
        return len(self.filenames)
