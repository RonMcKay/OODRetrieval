import numpy as np
import pickle as pkl
from torch.utils.data import Dataset
import h5py
import os
from os.path import join, isfile, splitext, basename, exists


class MetaSegData(Dataset):
    def __init__(self,
                 root='/data/poberdie/metaseg',
                 dataset='cityscapes',
                 model_name='deeplabv3plus',
                 loading=('input',
                          'metrics',
                          'components'),
                 class_dtype='probs'):
        super().__init__()
        self.root = root
        self.dataset = dataset
        self.model_name = model_name
        self.loading = loading
        self.class_dtype = class_dtype

        #         self._dir_names = {'INPUT_DIR': 'input',
        #                            'METRICS_DIR': 'metrics',
        #                            'COMPONENTS_DIR': 'components',
        #                            'IOU_SEG_VIS_DIR': 'iou_seg_vis',
        #                            'RESULTS_DIR': 'results',
        #                            'STATS_DIR': 'stats'}
        self._file_extensions = {'input': '.hdf5',
                                 'metrics': '.p',
                                 'components': '.p'}

        # get number of inputs:
        self.num_imgs = 0
        for f in os.listdir(join(self.root, self.loading[0], self.model_name, self.dataset)):
            if isfile(join(self.root, self.loading[0], self.model_name, self.dataset, f)) \
                    and splitext(f)[-1] == self._file_extensions[self.loading[0]] \
                    and all([exists(join(self.root,
                                         self.loading[i],
                                         self.model_name,
                                         self.dataset,
                                         self.loading[i] + str(self._index_of_file(f)) \
                                         + self._file_extensions[self.loading[i]])) for i in
                             range(1, len(self.loading))]):
                self.num_imgs += 1

    def __getitem__(self, index):
        data = []
        for i in self.loading:
            d = getattr(self, '_load_{}'.format(i))(index)
            if isinstance(d, tuple):
                data.extend(d)
            else:
                data.append(d)

        return (*data,)

    def __len__(self):
        return self.num_imgs

    def _load_input(self, index):
        filename = join(self.root, 'input', self.model_name, self.dataset, 'input{}.hdf5'.format(index))
        with h5py.File(filename, 'r') as f:
            probs = np.asarray(f['probabilities'])
            gt = np.asarray(f['ground_truths'])
            probs = np.squeeze(probs)
            gt = np.squeeze(gt)
            image_path = f['image_path'][0].decode('utf8')
        return probs, gt, image_path

    def _load_metrics(self, index):
        filename = join(self.root, 'metrics', self.model_name, self.dataset, 'metrics{}.p'.format(index))
        with open(filename, 'rb') as f:
            metrics = pkl.load(f)

        M = np.asarray([np.asarray(metrics[m]) for m in metrics.keys()])
        return M

    def _load_components(self, index):
        filename = join(self.root, 'components', self.model_name, self.dataset, 'components{}.p'.format(index))
        with open(filename, 'rb') as f:
            components = pkl.load(f)
        return components

    def _index_of_file(self, filename):
        f = splitext(basename(filename))[0]
        for i in self._file_extensions.keys():
            f = f.replace(i, '')
        f = f.replace('_', '')
        return int(f)
