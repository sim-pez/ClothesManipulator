import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import h5py

class Data_Q_T(data.Dataset):

    def __init__(self, data_root, filename_data, mode='train',shuffle=True):
        """
        Read file Couples_N_8.txt,maipolations_N_8.txt

        gallary_feats_train.npy (!PROBLEM it's too big!) idea di fare file npy per ogni vector feat!! 
        secondo sol:

        divider usanndo 
        """
        super(Data_Q_T, self).__init__()
        self.data_root = data_root 
        self.filename_data = filename_data
        self.data_ids=self._load_h5_file_with_data(self.filename_data)
        self.mode = mode
        self.shuffle = shuffle
        
    def __getitem__(self, index):

        inputs = self.inputs['data'][index]#return the data with such index.

        if self.shuffle:
            inputs = inputs[torch.randperm(len(index))] # shuffle the data. data at index[0,1,2]=[2,0,1]

        return (inputs,mani_vec, target)
    def _load_h5_file_with_data(self, file_name):
        path = os.path.join(self.dir_path, file_name)
        file = h5py.File(path)
        key = list(file.keys())[0]
        data = file[key]
        return dict(file=file, data=data)
    def __len__(self):
        return self.inputs['data'].shape[0]


class RandomBatchSampler(data.Sampler):
    """Sampling class to create random sequential batches from a given dataset
    E.g. if data is [1,2,3,4] with bs=2. Then first batch, [[1,2], [3,4]] then shuffle batches -> [[3,4],[1,2]]
    This is useful for cases when you are interested in 'weak shuffling'
    :param dataset: dataset you want to batch
    :type dataset: torch.utils.data.Dataset
    :param batch_size: batch size
    :type batch_size: int
    :returns: generator object of shuffled batch indices
    """
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.dataset_length = len(dataset)
        self.n_batches = self.dataset_length / self.batch_size
        self.batch_ids = torch.randperm(int(self.n_batches))

    def __len__(self):
        return self.batch_size

    def __iter__(self):
        for id in self.batch_ids:
            idx = torch.arange(id * self.batch_size, (id + 1) * self.batch_size)
            for index in idx:
                yield int(index)
        if int(self.n_batches) < self.n_batches:
            idx = torch.arange(int(self.n_batches) * self.batch_size, self.dataset_length)
            for index in idx:
                yield int(index)


def fast_loader(dataset, batch_size=300, drop_last=False, transforms=None):
    """Implements fast loading by taking advantage of .h5 dataset
    The .h5 dataset has a speed bottleneck that scales (roughly) linearly with the number
    of calls made to it. This is because when queries are made to it, a search is made to find
    the data item at that index. However, once the start index has been found, taking the next items
    does not require any more significant computation. So indexing data[start_index: start_index+batch_size]
    is almost the same as just data[start_index]. The fast loading scheme takes advantage of this. However,
    because the goal is NOT to load the entirety of the data in memory at once, weak shuffling is used instead of
    strong shuffling.
    :param dataset: a dataset that loads data from .h5 files
    :type dataset: torch.utils.data.Dataset
    :param batch_size: size of data to batch
    :type batch_size: int
    :param drop_last: flag to indicate if last batch will be dropped (if size < batch_size)
    :type drop_last: bool
    :returns: dataloading that queries from data using shuffled batches
    :rtype: torch.utils.data.DataLoader
    """
    return data.DataLoader(
        dataset, batch_size=None,  # must be disabled when using samplers
        sampler=data.BatchSampler(RandomBatchSampler(dataset, batch_size), batch_size=batch_size, drop_last=drop_last)
    )