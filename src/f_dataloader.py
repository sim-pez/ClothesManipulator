import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import h5py
import random
import parameters as par
from tqdm import tqdm
import pickle
from f_utils import listify_manip, create_n_manip

class Data_Q_T(data.Dataset):

    def __init__(self, filename_data,feat_file,label_file,shuffle=True):
        """
        Read file Couples_N_8.txt,maipolations_N_8.txt

        gallary_feats_train.npy (!PROBLEM it's too big!) idea di fare file npy per ogni vector feat!! 
        secondo sol:

        divider usanndo 
        """
        super(Data_Q_T, self).__init__()
        
        self.filename_data = filename_data
        self.hf=self._load_h5_file_with_data(self.filename_data)
        self.feat=np.load(feat_file)
        self.labels=np.loadtxt(label_file,dtype=int)
        self.shuffle = shuffle
        print("Dataset is loaded")
        
        
    def __getitem__(self, indexes):
        
        #q=self.hf['q'][indexes]
        #t=self.hf['t'][indexes]
       # id_t=self.hf['t_idx'][indexes]
        #manips=self.hf['manip'][indexes]
        t_id=self.hf["t"][indexes]
        q_id=self.hf["q"][indexes]
        q=self.feat[q_id]
        t=self.feat[t_id]
        label_q=self.labels[q_id]
        label_t=self.labels[t_id]
        #manips=np.random.rand(q.shape[0],8,151)
        #print(manips)

        manips=torch.tensor([create_n_manip(par.N,x,y) for x,y in zip(label_q,label_t)])
        
        if self.shuffle:
            ids=np.arange(q.shape[0])
            np.random.shuffle(ids)
            q=q[ids]
            t=t[ids]
            manips=manips[ids]
            
        
        #manips_separated = torch.tensor([listify_manip(x) for x in manips])
        #shuffle manipulation vectors
       # idx = torch.randperm(manips_separated.shape[0])
        #manips_separated = manips_separated[idx].view(manips_separated.size())

        return (q, t, manips,t_id)


    def _load_h5_file_with_data(self, file_name):
        hf = h5py.File(file_name)
        return hf

    def __len__(self):
        return self.hf['q'].shape[0]


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
    def __init__(self, dataset, batch_size,shuffl=True):
        self.batch_size = batch_size
        self.dataset_length = len(dataset)
        self.n_batches = self.dataset_length / self.batch_size
        self.batch_ids=torch.arange(0,int(self.n_batches))
        if shuffl==True:
            self.batch_ids = torch.randperm(int(self.n_batches))

    def __len__(self):
        return self.batch_size

    def __iter__(self):
        for id in self.batch_ids:
            idx = torch.arange(id * self.batch_size, (id + 1) * self.batch_size)
            for index in idx:# @Fatemah #TODO forse non serve questa for
                yield int(index)
        if int(self.n_batches) < self.n_batches:
            idx = torch.arange(int(self.n_batches) * self.batch_size, self.dataset_length)
            for index in idx:
                yield int(index)


def fast_loader(dataset, batch_size=300, drop_last=False, transforms=None,shuffl=True):
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
        sampler=data.BatchSampler(RandomBatchSampler(dataset, batch_size,shuffl), batch_size=batch_size, drop_last=drop_last)
    )

if __name__=="__main__":
    train_data =Data_Q_T(par.DATA_TRAIN,par.FEAT_TRAIN_SENZA_N,par.LABEL_TRAIN,shuffle=True)
    train_loader=fast_loader(train_data,batch_size=32,shuffl=True)
    #test_data =Data_Q_T(par.DATA_TEST,par.FEAT_TEST_SENZA_N,shuffle=False)
    #test_loader=fast_loader(test_data,batch_size=10,shuffl=False)
    for i, sample in enumerate(tqdm(train_loader)):
        qFeat,tFeat,mani_vects,id_t = sample
        print(qFeat.shape)
        print(mani_vects.shape)
        print(tFeat.shape)
        print(qFeat[10][100:110])
        print (train_data.__len__())
        break

