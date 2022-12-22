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
#from f_utils import get_manip_array

class Data_Q_T(data.Dataset):

    def __init__(self, filename_data,shuffle=True):
        """
        Read file Couples_N_8.txt,maipolations_N_8.txt

        gallary_feats_train.npy (!PROBLEM it's too big!) idea di fare file npy per ogni vector feat!! 
        secondo sol:

        divider usanndo 
        """
        super(Data_Q_T, self).__init__()
        
        self.filename_data = filename_data
        self.hf=self._load_h5_file_with_data(self.filename_data)
        self.cut_index, self.split_index=self._load_cut_index()
        
        self.shuffle = shuffle
        print("Dataset is loaded")
        
        
    def __getitem__(self, indexes):
        
        q=self.hf['q'][indexes]
        t=self.hf['t'][indexes]
        id_t=self.hf['t_idx'][indexes]
        manips=self.hf['manip'][indexes]
        
        if self.shuffle:
            ids=np.arange(q.shape[0])
            np.random.shuffle(ids)
            q=q[ids]
            t=t[ids]
            manips=manips[ids]
            id_t=id_t[ids]
        
        manips_separated = torch.tensor([self._get_manip_array(x) for x in manips])


        return (q, t, manips_separated,id_t)


    def _load_h5_file_with_data(self, file_name):
       
        hf = h5py.File(file_name)
        return hf
    def _load_cut_index(self):
        with open(par.FILE_CUT_INDEX, 'rb') as fp:
            cut_index = pickle.load(fp)
            fp.close()
        with open(par.FILE_SPLIT_INDEX, 'rb') as fp:
            split_index = pickle.load(fp)
            fp.close()
        return cut_index,split_index
    
    def _get_manip_array(self,v):
        Nsplit=np.split(v,self.split_index[:-1])# 12 list of attr_legnth
        manip_array=np.zeros((12,151),dtype=int)
        
        for j in range(manip_array.shape[0]):
            
            start,end=self.cut_index[j][0],self.cut_index[j][1]
            
            manip_array[j][start:end]=Nsplit[j] #change the part of th matrix relative to attribut j
        return manip_array[~np.all(manip_array == 0, axis=1)] #drop the zero rows

    def __len__(self):
        return self.hf['manip'].shape[0]


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
    test_data =Data_Q_T(par.DATA_TEST,shuffle=False)
    test_loader=fast_loader(test_data,batch_size=10,shuffl=False)
    for i, sample in enumerate(tqdm(test_loader)):
        qFeat,tFeat,mani_vects,id_t = sample
        
        #print(h_0.shape)
        print(id_t[0])


        

        break
