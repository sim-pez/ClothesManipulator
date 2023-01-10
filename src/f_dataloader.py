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

    def __init__(self, filename_data,feat_file,label_file):
        """
        Read file Couples_N_8.txt,maipolations_N_8.txt

        gallary_feats_train.npy (!PROBLEM it's too big!) idea di fare file npy per ogni vector feat!! 
        secondo sol:

        divider usanndo 
        """
        super(Data_Q_T, self).__init__()
        self.N=par.N
        
        self.filename_data = filename_data
        self.hf=self._load_h5_file_with_data(self.filename_data)
        print(self.hf.keys())
        self.q=self.hf['q']
        self.t=self.hf['t']
        self.feat=np.load(feat_file)
        if (self.N==1):
            self.manips=self.hf['manips_vec']
        else:
            self.labels=np.loadtxt(label_file,dtype=int)
        
        print("Dataset is loaded: ",self.__len__())
        
        
    def __getitem__(self, indexes):
       # t_id=self.hf["t"][indexes]
        #q_id=self.hf["q"][indexes]
        t_id=self.t[indexes]
        q_id=self.q[indexes]
        q=self.feat[q_id]
        t=self.feat[t_id]
        
        if(self.N==1):
            manips=torch.tensor(self.manips[indexes])
            manips = manips.unsqueeze(0)
            
        else:
            label_q=self.labels[q_id]
            label_t=self.labels[t_id]
        #print( label_q.shape, label_t.shape,q_id)
            manips=create_n_manip(par.N,label_q,label_t)
        
        """
       
        if self.shuffle:
            ids=np.arange(q.shape[0])
            np.random.shuffle(ids)
            q=q[ids]
            t=t[ids]
            manips=manips[ids]
            
         """
        
        #shuffle manipulation vectors
        #idx = torch.randperm(manips_separated.shape[0])
        #manips_separated = manips_separated[idx].view(manips_separated.size())

        return (q, t,manips,t_id)


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

class Data_Query(data.Dataset):
    def __init__(self,Data_test, gallery_feat,label_data):
        """
        Data_tests (q,t) gallery_feat
        Read file Couples_N_8.txt,maipolations_N_8.txt

        gallary_feats_train.npy (!PROBLEM it's too big!) idea di fare file npy per ogni vector feat!! 
        secondo sol:

        divider usanndo 
        """
        super(Data_Query, self).__init__()
        self.N=par.N
        self.VAL=par.VAL_ORIGINAL
        self.hf=Data_test
        self.q=self.hf['q']#id_query
        self.feat=gallery_feat
        if (self.N==1 or self.VAL ):
            self.manips=self.hf['manips_vec']
            self.t=self.hf['t_label']
        else:
            self.t=self.hf['t']
            self.labels=label_data
        print("Dataset is loaded: ",self.__len__())
    def __getitem__(self, indexes):
         
        q_id=self.q[indexes]
        q=self.feat[q_id]
        if(self.N==1 or self.VAL):
            manips=torch.tensor(self.manips[indexes])
            manips = manips.unsqueeze(0)
            t=self.t[indexes]
            t_id=self.q[indexes]
        else:
            t_id=self.t[indexes]
            t=self.feat[t_id]
            label_q=self.labels[q_id]
            label_t=self.labels[t_id]
            #print( label_q.shape, label_t.shape,q_id)
            manips=create_n_manip(par.N,label_q,label_t)
            
        return (q, t,manips,t_id)
    def __len__(self):
        return self.hf['q'].shape[0]




if __name__=="__main__":
    train_data =Data_Q_T(par.DATA_TRAIN,par.FEAT_TRAIN_SENZA_N,par.LABEL_TRAIN)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True,
                                               num_workers=1,
                                               drop_last=True)
    #test_data =Data_Q_T(par.DATA_TEST,par.FEAT_TEST_SENZA_N,par.LABEL_TEST)

    gallery_feat=np.load(par.FEAT_TEST_SENZA_N)
    test_labels = np.loadtxt(os.path.join(par.ROOT_DIR,par.LABEL_TEST), dtype=int)
    Data_test= h5py.File(par.DATA_TEST)
    
    test_data=Data_Query(Data_test=Data_test,gallery_feat=gallery_feat,label_data=test_labels)
    test_loader=torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False,
    sampler=torch.utils.data.SequentialSampler(test_data),
                                               num_workers=1,
                                               drop_last=False)
   
    tq=tqdm(test_loader)
    for i, sample in enumerate(tq):
        qFeat,tFeat,manips_vec,id_t = sample
        tq.set_description("process batch:{ind}, shapes{s}".format(ind=i,s=(qFeat.shape, manips_vec.shape)))
        
    
        
