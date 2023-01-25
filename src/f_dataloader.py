import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import h5py
import parameters as par
from tqdm import tqdm
from f_utils import create_n_manip

class Data_Q_T(data.Dataset):
   
    def __init__(self, filename_data,feat_file,label_file,N=par.N):
        super(Data_Q_T, self).__init__()
        self.N=N
        self.filename_data = filename_data
        self.hf=self._load_h5_file_with_data(self.filename_data)
        self.feat=np.load(feat_file)
        self.labels=np.loadtxt(label_file,dtype=int)
        print("Dataset is loaded: ",self.__len__())
        
        
    def __getitem__(self, indexes):
       # t_id=self.hf["t"][indexes]
        #q_id=self.hf["q"][indexes]
        t_id=self.hf['t'][indexes]
        q_id=self.hf['q'][indexes]
        q=self.feat[q_id]
        t=self.feat[t_id]
        label_q=self.labels[q_id]
        label_t=self.labels[t_id]
        manips,n=create_n_manip(self.N,label_q,label_t)
        return (q, t,manips,n)

    def _load_h5_file_with_data(self, file_name):
        hf = h5py.File(file_name)
        return hf

    def __len__(self):
        return self.hf['t'].shape[0]


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
            for index in idx:
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
    def __init__(self,Data_test, gallery_feat,label_data,N=par.N):
        """
        Data_tests (q,t) gallery_feat
        Read file Couples_N_8.txt,maipolations_N_8.txt

        gallary_feats_train.npy (!PROBLEM it's too big!) idea di fare file npy per ogni vector feat!! 
        secondo sol:

        divider usanndo 
        """
        super(Data_Query, self).__init__()
        self.N=N
        self.VAL=par.VAL_ORIGINAL
        self.hf=Data_test
        #self.q=self.hf['q']#id_query
        #self.t=self.hf['t']
        self.feat=gallery_feat
        self.labels=label_data
        print("Dataset is loaded. Size: ",self.__len__())

    def __getitem__(self, indexes):
        
        q_id=self.hf['q'][indexes]
        t_id=self.hf['t'][indexes]
        q=self.feat[q_id]
        label_q=self.labels[q_id]
        label_t=self.labels[t_id]
        manips,n=create_n_manip(self.N,label_q,label_t)
        return (q,label_t,manips,n)
    def __len__(self):
        return self.hf['t'].shape[0]
    def __set_N__(self,newN):
        self.N=newN




if __name__=="__main__":
    train_data =Data_Q_T(par.DATA_TRAIN,par.FEAT_TRAIN_SENZA_N,par.LABEL_TRAIN)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True,
                                               
                                               drop_last=True)
    #test_data =Data_Q_T(par.DATA_TEST,par.FEAT_TEST_SENZA_N,par.LABEL_TEST)

    gallery_feat=np.load(par.FEAT_TEST_SENZA_N)
    test_labels = np.loadtxt(os.path.join(par.ROOT_DIR,par.LABEL_TEST), dtype=int)
    Data_test= h5py.File(par.DATA_TEST)
    print(train_data.__len__())
    test_data=Data_Query(Data_test=Data_test,gallery_feat=gallery_feat,label_data=test_labels)
    test_loader=torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False,
    sampler=torch.utils.data.SequentialSampler(test_data),
                                               
                                               drop_last=False)
   
    def get_variable_legnth(manips_vec):
        manips_vec=manips_vec.numpy()
        f=lambda x: x[~np.all(x== 0, axis=1)]
        list_manips=[[],[],[],[],[],[],[],[]]
        id_x=[[],[],[],[],[],[],[],[]]
        for i in range(len(manips_vec)):
            l=f(manips_vec[i])
            list_manips [len(l)-1].append(l)
            id_x[len(l)-1].append(i)
        return list_manips,id_x



            
  
    """
    tq=tqdm(train_loader)
    for i, sample in enumerate(tq):
        qFeat,tFeat,manips_vec,legnths = sample
        list_manips,id_x=get_variable_legnth(manips_vec)
        index_per=np.random.permutation(len(list_manips))
        for n in index_per:
            list_manips_n=torch.tensor(list_manips[n])
            qFeat_n=qFeat[id_x[n]]
            tFeat_n=tFeat[id_x[n]]
           
        
        tq.set_description("process batch:{ind}, shapes{s}".format(ind=i,s=(qFeat.shape, manips_vec.shape, tFeat.shape,legnths.shape)))
        
        break
   
  
    tq=tqdm(test_loader)
    for i, sample in enumerate(tq):
        qFeat,tFeat,manips_vec ,legnths= sample
       
        tq.set_description("process batch:{ind}, shapes{s}".format(ind=i,s=(qFeat.shape, manips_vec.shape, tFeat.shape,legnths.shape)))
        
    """   