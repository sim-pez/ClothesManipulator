# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import os
import numpy as np
from tqdm import tqdm
import faiss
import torch
import h5py
#import torch.nn.functional as F
#import torchvision.transforms as transforms
#from dataloader import Data, DataQuery

from f_model import Extractor, LSTM_ManyToOne
from argument_parser import add_base_args, add_eval_args
from utils import split_labels,  compute_NDCG, get_target_attr
import constants as C
import torch.optim.lr_scheduler as lr_scheduler
from f_dataloader import Data_Q_T,fast_loader,Data_Query
import parameters as par
import torch.nn.functional as F


def calc_accuracy(database,queries,query_labels,test_labels,k,step,dim):
    num_database = database.shape[0]
    num_query = queries.shape[0]
    print("Step:{s} ,Num of image in database is {n1}, Num of query img is {n2}".format(s=step,n1=num_database,n2=num_query))

    index = faiss.IndexFlatL2(dim)
    index.add(database)
    _, knn = index.search(queries, k)
    assert (query_labels.shape[0] == queries.shape[0])
    #TODO essendo l'immagine target è stata usata più volte nel data set,
    #  ogni volta partendo da un'immagine di query diversa,
    #sarebbe utile confrontare quanto le stima dei feat_target che corrispondono
    #  allo stesso target sono simili tra loro
    #si poù confrontare anche a livello di attributi
    #compute top@k acc
    hits = 0
    tq=tqdm(range(num_query))
    for q in tq: # itera i dati predicted_tfeat
        neighbours_idxs = knn[q]# gli indici dei k-feat più simili alla predicted_tfeat[q]
        for n_idx in neighbours_idxs:
            if (test_labels[n_idx] == query_labels[q]).all():
                hits += 1
                break
        tq.set_description("Num of hit {h}".format(h=hits))

    print("Total hits",hits)
    print('Top@{k} accuracy: {acc}'.format(k=k, acc=(hits/num_query)*100))
    """ 
    #compute NDCG
    ndcg = []
    # ndcg_target = []  # consider changed attribute only
    #ndcg_others = []  # consider other attributes

    for q in tqdm(range(num_query)):
        rel_scores = []
    # target_scores = []
        #others_scores = []
        neighbours_idxs = knn[q]
    # indicator = query_inds[q]
        #target_attr = get_target_attr(indicator, gallery_data.attr_num)
        attr_num=np.loadtxt(os.path.join(par.ROOT_DIR, "splits/Shopping100k/attr_num.txt") ,dtype=int)
        target_label = split_labels(query_labels[q],attr_num)
        for n_idx in neighbours_idxs:
            n_label = split_labels(test_labels[n_idx], attr_num)
            # compute matched_labels number
            match_cnt = 0
            others_cnt = 0
            for i in range(len(n_label)):
                if (n_label[i] == target_label[i]).all():
                    match_cnt += 1
            rel_scores.append(match_cnt / len(attr_num))

        ndcg.append(compute_NDCG(np.array(rel_scores)))
    print    
    print('NDCG@{k}: {ndcg}'.format(k=k, ndcg=np.mean(ndcg)))
    """

if __name__ == '__main__':
    torch.cuda.set_device(1)
    #load_pretrained_model
    mode="test"
    gallery_feat=np.load(par.FEAT_TEST_SENZA_N)
    test_labels = np.loadtxt(os.path.join(par.ROOT_DIR,par.LABEL_TEST), dtype=int)
    #load the GT labels of queries images
    #path="/home/falhamdoosh/disentagledFeaturesExtractor/multi_manip/test/couples_N_6_small.h5"
    path=par.DATA_TEST
    Data_test = h5py.File(path)
    t_id=Data_test['t']#id del target 
    query_labels=test_labels[t_id]
        #test_data =Data_Q_T(par.DATA_TEST,par.FEAT_TEST_SENZA_N,par.LABEL_TEST)

    test_data=Data_Query(Data_test=Data_test,gallery_feat=gallery_feat,label_data=test_labels)
    gallery_loader=torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False,
                                          sampler=torch.utils.data.SequentialSampler(test_data),
                                               drop_last=False)
    model=LSTM_ManyToOne(input_size=151,seq_len=par.N,output_size=4080,hidden_dim=4080,n_layers=par.NUM_LAYER,drop_prob=0.5)
    last_train=par.MODEL_EVAL
    path_pretrained_model=os.path.join(par.LOG_DIR,"{last_train}/best_model.pkl".format(last_train=last_train))
    model.load_state_dict(torch.load(path_pretrained_model))
    model.cuda()
    model.eval()
    
    predicted_tfeat = []
    out_all=[]
    with torch.no_grad():
        for i, sample in enumerate(tqdm(gallery_loader)):
            qFeat,label_t,mani_vects= sample
            feat,out_all_batch = model(mani_vects,qFeat)
            #TODO check if we shoul do normalization!
            predicted_tfeat.append(feat.cpu().numpy())
            out_all.append(out_all_batch.cpu().numpy())
    predicted_tfeat= np.concatenate(predicted_tfeat, axis=0)
    #out_all=np.concatenate(out_all, axis=0)
    #print("Shape of out_all after concatinating",len(out_all),len(out_all[0]),len(out_all[0][0]))
    
   # np.save(os.path.join(par.DATA_TEST_DIR, 'predicted_tfeats.npy'),predicted_tfeat )
   # print('Saved indexed features at {dir}/predicted_tfeats.npy'.format(dir=par.DATA_TEST_DIR))
    
     

    #evaluate the top@k results
    dim = 4080  # dimension
    database = gallery_feat
    queries = predicted_tfeat# Dipende dallo step di tempo
    k = 50
    calc_accuracy(database,queries,query_labels,test_labels,k,"last step",dim)
    eval_all=False
    if(eval_all):
        for n in range(par.N-1):
            queries=out_all[:,n,:].copy(order='C')
            calc_accuracy(database,queries,query_labels,test_labels,k,n,dim)
     


  
