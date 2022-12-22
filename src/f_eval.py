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
from f_dataloader import Data_Q_T,fast_loader
import parameters as par
import torch.nn.functional as F


if __name__ == '__main__':
    #load_pretrained_model
    mode="test"
    gallery_feat=np.load(par.FEAT_TEST)
    
    test_data =Data_Q_T(par.DATA_TEST,shuffle=False)
    gallery_loader=fast_loader(test_data,batch_size=32,drop_last=False,shuffl=False)
    model=LSTM_ManyToOne(input_size=151,seq_len=8,output_size=4080,hidden_dim=4080,n_layers=1,drop_prob=0.5)
    loss=torch.nn.MSELoss().cuda()
    last_train="12-22-13:33"
    path_pretrained_model=os.path.join(par.LOG_DIR,"{last_train}/best_model.pkl".format(last_train=last_train))
    model.load_state_dict(torch.load(path_pretrained_model))
    model.cuda()
    model.eval()
    
    predicted_tfeat = []

    with torch.no_grad():
        for i, sample in enumerate(tqdm(gallery_loader)):
            qFeat,tFeat,mani_vects,id_t = sample
            feat,hidden = model(mani_vects,qFeat)
            #TODO check if we shoul do normalization!
            predicted_tfeat.append(F.normalize(feat).cpu().numpy())
    predicted_tfeat= np.concatenate(predicted_tfeat, axis=0)
    np.save(os.path.join(par.DATA_TEST_DIR, 'predicted_tfeats.npy'),predicted_tfeat )
    print('Saved indexed features at {dir}/predicted_tfeats.npy'.format(dir=par.DATA_TEST_DIR))
    
     

    #evaluate the top@k results
    dim = 4080  # dimension
    num_database = gallery_feat.shape[0]  # number of images in database
    num_query = predicted_tfeat.shape[0]  # number of 
    print("Num of image in database is {n1}, Num of query img is {n2}".format(n1=num_database,n2=num_query))

    database = gallery_feat
    queries = predicted_tfeat
    index = faiss.IndexFlatL2(dim)
    index.add(database)
    k = 30
    _, knn = index.search(queries, k)

    #load the GT labels for all gallery images
    label_data = np.loadtxt(os.path.join(par.ROOT_DIR,"splits/Shopping100k/labels_test.txt"), dtype=int)
    #load the GT labels of queries images
    hf = h5py.File(par.DATA_TEST)
    t_id=hf['t_idx']
    query_labels=label_data[t_id]
    
    assert (query_labels.shape[0] == predicted_tfeat.shape[0])
    #TODO essendo l'immagine target è stata usata più volte nel data set,
    #  ogni volta partendo da un'immagine di query diversa,
    #sarebbe utile confrontare quanto le stima dei feat_target che corrispondono
    #  allo stesso target sono simili tra loro
    #compute top@k acc
    hits = 0
    for q in tqdm(range(num_query)): # itera i dati predicted_tfeat
        neighbours_idxs = knn[q]# gli indici dei k-feat più simili alla predicted_tfeat[q]
        for n_idx in neighbours_idxs:
            if (label_data[n_idx] == query_labels[q]).all():
                hits += 1
                break
    print('Top@{k} accuracy: {acc}'.format(k=k, acc=hits/num_query))
"""

    #compute NDCG
    ndcg = []
    ndcg_target = []  # consider changed attribute only
    ndcg_others = []  # consider other attributes

    for q in tqdm(range(num_query)):
        rel_scores = []
        target_scores = []
        others_scores = []

        neighbours_idxs = knn[q]
        indicator = query_inds[q]
        target_attr = get_target_attr(indicator, gallery_data.attr_num)
        target_label = split_labels(gt_labels[q], gallery_data.attr_num)

        for n_idx in neighbours_idxs:
            n_label = split_labels(label_data[n_idx], gallery_data.attr_num)
            # compute matched_labels number
            match_cnt = 0
            others_cnt = 0

            for i in range(len(n_label)):
                if (n_label[i] == target_label[i]).all():
                    match_cnt += 1
                if i == target_attr:
                    if (n_label[i] == target_label[i]).all():
                        target_scores.append(1)
                    else:
                        target_scores.append(0)
                else:
                    if (n_label[i] == target_label[i]).all():
                        others_cnt += 1

            rel_scores.append(match_cnt / len(gallery_data.attr_num))
            others_scores.append(others_cnt / (len(gallery_data.attr_num) - 1))

        ndcg.append(compute_NDCG(np.array(rel_scores)))
        ndcg_target.append(compute_NDCG(np.array(target_scores)))
        ndcg_others.append(compute_NDCG(np.array(others_scores)))

    print('NDCG@{k}: {ndcg}, NDCG_target@{k}: {ndcg_t}, NDCG_others@{k}: {ndcg_o}'.format(k=k,
                                                                                          ndcg=np.mean(ndcg),
                                                                                          ndcg_t=np.mean(ndcg_target),
                                                                                          ndcg_o=np.mean(ndcg_others)))
"""