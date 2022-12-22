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
    query_feat=np.load(par.FEAT_TEST)
    query_labels = np.loadtxt(os.path.join(par.ROOT_DIR,"split/Shopping100k/labels_test"), dtype=int)

    test_data =Data_Q_T(par.DATA_TEST,shuffle=False)
    gallery_loader=fast_loader(test_data,batch_size=32,drop_last=False,shuffl=False)
    model=LSTM_ManyToOne(input_size=151,seq_len=8,output_size=4080,hidden_dim=4080,n_layers=1,drop_prob=0.5)
    loss=torch.nn.MSELoss().cuda()
    last_train="12-22-10:54"
    path_pretrained_model=os.path.join(par.LOG_DIR,"{last_train}/best_model.pkl".format(last_train=last_train))
    model.load_state_dict(torch.load(path_pretrained_model))
    model.cuda()
    model.eval()
    #indexing the gallery
    
    hf = h5py.File(par.DATA_TEST)
    t_id=hf['t_idx']
    t_labels=query_labels[t_id]
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
    
    assert (t_labels.shape[0] == predicted_tfeat.shape[0])

  

    with torch.no_grad():
        for i, (img, indicator) in enumerate(tqdm(query_loader)):
            indicator = indicator.float()
            if not args.use_cpu:
                img = img.cuda()
                indicator = indicator.cuda()

            dis_feat, _ = model(img)
            residual_feat = lstm(indicator)
            feat_manip = torch.cat(dis_feat, 1) + residual_feat

            query_fused_feats.append(F.normalize(feat_manip).cpu().numpy())

    if args.save_matrix:
        np.save(os.path.join(args.feat_dir, 'query_fused_feats.npy'), np.concatenate(query_fused_feats, axis=0))
        print('Saved query features at {dir}/query_fused_feats.npy'.format(dir=args.feat_dir))

    #evaluate the top@k results
    gallery_feat = np.concatenate(gallery_feat, axis=0).reshape(-1, args.dim_chunk * len(gallery_data.attr_num))
    fused_feat = np.array(np.concatenate(query_fused_feats, axis=0)).reshape(-1, args.dim_chunk * len(gallery_data.attr_num))
    dim = args.dim_chunk * len(gallery_data.attr_num)  # dimension
    num_database = gallery_feat.shape[0]  # number of images in database
    num_query = fused_feat.shape[0]  # number of queries

    database = gallery_feat
    queries = fused_feat
    index = faiss.IndexFlatL2(dim)
    index.add(database)
    k = args.top_k
    _, knn = index.search(queries, k)

    #load the GT labels for all gallery images
    label_data = np.loadtxt(os.path.join(file_root, 'labels_test.txt'), dtype=int)

    #compute top@k acc
    hits = 0
    for q in tqdm(range(num_query)):
        neighbours_idxs = knn[q]
        for n_idx in neighbours_idxs:
            if (label_data[n_idx] == gt_labels[q]).all():
                hits += 1
                break
    print('Top@{k} accuracy: {acc}'.format(k=k, acc=hits/num_query))

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

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.empty_cache()
    # load dataset
    print('Loading dataset...')
    train_data =Data_Q_T(par.DATA_TRAIN,shuffle=True)
    test_data =Data_Q_T(par.DATA_TEST,shuffle=False)
    
    train_loader=fast_loader(train_data,batch_size=32)
    test_loader=fast_loader(test_data,batch_size=32,drop_last=False)
    model=LSTM_ManyToOne(input_size=151,seq_len=8,output_size=4080,hidden_dim=4080,n_layers=1,drop_prob=0.5)
    # create the folder to save log, checkpoints and args config
    
    loss=torch.nn.MSELoss().cuda()
    trainer=Trainer(gpu=0,data_loader_train=train_loader, data_loader_test=test_loader,
    loss=loss,model=model,num_epochs=10)
    trainer.run()