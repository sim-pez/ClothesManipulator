# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
import os
import numpy as np
import torch
import h5py
from f_model import LSTM_ManyToOne
from f_utils import calc_accuracy,eval_help,eval_variable_help
from f_dataloader import Data_Query
import parameters as par


if __name__ == '__main__':
    torch.cuda.set_device(0)
    
    gallery_feat=np.load(par.FEAT_TEST_SENZA_N)
    test_labels = np.loadtxt(os.path.join(par.ROOT_DIR,par.LABEL_TEST), dtype=int)
    path=par.DATA_TEST
    Data_test = h5py.File(path)
    t_id=Data_test['t']#id del target 
    query_labels=test_labels[t_id]
    test_data=Data_Query(Data_test=Data_test,gallery_feat=gallery_feat,label_data=test_labels,N=par.N)
    gallery_loader=torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False,
                                          sampler=torch.utils.data.SequentialSampler(test_data),
                                               drop_last=False)
    
    model=LSTM_ManyToOne(input_size=151,seq_len=par.N,output_size=4080,hidden_dim=4080,n_layers=par.NUM_LAYER,drop_prob=0.5)
    last_train=par.MODEL_EVAL
    path_pretrained_model=os.path.join(par.LOG_DIR,"{last_train}/best_model.pkl".format(last_train=last_train))
    #load_pretrained_model
    model.load_state_dict(torch.load(path_pretrained_model))
    model.cuda()
    if (par.Eval_variable_legnth):
        predicted_tfeat=eval_variable_help(model,gallery_loader)
    else:
        predicted_tfeat= eval_help(model,gallery_loader)
    log_dir=os.path.join(par.LOG_DIR,"log_eval.txt")
    #evaluate the top@k results
    dim = 4080  # dimension
    database = gallery_feat
    queries = predicted_tfeat# Dipende dallo step di tempo
    k = par.K
    print("N is :",par.N)
    acc,res=calc_accuracy(database,queries,query_labels,test_labels,k,"last step",dim)
    with open(log_dir, 'a') as f:
        f.write("\n N is :{n}, res: {res}".format(n=par.N,res=res))
    eval_all=True
    #dataset_distance=par.all_data[par.name_data_set]
    if(eval_all):
        for n in range(1,par.N):
            test_data.__set_N__(n)
            print("N is :",n)
            predicted_tfeat= eval_help(model,gallery_loader)
            queries = predicted_tfeat
            acc,res=calc_accuracy(database,queries,query_labels,test_labels,k,n,dim)
            with open(log_dir, 'a') as f:
                f.write("\n N is :{n1}, res: {res}".format(n1=n,res=res))


     


  
