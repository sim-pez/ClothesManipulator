# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import numpy as np
import torch
import torch.nn as nn
import torchvision
from f_dataloader import Data_Q_T,fast_loader
import parameters as par
from tqdm import tqdm

class Extractor(nn.Module):
    """
    Extract attribute-specific embeddings and add attribute predictor for each.
    Args:
        attr_nums: 1-D list of numbers of attribute values for each attribute
        backbone: String that indicate the name of pretrained backbone
        dim_chunk: int, the size of each attribute-specific embedding
    """
    def __init__(self, attr_nums, backbone='alexnet', dim_chunk=340):
        super(Extractor, self).__init__()

        self.attr_nums = attr_nums
        if backbone == 'alexnet':
            self.backbone = torchvision.models.alexnet(pretrained=True)
            self.backbone.classifier = self.backbone.classifier[:-2]
            dim_init = 4096
        if backbone == 'resnet':
            self.backbone = torchvision.models.resnet18(pretrained=True)
            self.backbone.fc = nn.Sequential()
            dim_init = 512

        dis_proj = []
        for i in range(len(attr_nums)):
            dis_proj.append(nn.Sequential(
                    nn.Linear(dim_init, dim_chunk),
                    nn.ReLU(),
                    nn.Linear(dim_chunk, dim_chunk)
                )
            )
        self.dis_proj = nn.ModuleList(dis_proj)

        attr_classifier = []
        for i in range(len(attr_nums)):
            attr_classifier.append(nn.Sequential(
                nn.Linear(dim_chunk, attr_nums[i]))
            )
        self.attr_classifier = nn.ModuleList(attr_classifier)

    def forward(self, img):
        """
        Returns:
            dis_feat: a list of extracted attribute-specific embeddings
            attr_classification_out: a list of classification prediction results for each attribute
        """
        feat = self.backbone(img)
        dis_feat = []
        for layer in self.dis_proj:
            dis_feat.append(layer(feat))

        attr_classification_out = []
        for i, layer in enumerate(self.attr_classifier):
            attr_classification_out.append(layer(dis_feat[i]).squeeze())
        return dis_feat, attr_classification_out

"""
(32,label_legnth ,N)
input size (batch size, sequence length, input dimension).
 The hidden state and cell state is stored in a tuple with the format

 hidden_state = torch.randn(n_layers, batch_size, hidden_dim) // si inizializa con l'img
cell_state = torch.randn(n_layers, batch_size, hidden_dim)
hidden = (hidden_state, cell_state) 
 (1,32,features_vector_dim)

 -----------------
 # Obtaining the last output
out = out.squeeze()[-1, :]
print(out.shape)
--------------
"""
class LSTM_ManyToOne(nn.Module):
     # domanda è megio tenere i chunk? e addestrare rispetto ai chunck?
     #c_0: tensor of shape (D * \text{num\_layers}, H_{cell})(D∗num_layers,H cell)
    def __init__(self, input_size=151,seq_len=8, output_size=4080, hidden_dim=4080, n_layers=1, drop_prob=0.5):
        super(LSTM_ManyToOne, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.input_size=input_size
        self.seq_len=seq_len
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, dropout=drop_prob, batch_first=True,bidirectional=False)
        self.dropout = nn.Dropout(drop_prob)
        self.fc=nn.Sequential(nn.Linear(hidden_dim,output_size),
                                     nn.ReLU(),
                                     nn.Linear(output_size,output_size))
        print("Model is loaded...")
        
    #ad ogni iterazione viene passato il hidden precedente e il nuovo input    
    def forward(self, x, qFeat):
        batch_size = x.size(0)
        hidden=self.init_hidden(batch_size,qFeat)
        x = x.float().cuda() #is equivalent to self.to(torch.int64)
        lstm_out, hidden = self.lstm(x, hidden)
        
        #take the last output
        lstm_out = lstm_out.contiguous()
        lstm_out = lstm_out[:,-1,:]  
        #  contiguous: this function returns the self tensor \\ copy of tensor
        # view: Returns a new tensor with the same data as the self tensor but of a different shape
        #each new view dimension must either be a subspace of an original dimension, or only span across original dimensions 
        out = self.dropout(lstm_out)
        out = self.fc(out)
        return out, hidden
    def init_hidden(self, batch_size,qFeat):
       
        ########init_hidden
        h_0=tuple([ qFeat for k in range (self.n_layers)])
        h_0=torch.stack(h_0,dim=0)
        print(h_0.shape)
        c_0=torch.zeros(self.n_layers, batch_size, self.hidden_dim,dtype=torch.float32)
        hidden=(h_0.cuda(),c_0.cuda())
        return hidden

if __name__=="__main__":
    test_data =Data_Q_T(par.DATA_TEST,par.FEAT_TEST_SENZA_N,par.LABEL_TEST,shuffle=True)
    test_loader=fast_loader(test_data,batch_size=32)
    model=LSTM_ManyToOne(input_size=151,seq_len=8,output_size=4080,hidden_dim=4080,n_layers=1,drop_prob=0.5)
    torch.cuda.set_device(1)
    model.cuda()
    """
    
    for i, sample in enumerate(tqdm(test_loader)):
        qFeat,tFeat,mani_vects = sample
        print(tFeat,qFeat)
        out,hidden=model(mani_vects,qFeat)
        print(out)
        
        break

        #print(i,out.shape)
    """ 