
import argparse
import datetime
import json

import constants as C
import os
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms
from argument_parser import add_base_args, add_train_args
from dataloader import Data
from loss_function import hash_labels, TripletSemiHardLoss
from f_model import LSTM_ManyToOne
from tqdm import tqdm

torch.manual_seed(100)

vocab_size = len(word2idx) + 1
output_size = 1
embedding_dim = 400
hidden_dim = 512
n_layers = 2

model = LSTM_ManyToOne(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
model.to(device)

lr=0.005
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def train():
    pass

if __name__=="main":

    pass
