import argparse
import datetime
import json
import pprint

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
from model import Extractor
from tqdm import tqdm

torch.manual_seed(100)


if __name__ == '__main__':
    
    
    file_root = 'mini_ds/splits/Shopping100k'
    #img_root_path = '/Users/simone/Desktop/VMR/Dataset/Shopping100k/Images'
    img_root_path = 'mini_ds/Images'

    # load dataset
    print('Loading dataset...')
    train_data = Data(file_root,  img_root_path, 
                      transforms.Compose([
                          transforms.Resize((C.TRAIN_INIT_IMAGE_SIZE, C.TRAIN_INIT_IMAGE_SIZE)),
                          transforms.RandomHorizontalFlip(),
                          transforms.CenterCrop(C.TARGET_IMAGE_SIZE),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                      ]), 'train')
    valid_data = Data(file_root,  img_root_path,
                      transforms.Compose([
                          transforms.Resize((C.TARGET_IMAGE_SIZE, C.TARGET_IMAGE_SIZE)),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                      ]), 'test')

    train_loader = data.DataLoader(train_data, shuffle=True, drop_last=True)
    valid_loader = data.DataLoader(valid_data, shuffle=False, drop_last=False)
    

    for imgs, one_hots, labels, indicator in tqdm(train_loader):
        print(imgs)
        print(one_hots)
        print(labels)
        print(indicator)
        
