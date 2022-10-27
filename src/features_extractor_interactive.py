#@Fatemah Alhamdoosh
#@imome Pezzulla
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#%%
import argparse
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from dataloader import Data
from model import Extractor
from argument_parser import add_base_args, add_eval_args

import constants as C
DATASET_PATH="dati/Images" #/path/to/dataset/folder/that/contain/img/subfolder"
DATASET_NAME="Shopping100k"
MODELS_DIR="models/Shopping100k"
ready=True
#%%
print("hello ")
#%%

if not torch.cuda.is_available():
    print('Warning: Using CPU')
else:
    torch.cuda.set_device(0)
#%%
print(os.path.abspath(__file__))    
#%%
file_root = r"/home/falhamdoosh/disentagledFeaturesExtractor/splits/"+DATASET_NAME
img_root_path =r"/home/falhamdoosh/disentagledFeaturesExtractor/"+DATASET_PATH

    # load dataset
print('Loading gallery...')
gallery_data = Data(file_root, img_root_path,
                        transforms.Compose([
                            transforms.Resize((C.TARGET_IMAGE_SIZE, C.TARGET_IMAGE_SIZE)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                        ]), mode='test')

gallery_loader = torch.utils.data.DataLoader(gallery_data, batch_size=64, shuffle=False,
                                     sampler=torch.utils.data.SequentialSampler(gallery_data),
                                     num_workers=16,
                                     drop_last=False)
print("Extractor...")
model = Extractor(gallery_data.attr_num, backbone="alexnet", dim_chunk=340)

#%%   

model_pretrained=r"/home/falhamdoosh/disentagledFeaturesExtractor/"+MODELS_DIR+"/extractor_best.pkl"
print('load {path} \n'.format(path=model_pretrained))

#%%
model.load_state_dict(torch.load(model_pretrained))
model.cuda()
model.eval()
#%%   
    #indexing the gallery
gallery_feat = []
with torch.no_grad():
    for i, (img, _) in enumerate(tqdm(gallery_loader)):
        img = img.cuda()
        dis_feat, _ = model(img)
        print(len(dis_feat))
       
        gallery_feat.append(F.normalize(torch.cat(dis_feat, 1)).squeeze().cpu().numpy())

    
np.save(os.path.join("/home/falhamdoosh/disentagledFeaturesExtractor/eval_out", 'gallery_feats_test.npy'), np.concatenate(gallery_feat, axis=0))
print('Saved indexed features at /gallery_feats.npy')

#%%
gallery_feat[310][1]

#%%
"""
export DATASET_PATH="dati/Images" /path/to/dataset/folder/that/contain/img/subfolder"
export DATASET_NAME="Shopping100k"
export MODELS_DIR="models/Shopping100k" #"/path/to/saved/model/checkpoints"

python src/features_extractor.py --dataset_name ${DATASET_NAME} --file_root splits/${DATASET_NAME} 
--img_root ${DATASET_PATH} --load_pretrained_extractor ${MODELS_DIR}/extractor_best.pkl 


"""
