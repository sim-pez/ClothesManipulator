#Used to extract all features from dataset

import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from dataloader import Data
from model import Extractor
from argument_parser import add_base_args, add_eval_args

import constants as C
DATASET_PATH="dati/Images" 
DATASET_NAME="Shopping100k"
MODELS_DIR="models/Shopping100k"
ready=True

if not torch.cuda.is_available():
    print('Warning: Using CPU')
else:
    torch.cuda.set_device(0)
 
file_root = r"/home/falhamdoosh/disentagledFeaturesExtractor/splits/"+DATASET_NAME
img_root_path =r"/home/falhamdoosh/disentagledFeaturesExtractor/"+DATASET_PATH

    # load dataset
print('Loading gallery...')
gallery_data = Data(file_root, img_root_path,
                        transforms.Compose([
                            transforms.Resize((C.TARGET_IMAGE_SIZE, C.TARGET_IMAGE_SIZE)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                        ]), mode='train')

gallery_loader = torch.utils.data.DataLoader(gallery_data, batch_size=64, shuffle=False,
                                     sampler=torch.utils.data.SequentialSampler(gallery_data),
                                     num_workers=16,
                                     drop_last=False)

model = Extractor(gallery_data.attr_num, backbone="alexnet", dim_chunk=340)

model_pretrained=r"/home/falhamdoosh/disentagledFeaturesExtractor/"+MODELS_DIR+"/extractor_best.pkl"
print('load {path} \n'.format(path=model_pretrained))

model.load_state_dict(torch.load(model_pretrained))
model.cuda()
model.eval()

#indexing the gallery
gallery_feat = []
with torch.no_grad():
    for i, (img, _) in enumerate(tqdm(gallery_loader)):
        img = img.cuda()
        dis_feat, _ = model(img)
        gallery_feat.append(torch.cat(dis_feat, 1).squeeze().cpu().numpy())
        
dim_chunk=340
gallery_feat = np.concatenate(gallery_feat, axis=0).reshape(-1,dim_chunk * len(gallery_data.attr_num))  

print("shape of gallery_feat_test: {}".format(len(gallery_feat)))
np.save("/home/falhamdoosh/disentagledFeaturesExtractor/eval_out/feat_train_senzaNorm.npy", gallery_feat)
print('Saved indexed features at /feat_train_senzaNorm.npy')