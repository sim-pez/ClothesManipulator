#@Fatemah Alhamdoosh
#@imome Pezzulla
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_base_args(parser)
    add_eval_args(parser)
    args = parser.parse_args()
    if not args.use_cpu and not torch.cuda.is_available():
        print('Warning: Using CPU')
        args.use_cpu = True
    else:
        torch.cuda.set_device(args.gpu_id)

    file_root = args.file_root
    img_root_path = args.img_root

    # load dataset
    print('Loading gallery...')
    gallery_data = Data(file_root, img_root_path,
                        transforms.Compose([
                            transforms.Resize((C.TARGET_IMAGE_SIZE, C.TARGET_IMAGE_SIZE)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                        ]), mode='test')

    gallery_loader = torch.utils.data.DataLoader(gallery_data, batch_size=args.batch_size, shuffle=False,
                                     sampler=torch.utils.data.SequentialSampler(gallery_data),
                                     num_workers=args.num_threads,
                                     drop_last=False)

    model = Extractor(gallery_data.attr_num, backbone=args.backbone, dim_chunk=args.dim_chunk)
    

    if args.load_pretrained_extractor:
        print('load {path} \n'.format(path=args.load_pretrained_extractor))
        model.load_state_dict(torch.load(args.load_pretrained_extractor))
    else:
        print('Pretrained extractor not provided. Use --load_pretrained_extractor or the model will be randomly initialized.')
    if not os.path.exists(args.feat_dir):
        os.makedirs(args.feat_dir)

    if not args.use_cpu:
        model.cuda()
        

    model.eval()
   

    #indexing the gallery
    gallery_feat = []
    with torch.no_grad():
        for i, (img, _) in enumerate(tqdm(gallery_loader)):
            if not args.use_cpu:
                img = img.cuda()

            dis_feat, _ = model(img)
            gallery_feat.append(F.normalize(torch.cat(dis_feat, 1)).squeeze().cpu().numpy())

    
    np.save(os.path.join(args.feat_dir, 'gallery_feats.npy'), np.concatenate(gallery_feat, axis=0))
    print('Saved indexed features at {dir}/gallery_feats.npy'.format(dir=args.feat_dir))
"""
export DATASET_PATH="/path/to/dataset/folder/that/contain/img/subfolder"
export DATASET_NAME="Shopping100k"
export MODELS_DIR="/path/to/saved/model/checkpoints"

python src/eval.py --dataset_name ${DATASET_NAME} --file_root splits/${DATASET_NAME} 
--img_root ${DATASET_PATH} --load_pretrained_extractor ${MODELS_DIR}/checkpoints/extractor_best.pkl 


"""
