import os
import numpy as np
from PIL import Image, ImageFile
import torch.utils.data as data
import torchvision.transforms as transforms
from utils import get_idx_label

class Data(data.Dataset):

    def __init__(self, file_root,  img_root_path, ref_ids,  query_inds, img_transform=None,
                    mode='train'):
            super(Data, self).__init__()


    def _load_dataset(self):
        pass
    def __len__(self):
        pass