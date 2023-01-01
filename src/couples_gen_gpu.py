from cProfile import label
from operator import truediv
from tqdm.auto import tqdm
import constants as C
import random
import torchvision.transforms as transforms
import numpy as np
import h5py
from dataloader import Data
from f_utils import cut_index
from numba import jit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

from f_utils import listify_manip
from utils import split_labels

@jit
def same_zero_attributes(q_lbl, t_lbl, cut_index_np):
    '''
    return True if q and t have the same all-zero attributes but are not the same array
    '''    
    if np.array_equal(q_lbl, t_lbl):
        return False
    for ci in cut_index_np:
        if (not np.any(q_lbl[ci[0]:ci[1]])) != (not np.any(t_lbl[ci[0]:ci[1]])):
            return False
    return True

@jit
def too_much_distance(q, t, N):
    #only works for same zero attributes arrays
    multi_manip =  np.subtract(q, t)
    distance = np.count_nonzero(multi_manip == 1)
    if distance > N:
        return True
    else:
        return False

@jit
def find_couples(labels, N, max_manip):

    n_labels = labels.shape[0]

    found_q = np.full(n_labels * max_manip, -1, dtype=int)
    found_t = np.full(n_labels * max_manip, -1, dtype=int)

    cut_index_np = np.array(cut_index)

    
    for q_id, q_lbl in enumerate(labels):
    
        count = 0

        t_indexes = np.arange(n_labels)
        np.random.shuffle(t_indexes)

        for t_id in t_indexes:
            if same_zero_attributes(q_lbl, labels[t_id], cut_index_np):
                if not too_much_distance(q_lbl, labels[t_id], N):
                    found_q[q_id * max_manip + count] = q_id
                    found_t[q_id * max_manip + count] = t_id
                    count += 1
            if count == max_manip:
                break
            
    return found_q, found_t


def generate_couples(file_root, img_root_path, N, mode, max_manip):
    '''
    Finds couples s.t.:
    - they have the same all-zero attributes
    - their distance is <= N 
      Output file in multi_manip.


    # to reopen file:
    # hf = h5py.File(f'multi_manip/train/couples_N_{N}.h5', 'r')
    # q = hf['q'][0]
    # t = hf['t'][0]
    # hf.close()
    '''        

    #load attributes
    print('Loading attributes')
    if mode == 'train':
        data = Data(file_root,  img_root_path, 
                          transforms.Compose([
                              transforms.Resize((C.TRAIN_INIT_IMAGE_SIZE, C.TRAIN_INIT_IMAGE_SIZE)),
                              transforms.RandomHorizontalFlip(),
                              transforms.CenterCrop(C.TARGET_IMAGE_SIZE),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                          ]), 'train')
    elif mode == 'test':
        data = Data(file_root,  img_root_path,
                      transforms.Compose([
                          transforms.Resize((C.TARGET_IMAGE_SIZE, C.TARGET_IMAGE_SIZE)),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                      ]), 'test')
    else:
        print("Argument mode not valid")
    
    labels = data.label_data

    print("Finding couples")
    q_list, t_list = find_couples(labels, N, max_manip)


    q_list = [e for e in q_list if e != -1]
    t_list = [e for e in t_list if e != -1]

    assert len(q_list) == len(t_list)

    couples = list(zip(q_list, t_list))
    random.shuffle(couples)
    q_list, t_list = zip(*couples)
    
    #save triplets
    print("writing triplets")
    hf = h5py.File(f'multi_manip/{mode}/couples_N_{N}.h5', 'w')
    hf.create_dataset('q', data=np.array(q_list))
    hf.create_dataset('t', data=np.array(t_list))
    hf.close()

    print("done!")


def check_couples(file_root, img_root_path, N, mode):

    def too_much_distance_old(q, t, N):
        multi_manip =  np.subtract(q, t)
        manip_list = listify_manip(multi_manip)
        distance = len(manip_list)
        if distance > N:
            return True
        else:
            return False

    def same_zero_attributes_old(q_splitted, t_splitted):
        '''
        return True if q and t have the same all-zero attributes but are not the same array
        '''    
        same = True
        for attr_q, attr_t in zip(q_splitted, t_splitted):
            if all(e == 0 for e in attr_t) != all(f == 0 for f in attr_q):
                return False
            if not np.array_equal(attr_q, attr_t):
                same = False
        if same:
            return False
        return True

    hf = h5py.File(f'multi_manip/{mode}/couples_N_{N}.h5', 'r')

    if mode == 'train':
        data = Data(file_root,  img_root_path, 
                          transforms.Compose([
                              transforms.Resize((C.TRAIN_INIT_IMAGE_SIZE, C.TRAIN_INIT_IMAGE_SIZE)),
                              transforms.RandomHorizontalFlip(),
                              transforms.CenterCrop(C.TARGET_IMAGE_SIZE),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                          ]), 'train')
    elif mode == 'test':
        data = Data(file_root,  img_root_path,
                      transforms.Compose([
                          transforms.Resize((C.TARGET_IMAGE_SIZE, C.TARGET_IMAGE_SIZE)),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                      ]), 'test')
    else:
        print("Argument mode not valid")
    
    print("Checking couples...")

    labels = data.label_data
    attr_num = data.attr_num

    splitted_labels = [split_labels(lbl, attr_num) for lbl in labels]

    for q_id, t_id in zip(hf['q'], hf['t']):
        if too_much_distance_old(labels[q_id], labels[t_id], N):
            print("Found an error: distance is too much")
            print(f"Q: {q_id}, T: {t_id}")
        if not same_zero_attributes_old(splitted_labels[q_id], splitted_labels[t_id]):
            print("Found an error: zero attributes are not the same")
            print(f"Q: {q_id}, T: {t_id}")

    print("Done!")
                

    
if __name__ == '__main__':
    
    N = 8
    max_manip = 100
    mode = 'train'

    file_root = 'splits/Shopping100k'
    img_root_path = '/Users/simone/Desktop/VMR/Dataset/Shopping100k/Images'

    generate_couples(file_root, img_root_path, N, mode, max_manip)
    check_couples(file_root, img_root_path, N, mode)

    ########TODO delete #############
    print('now doing for test set')
    mode = 'test'
    generate_couples(file_root, img_root_path, N, mode, max_manip)
    check_couples(file_root, img_root_path, N, mode)
    #############################









