#used to generate couples from the dataset
import constants as C
import random
import torchvision.transforms as transforms
import numpy as np
import h5py
from dataloader import Data
from utils import cut_index
from numba import jit

from utils import split_labels

@jit
def too_much_distance(q_lbl, t_lbl, N, cut_index_np):
    multi_manip =  np.subtract(q_lbl, t_lbl)
    distance = 0
    for ci in cut_index_np:
        if np.any(multi_manip[ci[0]:ci[1]]):
            distance += 1
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
            if not np.array_equal(q_lbl, labels[t_id]):
                if not too_much_distance(q_lbl, labels[t_id], N, cut_index_np):
                    found_q[q_id * max_manip + count] = q_id
                    found_t[q_id * max_manip + count] = t_id
                    count += 1
            if count == max_manip:
                break
            
    return found_q, found_t


def generate_couples(file_root, img_root_path, N, mode, max_manip):

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
    '''
    read and validate a generated dataset
    '''

    def too_much_distance_old(q, t, N):
        multi_manip =  np.subtract(q, t)
        manip_list_splitted = split_labels(multi_manip, attr_num)
        distance = 0
        for attr in manip_list_splitted:
            found = False
            for e in attr:
                if e != 0:
                    found = True
            if found:
                distance += 1
        if distance > N:
            return True
        else:
            return False

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
    print("Done!")
                

    
if __name__ == '__main__':
    
    N = 8
    max_manip = 100000
    mode = 'train'

    file_root = 'splits/Shopping100k'
    img_root_path = '/Users/simone/Desktop/VMR/Dataset/Shopping100k/Images'

    generate_couples(file_root, img_root_path, N, mode, max_manip)
    check_couples(file_root, img_root_path, N, mode)

    print('now doing for test set')
    mode = 'test'
    generate_couples(file_root, img_root_path, N, mode, max_manip)
    check_couples(file_root, img_root_path, N, mode)