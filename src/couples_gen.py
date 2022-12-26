from cProfile import label
from operator import truediv
from tqdm.auto import tqdm
import constants as C
import random
import torchvision.transforms as transforms
import numpy as np
import h5py
from dataloader import Data
from joblib import Parallel, delayed
import multiprocessing

from f_utils import listify_manip
from utils import split_labels, flatten_labels

def same_zero_attributes(q_splitted, t_splitted):
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


def too_much_distance(q, t, N):
    multi_manip =  np.subtract(q, t)
    manip_list = listify_manip(multi_manip)
    distance = len(manip_list)
    if distance > N:
        return True
    else:
        return False


def find_couples(q_id, labels, splitted_labels, N):
    
    q_splitted = splitted_labels[q_id]
    found_couples = []

    for t_id, t_splitted in enumerate(splitted_labels):
        if not too_much_distance(labels[q_id], labels[t_id], N):
            if not same_zero_attributes(q_splitted, t_splitted):
                found_couples.append((q_id, t_id))
            
    return found_couples


def generate_couples(file_root, img_root_path, N, mode):
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
    
    #features_data = np.load(f'eval_out/feat_{mode}.npy')
    labels = data.label_data
    attr_num = data.attr_num

    print("Finding couples")
    
    splitted_labels = [split_labels(lbl, attr_num) for lbl in labels]

    num_cores = multiprocessing.cpu_count()
    couples_unflattened = Parallel(n_jobs=num_cores)(delayed(find_couples)(q_id, labels, splitted_labels, N) for q_id in tqdm(range(len(labels))))
    
    del data #, feature_data 


    #flatten triplets
    print("processing data...")
    couples = []
    for couples_list in couples_unflattened:
        for couple in couples_list:
            couples.append(couple) 
    del couples_unflattened

    random.shuffle(couples)
    
    q = np.array([couple[0] for couple in couples])
    t = np.array([couple[1] for couple in couples])
    del couples

    #save triplets
    print("writing triplets")
    hf = h5py.File(f'multi_manip/{mode}/couples_N_{N}.h5', 'w')
    hf.create_dataset('q', data=q)
    hf.create_dataset('t', data=t)
    hf.close()

    print("done!")


def check_couples(file_root, img_root_path, N, mode):
    hf = h5py.File(f'multi_manip/train/couples_N_{N}.h5', 'r')

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
    attr_num = data.attr_num

    splitted_labels = [split_labels(lbl, attr_num) for lbl in labels]


    for q_id in tqdm(hf['q']):
        q_lbl = labels[q_id]
        q_splitted = splitted_labels[q_id]
        for t_id in hf['t']:
            if too_much_distance(q_lbl, labels[t_id], N):
                print("found an error: distance is too much")
            if not same_zero_attributes(q_splitted, splitted_labels[t_id]):
                print("found an error: zero attributes are not the same")
            print(labels[q_id])
            print(labels[t_id])
            print("+++++++++++++++++")
                

    
if __name__ == '__main__':
    
    N = 8
    mode = 'test'

    file_root = 'splits/Shopping100k'
    img_root_path = '/Users/simone/Desktop/VMR/Dataset/Shopping100k/Images'

    generate_couples(file_root, img_root_path, N, mode)
    print("checking couples...")
    check_couples(file_root, img_root_path, N, mode)
    print("couple check finished")











