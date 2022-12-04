from cProfile import label
from operator import truediv
from tqdm.auto import tqdm
import constants as C
import random
import torchvision.transforms as transforms
import numpy as np
import h5py
from dataloader import Data
from pprint import pprint
from joblib import Parallel, delayed
import multiprocessing

from utils import split_labels, flatten_labels

def multi_manipulation_gen(file_root, img_root_path, N, max_manip, mode):
    '''
    Output files:
    couples.txt  ->  couples of Q and T that have the same non-zero attribute index (problably it is only for internal use)
    selected_couples_for_manip.txt  ->  couples of couples.txt which len(non-zero differente attributes) > N
    manipulations.txt  ->  generates manipulation vector from couples of selected_couples_for_manip.txt
    '''        

    def create_n_manipulations(q_labels, t_labels, N, attr_num):
        '''
        generate a multi-manipulation of N manipulations.
        Return None if number of candidate attributes are not enough to do random manipulations.
        Removes manipulations if difference of Q and T is > N
        '''

        def find_available_attribute(manipulations_splitted, q_labels_splitted, attr_num):
            '''
            find attribute id which is still not manipulated and it's label is not all zero
            '''

            return [i for i, m in enumerate(manipulations_splitted)
                        if all(v == 0 for v in m)
                             if 1 in q_labels_splitted[i]]

        def random_manipulate(manipulations_splitted, attr, labels_splitted):
            '''
                takes attribute labels_splitted[attr] and adds a manipulation saved on manipulations_splitted.
            '''

            negative_idx = np.where(labels_splitted[attr] == 1)[0][0]
            manipulations_splitted[attr][negative_idx] = -1

            #not tested yet
            zero_indexes = [i for i, x in enumerate(manipulations_splitted[attr]) if x!=-1]
            positive_idx = random.choice(zero_indexes)
            manipulations_splitted[attr][positive_idx] = 1

            return manipulations_splitted

        def casual_remove_manipulations(manipulations_splitted, r):
            '''
                removes r manipulation casually for manipulations_splitted
            '''

            candidate_attributes = [i for i, attr in enumerate(manipulations_splitted) if 1 in attr]
            random.shuffle(candidate_attributes)
            for attr in candidate_attributes[0: r]:
                manipulations_splitted[attr] = np.zeros(len(manipulations_splitted[attr]), dtype = int)

            return manipulations_splitted

        q_labels_splitted = split_labels(q_labels, attr_num)

        #extract needed manipulation to go from q to t
        manipulations = t_labels - q_labels
        manipulations_splitted = split_labels(manipulations, attr_num)
        needed = max(np.count_nonzero(manipulations == 1), np.count_nonzero(manipulations == -1))
        remaining = N - needed

        # remove casually some manip if too many
        if remaining < 0:
            manipulations_splitted = casual_remove_manipulations(manipulations_splitted, - remaining)

        #extract N - needed random manipulations
        elif remaining > 0:
            for _ in range(remaining):
                candidate_attributes = find_available_attribute(manipulations_splitted, q_labels_splitted, attr_num)
                if len(candidate_attributes) < remaining:
                    return None
                attribute = random.choice(candidate_attributes)
                manipulations_splitted = random_manipulate(manipulations_splitted, attribute, q_labels_splitted)

        multi_manipulation = flatten_labels(manipulations_splitted)

        return multi_manipulation
    
    def is_couple_match(q_labels, t_labels, N, attr_num):

        def count_available_attributes(q_labels, manipulations, attr_num):

            minus_manip = list(manipulations)
            for i,e in enumerate(minus_manip):
                if e == 1:
                    minus_manip[i] = 0

            avail_labels = q_labels + minus_manip
            avail_labels_splitted = split_labels(avail_labels, attr_num)
            
            count = 0
            for attr in avail_labels_splitted:
                if 1 in attr:
                    count += 1
            
            return count

            
        t_labels_splitted = split_labels(t_labels, attr_num)
        q_labels_splitted = split_labels(q_labels, attr_num)


        #check if zero attributes are the same
        for attr_q, attr_t in zip(q_labels_splitted, t_labels_splitted):
            if all(e == 0 for e in attr_t):
                if not all(f == 0 for f in attr_q):
                    return False
            if all(f == 0 for f in attr_q):
                if not all(e == 0 for e in attr_t):
                    return False

        #check if there are more available attributes than needed
        manipulations = t_labels - q_labels
        needed = max(np.count_nonzero(manipulations == 1), np.count_nonzero(manipulations == -1))
        remaining = N - needed
        if remaining > 0:
            available_att = count_available_attributes(q_labels, manipulations, attr_num)
            if available_att < remaining:
                return False
                
        return True
    #load attributes
    print('Loading attributes')
    #train set
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
        #validation set
        data = Data(file_root,  img_root_path,
                      transforms.Compose([
                          transforms.Resize((C.TARGET_IMAGE_SIZE, C.TARGET_IMAGE_SIZE)),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                      ]), 'test')
    else:
        print("Argument mode not valid")
    
    features_data = np.load(f'eval_out/feat_{mode}.npy')
    labels = data.label_data
    attr_num = data.attr_num

    print("Finding triplets")

    def find_triples(q_id, labels, N, attr_num, max_manip):
        found_triplets = []
        count = 0
        q_labels = labels[q_id]
        t_shuffled_idx = np.array([i for i in range(len(labels))])
        random.shuffle(t_shuffled_idx)

        for t_id in t_shuffled_idx:
            t_labels = labels[t_id]
            if q_id != t_id and is_couple_match(q_labels, t_labels, N, attr_num):
                manipulations = create_n_manipulations(q_labels, t_labels, N, attr_num)
                if manipulations is not None:
                    found_triplets.append((features_data[q_id], features_data[t_id], manipulations))
                    count += 1
                    if count >= max_manip:
                        break
        return found_triplets

    num_cores = multiprocessing.cpu_count() # @fate perchè non usi gpu? @simo sì la posso usare ma quando l'ho fatto non pensavo fosse necessario
    triplets_unflattened = Parallel(n_jobs=num_cores)(delayed(find_triples)(q_id, labels, N, attr_num, max_manip) for q_id in tqdm(range(len(labels))))
    
    del data, features_data 


    #flatten triplets
    print("processing data...")
    triplets = []
    for triplet_list in triplets_unflattened:
        for triplet in triplet_list:
            triplets.append(triplet) 
    del triplets_unflattened

    random.shuffle(triplets)
    
    q = np.array([triple[0] for triple in triplets])
    t = np.array([triple[1] for triple in triplets])
    manip = np.array([triple[2] for triple in triplets])
    del triplets


    #save triplets
    print("writing triplets")
    hf = h5py.File(f'multi_manip/{mode}/triplets_N_{N}.h5', 'w')
    hf.create_dataset('q', data=q)
    hf.create_dataset('t', data=t)
    hf.create_dataset('manip', data=manip)
    hf.close()

    print("done!")

    

if __name__ == '__main__':
    
    N = 9
    mode = 'train'
    max_manip = 3

    file_root = 'splits/Shopping100k'
    img_root_path = '/Users/simone/Desktop/VMR/Dataset/Shopping100k/Images'

    #dataset for testing purpouses
    #file_root = 'mini_ds/splits/Shopping100k'
    #img_root_path = 'mini_ds/Images'
    
    #multi_manipulation_gen(file_root, img_root_path, N, max_manip, mode)

    
    #to reopen file:
    hf = h5py.File(f'multi_manip/train/triplets_N_{N}.h5', 'r')
    '''
    for line in hf['q']:
        print(line)
        break
    for line in hf['t']:
        print(line)
        break
    for line in hf['manip']:
        print(line)
        break
    '''
    q = hf['q'][0]
    t = hf['t'][0]

    hf.close()


