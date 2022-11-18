from cProfile import label
from operator import truediv
from tqdm.auto import tqdm
import constants as C
import random
import torchvision.transforms as transforms
import numpy as np
from dataloader import Data
from pprint import pprint

from utils import split_labels, flatten_labels

def multi_manipulation_gen(file_root, img_root_path, N):
    '''
    Output files:
    couples.txt  ->  couples of Q and T that have the same non-zero attribute index (problably it is only for internal use)
    selected_couples_for_manip.txt  ->  couples of couples.txt which len(non-zero differente attributes) > N
    manipulations.txt  ->  generates manipulation vector from couples of selected_couples_for_manip.txt
    '''

    def find_couples(data):
        '''
        return couples of q,t that are compatible, i.e. that have the same all-zero attributes
        '''

        def check_match(q,t):

            for attr_q, attr_t in zip(q,t):
                if all(e == 0 for e in attr_t):
                    if not all(f == 0 for f in attr_q):
                        return False
                if all(f == 0 for f in attr_q):
                    if not all(e == 0 for e in attr_t):
                        return False

            return True


        attr_num = data.attr_num
        labels = data.label_data

        splitted_labels = [split_labels(l, attr_num) for l in labels]
        couples = [(i,j) for i, q in enumerate(tqdm(splitted_labels))
                            for j, t in enumerate(splitted_labels)
                                if i != j
                                    if check_match(q,t)]


        return couples


    def create_n_manipulations(data, t_id, q_id, N):
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

        #initialization
        q_labels = data.label_data[q_id]
        t_labels = data.label_data[t_id]
        attr_num = data.attr_num
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
    

    '''
    dataset for testing purpouses

    #file_root = 'mini_ds/splits/Shopping100k'
    #img_root_path = 'mini_ds/Images'
    
    '''

    print('Loading attributes')
    train_data = Data(file_root,  img_root_path, 
                      transforms.Compose([
                          transforms.Resize((C.TRAIN_INIT_IMAGE_SIZE, C.TRAIN_INIT_IMAGE_SIZE)),
                          transforms.RandomHorizontalFlip(),
                          transforms.CenterCrop(C.TARGET_IMAGE_SIZE),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                      ]), 'train')
    

    #find couples
    print("Finding couples of Q and T")
    couples = find_couples(train_data)
    random.shuffle(couples)

    #write couples
    print("Writing couples")
    f = open("multi_manip/couples_N_{N}.txt", "w")
    for c in tqdm(couples):
        f.write(str(c[0]) + " " + str(c[1]) + "\n")
    f.close()

    #find and write manipulations
    print("Finding and writing manipulations")
    f = open(f"multi_manip/couples_N_{N}.txt", "r")
    g = open(f"multi_manip/selected_couples_for_manip_N_{N}.txt", "w")
    e = open(f"multi_manip/manipulations_N_{N}.txt", "w")
    for line in tqdm(f):
        cpl = (list(line.strip().split(" ")))
        manipulations = create_n_manipulations(train_data, int(cpl[0]), int(cpl[1]), N)
        if manipulations is not None:
            manipulations_str = [str(m) for m in manipulations]
            g.write(str(cpl[0]) + " " + str(cpl[1]) + "\n")
            e.write(' '.join(manipulations_str) + "\n")
    
    f.close()
    e.close()
    g.close()


   
if __name__ == '__main__':
    
    N = 9
    file_root = 'splits/Shopping100k'
    img_root_path = '/Users/simone/Desktop/VMR/Dataset/Shopping100k/Images'
    multi_manipulation_gen(file_root, img_root_path, N)
