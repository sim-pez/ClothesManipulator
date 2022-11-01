from cProfile import label
from operator import truediv
import constants as C
import random
import torchvision.transforms as transforms
import numpy as np
from argument_parser import add_base_args, add_train_args
from dataloader import Data
from loss_function import hash_labels, TripletSemiHardLoss
from pprint import pprint


#   ---------------------------------------------------    generate multimanipulations   --------------------------------------------------


def generate_multimanipulations(data, q_id, t_id, N):

    def flatten(l):
        return [item for sublist in l for item in sublist]

    def sublistify_by_attributes(list, attr_num):
        offset = 0
        sublisted = []
        for domain in attr_num:
            sublisted.append(list[offset: offset + domain].tolist())
            offset += domain
        return sublisted


        ordered_manip = [ [] for _ in range(len(sublisted_manipulations)) ]

        shuffled_index = (list(range(len(sublisted_manipulations))))
        random.shuffle(shuffled_index)

        count = 0
        
        for i in shuffled_index:
            found_non_zero = False
            for e in sublisted_manipulations[i]:
                if e == 0:
                    ordered_manip[i].append(0)
                elif e == 1:
                    ordered_manip[i].append(count)
                    found_non_zero = True
                elif e == -1:
                    ordered_manip[i].append(- count)
                    found_non_zero = True
            if found_non_zero:
                count += 1
        return ordered_manip

    def split_manipulations(manipulations, attr_num):

        manipulations = sublistify_by_attributes(manipulations, attr_num)

        manipulated_attributes_idx = []
        for i, manip in enumerate(manipulations):
            if 1 in manip:
                manipulated_attributes_idx.append(i)

        random.shuffle(manipulated_attributes_idx)

        splitted_manipulations = []

        for attribute_idx in manipulated_attributes_idx:
            attribute_man = []
            for i, m in enumerate(manipulations):
                if i == attribute_idx:
                    attribute_man.append(m)
                else:
                    attribute_man.append([0] * len(m))
            attribute_man = flatten(attribute_man)
            splitted_manipulations.append(np.array(attribute_man))

        return splitted_manipulations
            
    def find_negative_idx(chosen_idx, attr_num, labels):
        offset = 0
        for domain in attr_num:
            if offset <= chosen_idx < offset + domain:
                lbl = [labels[i] for i in range(offset, offset + domain)]
                negative_idx = np.where(labels[offset : offset + domain] == 1)
                if np.any(negative_idx):
                    return offset + negative_idx[0][0]
                else:
                    return None
            offset += domain
        raise Exception("Error: index to be changed not found")

    def specular(manipulation):
        specular_manip = np.zeros(len(manipulation), dtype = int)
        for i, m in enumerate(manipulation):
            if m != 0:
                specular_manip[i] = -m
        return specular_manip


    #initialization
    q_labels = data.label_data[q_id]
    t_labels = data.label_data[t_id]
    attr_num = data.attr_num


    #get needed manipulation to go from q to t
    manipulations = t_labels - q_labels
    manipulations_list = split_manipulations(manipulations, attr_num)
    needed = np.count_nonzero(manipulations == 1)
    remaining = N - needed
    if remaining <= 0:
        raise Exception("Error: need " + str(needed) + " manipulations but N is too low!")


    #creates extra random manipulations: they will be couples of specular manipulations
    remaining_to_randomize = int(remaining / 2) 
    actual_labels = t_labels
    for _ in range(remaining_to_randomize):

        available_indexes = [i for i,x in enumerate(actual_labels) if x == 0]
        chosen_idx = random.choice(available_indexes)
        negative_idx = find_negative_idx(chosen_idx, attr_num, actual_labels)

        new_manipulation = np.zeros(len(t_labels), dtype=int)
        new_manipulation[chosen_idx] = 1
        if negative_idx is not None:
            new_manipulation[negative_idx] = -1
        
        manipulations_list.append(new_manipulation)
        manipulations_list.append(specular(new_manipulation))

        actual_labels = np.add(actual_labels, new_manipulation)

    #if one manipulation missing we add a manipulation array of zero
    if len(manipulations_list) == N - 1:
        manipulations_list.append(np.zeros(len(t_labels), dtype = int))
        
    return manipulations_list


#   ---------------------------------------------------    n transform    --------------------------------------------------

def get_random_manipulation(q_labels, attr_num):
    '''
        gets label of q and outputs a compatible manipulation

        Nota: è veramente a caso, non si torna indietro
    '''
        
    zero_indexes = [i for i, label in enumerate(q_labels) if label == 0]
    positive_index = random.choice(zero_indexes)	
    
    offset = 0
    for domain in attr_num:
        if offset <= positive_index < domain + offset:
            possible_choices = [i for i in range(offset, domain + offset) if i != positive_index]
            negative_index = random.choice(possible_choices)
            break
        offset += domain

    manipulation = np.zeros(len(q_labels), dtype=int)
    manipulation[positive_index] = 1
    manipulation[negative_index] = -1
    return manipulation


def n_transform(data, t_id, q_id, N):

    q_labels = data.label_data[q_id]
    t_labels = data.label_data[t_id]


    for i in range(N):
        ramaining = N - i
        
        manipulations = t_labels - q_labels
        needed = max(np.count_nonzero(manipulations == 1), np.count_nonzero(manipulations == -1))

        if needed < ramaining:
            # Q* = modifica(Q*, trasformazioni_necessarie[0]) TODO
            pass
        else:
            random_manipulation = get_random_manipulation(q_labels, train_data.attr_num)
            # Q* = modifica(Q*, get_random_manipulation() ) TODO
   

#    ---------------------------------------------------    main    --------------------------------------------------


#usage example
if __name__ == '__main__':
    
    file_root = 'mini_ds/splits/Shopping100k'
    img_root_path = 'mini_ds/Images'
    #file_root = 'splits/Shopping100k'
    #img_root_path = '/Users/simone/Desktop/VMR/Dataset/Shopping100k/Images'

    print('Loading attributes')
    train_data = Data(file_root,  img_root_path, 
                      transforms.Compose([
                          transforms.Resize((C.TRAIN_INIT_IMAGE_SIZE, C.TRAIN_INIT_IMAGE_SIZE)),
                          transforms.RandomHorizontalFlip(),
                          transforms.CenterCrop(C.TARGET_IMAGE_SIZE),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                      ]), 'train')

    '''
    probably useless code:
    #train_loader = data.DataLoader(train_data, shuffle=True, drop_last=True)
    #valid_loader = data.DataLoader(valid_data, shuffle=False, drop_last=False)

    useful attributes:
    train_data.label_data, train_data.attr_num
    '''

    N = 11
    t_id = 0
    q_id = 1
    n_transform(train_data, t_id, q_id, N)
    multi_manipulations = generate_multimanipulations(train_data, q_id, t_id, N)

