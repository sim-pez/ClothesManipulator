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

from utils import split_labels

# def find_couples() TODO


def create_n_manipulations(data, t_id, q_id, N):
    '''
    generate a multi-manipulation of N manipulations
    '''

    def find_available_attribute(manipulations_splitted, q_labels_splitted, attr_num):

        attributes = []
        for i, m in enumerate(manipulations_splitted):
            if all(v == 0 for v in m):
                if 1 in q_labels_splitted[i]:
                    attributes.append(i)

        return attributes

    def flatten(l):
        return [item for sublist in l for item in sublist]

    def random_manipulate(manipulations_splitted, attr, q_labels_splitted):

        negative_idx = np.where(q_labels_splitted[attr] == 1)[0][0]
        manipulations_splitted[attr][negative_idx] = -1

        #not tested yet
        zero_indexes = [i for i, x in enumerate(manipulations_splitted[attr]) if x!=-1]
        positive_idx = random.choice(zero_indexes)
        manipulations_splitted[attr][positive_idx] = 1

        return manipulations_splitted

    def casual_remove_manipulations(manipulations_splitted, r):

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
        print("Warning: need " + str(needed) + " manipulations but N is " + str(N) + ". Removing some manipulations")
        print("(Q : " + str(q_id) + "    T : " + str(t_id) + ")")
        manipulations_splitted = casual_remove_manipulations(manipulations_splitted, - remaining)

    #extract N - needed random manipulations
    elif remaining > 0:
        for _ in range(remaining):
            candidate_attributes = find_available_attribute(manipulations_splitted, q_labels_splitted, attr_num)
            attribute = random.choice(candidate_attributes)                                   
            manipulations_splitted = random_manipulate(manipulations_splitted, attribute, q_labels_splitted)

    manipulations = flatten(manipulations_splitted)

    return manipulations

   
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

    N = 8
    t_id = 2
    q_id = 3
    manipulations = create_n_manipulations(train_data, t_id, q_id, N)