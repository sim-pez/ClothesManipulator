import numpy as np
import parameters as par
import pickle
import random
from utils import split_labels
from pprint import pprint
import time

with open(par.FILE_SPLIT_INDEX, 'rb') as fp:
    split_index = pickle.load(fp)
    fp.close()
with open(par.FILE_CUT_INDEX, 'rb') as fp:
    cut_index = pickle.load(fp)
    fp.close()


def listify_manip(multi_manip):

    Nsplit=np.split(multi_manip, split_index[:-1])
    manip_array=np.zeros((12,151),dtype=int)
    
    for j in range(manip_array.shape[0]):
        
        start, end=cut_index[j][0], cut_index[j][1]
        manip_array[j][start:end]=Nsplit[j]
    
    return manip_array[~np.all(manip_array == 0, axis=1)]



def create_n_manip(N, q_lbl, t_lbl):

    def addTriangular(manip_list, remaining):
        
        while True:
            mod_idx = random.randint(0, len(manip_list) - 1)
            pos = np.where(manip_list[mod_idx])[0][0]
            if pos != split_index[5] and pos != split_index[5] + 1:  #the only attribute that has size two
                break
        old_manip = manip_list[mod_idx]
        n_old = np.where(old_manip == -1)[0][0]
        p_old = np.where(old_manip == 1)[0][0] 
        for ci in cut_index:
            if ci[0] <= p_old <= ci[1] - 1:
                ci_low = ci[0]
                ci_high = ci[1]
                break
        
        first_manip = np.copy(old_manip)
        first_manip[p_old] = 0
        while True:
            p_first = random.randint(ci_low, ci_high - 1)
            if p_first != n_old and p_first != p_old:
                break
        first_manip[p_first] = 1

        second_manip = np.copy(old_manip)
        second_manip[np.where(second_manip == -1)[0][0]] = 0
        second_manip[n_old] = 0

        n_scnd = p_first
        second_manip[n_scnd] = -1
        
        # assert np.array_equal(old_manip, np.add(first_manip,second_manip))

        manip_list[mod_idx] = second_manip
        first_idx = 0
        for i, manip in enumerate(manip_list[:mod_idx]):
            if manip[n_old] == 1:
                first_idx = i + 1
        manip_list = np.insert(manip_list, random.randint(first_idx, mod_idx), first_manip, 0)

        remaining -= 1

        return manip_list, remaining


    def addForwardBackward(manip_list, remaining, q_lbl):
        
        #initializing
        fw_idx = random.randint(0, len(manip_list) - 1)

        actual_lbl = q_lbl
        for manip in manip_list[:fw_idx]:
            actual_lbl = np.add(actual_lbl, manip)
        
        index_to_change = random.choice(np.where(actual_lbl == 1)[0])
        for ci in cut_index:
            if ci[0] <= index_to_change <= ci[1] - 1:
                ci_low = ci[0]
                ci_high = ci[1]
                break
        
        #create manip_fw
        manip_fw = np.zeros(len(actual_lbl), dtype=int)
        manip_fw[index_to_change] = -1
        candidate_p = list(range(ci_low, ci_high))
        candidate_p.remove(index_to_change)
        p = random.choice(candidate_p)
        manip_fw[p] = 1

        #create manip_bw
        manip_bw = manip_fw * -1

        #insert created manipulations
        manip_list = np.insert(manip_list, fw_idx, manip_fw, 0)

        candidate_bw_idx = []
        for i in range(fw_idx + 1, len(manip_list) + 1):
            candidate_bw_idx.append(i)
            if i < len(manip_list):
                positives = np.where(manip_list[i] == 1)[0]
                if np.any((ci_low <= positives) & (positives < ci_high)):
                     break
        if len(candidate_bw_idx) == 0: 
            candidate_bw_idx.append(fw_idx + 1)
        
        bw_idx = random.choice(candidate_bw_idx)
        manip_list = np.insert(manip_list, bw_idx, manip_bw, 0)

        # assert not np.any( np.add(manip_bw, manip_fw) ) # check if sum is all zero

        remaining -= 2
        return manip_list, remaining

    multi_manip =  np.subtract(t_lbl, q_lbl)

    manip_list = listify_manip(multi_manip)
    original_distance = len(manip_list)
    remaining = N - original_distance

    if remaining > N and not 0 <= original_distance <= 8:
        raise Exception("q and t had not to be selected!")

    sequence = []

    while(True):

        if remaining == 1:
            manip_list, remaining = addTriangular(manip_list, remaining)
        elif remaining >= 2:
            if original_distance == 0:
                addForwardBackward(manip_list, remaining, q_lbl)
            else:
                doTriangular = bool(random.getrandbits(1))
                if doTriangular:
                    manip_list, remaining = addTriangular(manip_list, remaining)
                else:
                    manip_list, remaining = addForwardBackward(manip_list, remaining, q_lbl)
        elif remaining == 0:
            break
        else:
            print(f"distance is {remaining}")
            raise Exception("distance value not accepted")

        assert N - remaining == len(manip_list)

    return manip_list
#TODO generate a matrix with N row instead o vector    