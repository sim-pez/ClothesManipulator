import numpy as np
import parameters as par
import pickle
import random
from utils import split_labels
from pprint import pprint
from parameters import CREATE_ZERO_MANIP_ONLY, N


with open(par.FILE_SPLIT_INDEX, 'rb') as fp:
    split_index = pickle.load(fp)
    fp.close()
with open(par.FILE_CUT_INDEX, 'rb') as fp:
    cut_index = pickle.load(fp)
    fp.close()


def findCutIndexInterval(index):
        for ci in cut_index:
            if ci[0] <= index <= ci[1] - 1:
                ci_low = ci[0]
                ci_high = ci[1]
                return ci_low, ci_high


def listify_manip(multi_manip):

    Nsplit=np.split(multi_manip, split_index[:-1])
    manip_array=np.zeros((12,151),dtype=int)
    
    for j in range(manip_array.shape[0]):
        start, end=cut_index[j][0], cut_index[j][1]
        manip_array[j][start:end] = Nsplit[j]

    manip_list = manip_array[~np.all(manip_array == 0, axis=1)] 

    if CREATE_ZERO_MANIP_ONLY:
        for _ in range(len(manip_list), N):
            zero_manip = np.zeros(len(multi_manip), dtype = int)
            manip_list = np.insert(manip_list, 0, zero_manip, 0)
    
    manip_list = manip_list[np.random.permutation(len(manip_list))]

    return manip_list


def create_n_manip(N, q_lbl, t_lbl):
    
    def decomposeOneManip(manip_list, remaining):

        mod_idx_candidates = list(range(len(manip_list)))
        random.shuffle(mod_idx_candidates)


        for candidate_idx in mod_idx_candidates:
            old_manip = manip_list[candidate_idx]
            old_p_list = np.where(old_manip == 1)
            old_n_list = np.where(old_manip == -1)
            if len(old_p_list[0]) == 1 and len(old_n_list[0]) == 1:
                old_p = old_p_list[0][0]
                old_n = old_n_list[0][0]

                #create new manips
                first_manip = np.zeros(len(old_manip), dtype=int)
                first_manip[old_n] = -1
                second_manip = np.zeros(len(old_manip), dtype=int)
                second_manip[old_p] = 1

                #insert new manips
                manip_list[candidate_idx] = first_manip
                ci_low, ci_high = findCutIndexInterval(old_n)
                second_candidate_idx = []
                for i in range(candidate_idx + 1, len(manip_list) + 1):
                    second_candidate_idx.append(i)
                    if i < len(manip_list):
                        if np.any(manip_list[i][ci_low:ci_high]):
                             break
                second_idx = random.choice(second_candidate_idx)
                manip_list = np.insert(manip_list, second_idx, second_manip, 0)

                return manip_list, remaining - 1, True
        
        return manip_list, remaining, False
                

    def addTriangular(manip_list, remaining):

        mod_idx_candidates = list(range(len(manip_list)))
        random.shuffle(mod_idx_candidates)

        for candidate in mod_idx_candidates:
            candidate_manip = manip_list[candidate]
            try:   
                value_idx = np.where(candidate_manip == 1)[0][0]
            except:
                value_idx = np.where(candidate_manip == -1)[0][0]
            if value_idx != split_index[5] and value_idx != split_index[5] + 1:
                mod_idx = candidate
                break
            if candidate == mod_idx_candidates[-1]:
                return manip_list, remaining, False  #available manips are all on size 2 attributes: you need fwbw

        old_manip = manip_list[mod_idx]

        if np.count_nonzero(candidate_manip == 1) == 1 and np.count_nonzero(candidate_manip == -1) == 1: # il classico triangular

            n_old = np.where(old_manip == -1)[0][0]
            p_old = np.where(old_manip == 1)[0][0]
            ci_low, ci_high = findCutIndexInterval(p_old)
                
            first_manip = np.copy(old_manip)
            first_manip[p_old] = 0
            p_first_candidates = list(range(ci_low, ci_high))
            for candidate in p_first_candidates:
                if candidate != n_old and candidate != p_old:
                    p_first = candidate
                    break
                if candidate == p_first_candidates[-1]:
                    return manip_list, remaining, False

            first_manip[p_first] = 1

            second_manip = np.copy(old_manip)
            second_manip[np.where(second_manip == -1)[0][0]] = 0
            second_manip[n_old] = 0

            n_scnd = p_first
            second_manip[n_scnd] = -1

            manip_list[mod_idx] = second_manip
            first_idx = 0
            for i, manip in enumerate(manip_list[:mod_idx]):
                if manip[n_old] == 1:
                    first_idx = i + 1
            manip_list = np.insert(manip_list, random.randint(first_idx, mod_idx), first_manip, 0)


        else:   # new for single value manipulation


            if np.count_nonzero(old_manip == 1) > 0: # old_manip: [0, 0, 0, 1]

                p_old = np.where(old_manip == 1)[0][0]
                ci_low, ci_high = findCutIndexInterval(p_old)

                #create manips
                first_manip = np.zeros(len(old_manip), dtype = int)                
                candidate_indexes = list(range(ci_low, ci_high))
                candidate_indexes.remove(p_old)
                first_manip[random.choice(candidate_indexes)] = 1
                second_manip = np.subtract(old_manip, first_manip)

            else:

                n_old = np.where(old_manip == -1)[0][0]
                ci_low, ci_high = findCutIndexInterval(n_old)
                
                #create manips
                second_manip = np.zeros(len(old_manip), dtype=int)
                candidate_indexes = list(range(ci_low, ci_high))
                candidate_indexes.remove(n_old)
                second_manip[random.choice(candidate_indexes)] = -1
                first_manip = np.subtract(old_manip, second_manip)

            #insert manips
            manip_list[mod_idx] = second_manip
            first_idx = 0
            for i, manip in enumerate(manip_list[:mod_idx]):
                if np.any(manip[ci_low:ci_high]):
                    first_idx = i + 1
            manip_list = np.insert(manip_list, random.randint(first_idx, mod_idx), first_manip, 0)

        return manip_list, remaining - 1, True


    def addForwardBackward(manip_list, remaining, q_lbl):
        
        #initializing
        fw_idx = random.randint(0, len(manip_list) - 1)

        actual_lbl = q_lbl
        for manip in manip_list[:fw_idx]:
            actual_lbl = np.add(actual_lbl, manip)
        
        index_to_change = random.choice(np.where(actual_lbl == 1)[0])
        ci_low, ci_high = findCutIndexInterval(index_to_change)
        
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
                if np.any(manip_list[i][ci_low:ci_high]):
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

    orig_manip_list = np.copy(manip_list)
    original_distance = len(manip_list)


    remaining = N - original_distance

    if remaining > N and not 0 <= original_distance <= 8:
        raise Exception("q and t had not to be selected!")


    while(True):
        if remaining == 1:
            manip_list, remaining, success = decomposeOneManip(manip_list, remaining)
            if not success:
                manip_list, remaining, success = addTriangular(manip_list, remaining)
                if not success:
                    manip_list = np.copy(orig_manip_list)
                    remaining = N - original_distance
                    second_loop = True
        elif remaining >= 2:
            if original_distance == 0:
                manip_list, remaining = addForwardBackward(manip_list, remaining, q_lbl)
            else:
                doTriangular = bool(random.getrandbits(1))
                if doTriangular:
                    manip_list, remaining, success = addTriangular(manip_list, remaining)
                    if not success:
                        manip_list, remaining = addForwardBackward(manip_list, remaining, q_lbl)
                else:
                    manip_list, remaining = addForwardBackward(manip_list, remaining, q_lbl)
        elif remaining == 0:
            break
        else:
            print(f"distance is {remaining}")
            raise Exception("distance value not accepted")

        assert N - remaining == len(manip_list)

    return manip_list