import numpy as np
import parameters as par
import pickle
import random
from parameters import CREATE_ZERO_MANIP_ONLY, MOVE_ZERO_MANIP_LAST
import faiss
from tqdm import tqdm
from utils import split_labels,  compute_NDCG, get_target_attr
import warnings
import torch
warnings.simplefilter('once', RuntimeWarning)


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


def listify_manip(multi_manip, N):

    Nsplit=np.split(multi_manip, split_index[:-1])
    manip_array=np.zeros((12,151),dtype=int)
    
    for j in range(manip_array.shape[0]):
        start, end=cut_index[j][0], cut_index[j][1]
        manip_array[j][start:end] = Nsplit[j]

    manip_list = manip_array[~np.all(manip_array == 0, axis=1)] 

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
                return manip_list, remaining, False

        old_manip = manip_list[mod_idx]

        if np.count_nonzero(candidate_manip == 1) == 1 and np.count_nonzero(candidate_manip == -1) == 1:

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


        else:  
            if np.count_nonzero(old_manip == 1) > 0:

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

        remaining -= 2
        return manip_list, remaining


    multi_manip =  np.subtract(t_lbl, q_lbl)
    manip_list = listify_manip(multi_manip, N)
    original_distance = len(manip_list)
    orig_manip_list = np.copy(manip_list)
    

    if CREATE_ZERO_MANIP_ONLY and (N - len(manip_list)) > 0:
        zero_manips = np.zeros((N - len(manip_list),len(multi_manip)), dtype = int)
        manip_list = np.concatenate((manip_list, zero_manips))
        assert N == len(manip_list)
        
        if not MOVE_ZERO_MANIP_LAST:
            np.random.shuffle(manip_list)


    remaining = N - len(manip_list)

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
            warnings.warn("There are couple wich distance is > N!", RuntimeWarning)
            del_idx = list(range(len(manip_list)))
            random.shuffle(del_idx)
            manip_list = np.delete(manip_list, del_idx[:(-remaining)],0)
            return manip_list, original_distance

        assert N - remaining == len(manip_list)

    return manip_list, original_distance


def calc_accuracy(database,queries,query_labels,test_labels,k,step,dim = 4080):
    num_database = database.shape[0]
    num_query = queries.shape[0]
    print("Step:{s} ,Num of image in database is {n1}, Num of query img is {n2}".format(s=step,n1=num_database,n2=num_query))

    index = faiss.IndexFlatL2(dim)
    index.add(database)
    _, knn = index.search(queries, k)
    assert (query_labels.shape[0] == queries.shape[0])
    #TODO essendo l'immagine target è stata usata più volte nel data set,
    #  ogni volta partendo da un'immagine di query diversa,
    #sarebbe utile confrontare quanto le stima dei feat_target che corrispondono
    #  allo stesso target sono simili tra loro
    #si poù confrontare anche a livello di attributi
    #compute top@k acc
    hits = 0
    tq=tqdm(range(num_query))
    for q in tq: # itera i dati predicted_tfeat
        neighbours_idxs = knn[q]# gli indici dei k-feat più simili alla predicted_tfeat[q]
        for n_idx in neighbours_idxs:
            if (test_labels[n_idx] == query_labels[q]).all():
                hits += 1
                break
        tq.set_description("Num of hit {h}".format(h=hits))
    acc=(hits/num_query)*100 
    result_string='Top@{k} accuracy: {acc} ,Total hits{h}'.format(k=k, acc=acc , h=hits)
    print(result_string)
    """ 
    #compute NDCG
    ndcg = []
    # ndcg_target = []  # consider changed attribute only
    #ndcg_others = []  # consider other attributes

    for q in tqdm(range(num_query)):
        rel_scores = []
    # target_scores = []
        #others_scores = []
        neighbours_idxs = knn[q]
    # indicator = query_inds[q]
        #target_attr = get_target_attr(indicator, gallery_data.attr_num)
        attr_num=np.loadtxt(os.path.join(par.ROOT_DIR, "splits/Shopping100k/attr_num.txt") ,dtype=int)
        target_label = split_labels(query_labels[q],attr_num)
        for n_idx in neighbours_idxs:
            n_label = split_labels(test_labels[n_idx], attr_num)
            # compute matched_labels number
            match_cnt = 0
            others_cnt = 0
            for i in range(len(n_label)):
                if (n_label[i] == target_label[i]).all():
                    match_cnt += 1
            rel_scores.append(match_cnt / len(attr_num))

        ndcg.append(compute_NDCG(np.array(rel_scores)))
    print    
    print('NDCG@{k}: {ndcg}'.format(k=k, ndcg=np.mean(ndcg)))
    """
    return acc,result_string


def eval_help(model,data_loader):
    model.eval()
    predicted_tfeat = []
    with torch.no_grad():
        tq=tqdm(data_loader)
        for i, sample in enumerate(tq):
            qFeat,label_t,mani_vects,legnths = sample
            out,hidden = model(mani_vects,qFeat)
            predicted_tfeat.append(out.cpu().numpy())
    predicted_tfeat= np.concatenate(predicted_tfeat, axis=0)
    return predicted_tfeat
def get_variable_legnth(manips_vec):
    #Tested and work for N=8, 
    manips_vec=manips_vec.numpy()
    f=lambda x: x[~np.all(x== 0, axis=1)]
    list_manips=[]
    id_x=[]
    for i in range(par.N):
        list_manips.append([])
        id_x.append([])

    #list_manips=[[],[],[],[],[],[],[],[]]
    #id_x=[[],[],[],[],[],[],[],[]]
    for i in range(len(manips_vec)):
        l=f(manips_vec[i])
        list_manips [len(l)-1].append(l)
        id_x[len(l)-1].append(i)
    return list_manips,id_x

def eval_variable_help(model,data_loader):
    predicted_tfeat = []
    with torch.no_grad():
        tq=tqdm(data_loader)
        for i, sample in enumerate(tq):
            qFeat,label_t,mani_vects,legnths = sample
            out_batch=[]
            idx_n=[]
            list_manips,id_x=get_variable_legnth(mani_vects)
            for n in range(len(list_manips)):
                if(len(list_manips[n])>0):
                    list_manips_n=torch.tensor(list_manips[n])
                    qFeat_n=qFeat[id_x[n]]
                    out,hidden = model(list_manips_n,qFeat_n)
                    idx_n.append(id_x[n])
                    out_batch.append(out)
            idx_n=torch.cat(idx_n,axis=0)
            out_batch=torch.cat(out_batch, axis=0)
            orderd_out_batch=torch.zeros(out_batch.shape)
            for i in range(len(idx_n)):
                orderd_out_batch[id_x[i]]=out_batch[i] # es: idx[1]=17 corrispond to out batch[1] then i want map out_batch[1] to orderd_out_batch[17]
            predicted_tfeat.append(orderd_out_batch.cpu().numpy())
    predicted_tfeat= np.concatenate(predicted_tfeat, axis=0)
    
            
    return predicted_tfeat
