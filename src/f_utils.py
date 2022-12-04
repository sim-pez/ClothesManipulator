import numpy as np
import parameters as par
attr_num = np.loadtxt(r"/home/falhamdoosh/disentagledFeaturesExtractor/splits/Shopping100k/attr_num.txt",dtype=int)
cut_index=[]
start=0
for i ,num_attr in enumerate(attr_num):
    cut_index.append((start, start+num_attr))
    start+=num_attr
split_index=[inx[1] for i,inx in enumerate (cut_index)]

import pickle
with open(par.FILE_SPLIT_INDEX, 'wb') as fp:
        pickle.dump(split_index, fp)
        fp.close
with open(par.FILE_SPLIT_INDEX, 'rb') as fp:
    object_file = pickle.load(fp)
    fp.close()
print(object_file)