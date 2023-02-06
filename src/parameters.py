#%%
import os

"""
expirment in 29/01
zero padding in random position,

"""

NUM_LAYER=2
NUM_EPOCH=100
N = 8
K=50 # for evalation ,top @K
LR=0.001#0.01 try this
step_decay=15
weight_decay=0.3

contin_training=False#!!!# Attribute if set True will continuo training from pretrain_model
pretrain_model= "" # Name of folder that contain the best model to presume training

name_data_set="couples_N_1_4_6_8_mixed.h5"
all_data={"couples_N_1_amazon.h5":1,"couples_N_4_small.h5":4,"couples_N_6_small.h5":6,"couples_N_8_small.h5":8,"couples_N_1_4_6_8_mixed.h5":8}
#Manipolation vectors
CREATE_ZERO_MANIP_ONLY = False# sul file 01-13-15:56 era true. 
MOVE_ZERO_MANIP_LAST = False# If true zero manipolation are at the end. useless if CREATE_ZERO_MANIP_ONLY == False
Train_variable_legnth=False # If true will train with variable legnth, CREATE_ZERO_MANIP_ONLY must be setted to TRUE
Eval_variable_legnth=False

VAL_ORIGINAL=True # if set true, will validate on the amazon data set.
MODEL_EVAL= "01-14-04:07" #Folder of model to validate, used in eval.py
EVAL_ALL=False
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR=lambda x,y :os.path.join(y,"multi_manip/{}".format(x))
DATA_IDS=lambda y: os.path.join(y,name_data_set)

DATA_TRAIN_DIR=DATA_DIR("train",ROOT_DIR)
DATA_TEST_DIR=DATA_DIR("test",ROOT_DIR)
DATA_TRAIN=DATA_IDS(DATA_TRAIN_DIR)
if VAL_ORIGINAL:
    DATA_TEST= os.path.join(DATA_TEST_DIR,"couples_N_1_amazon.h5")
else:
    DATA_TEST=DATA_IDS(DATA_TEST_DIR)

FILE_CUT_INDEX=DATA_DIR("cut_index.obj",ROOT_DIR) #used to creat manipolation
FILE_SPLIT_INDEX=DATA_DIR("split_index.obj",ROOT_DIR)#used to create manipolation
LOG_DIR=os.path.join(ROOT_DIR,"log")
FEAT_TRAIN=os.path.join(ROOT_DIR,"eval_out/feat_train.npy")
FEAT_TEST=os.path.join(ROOT_DIR,"eval_out/feat_test.npy")
FEAT_TRAIN_SENZA_N=os.path.join(ROOT_DIR,"eval_out/feat_train_senzaNorm.npy")
FEAT_TEST_SENZA_N=os.path.join(ROOT_DIR,"eval_out/feat_test_senzaNorm.npy")
LABEL_TRAIN=os.path.join(ROOT_DIR,"splits/Shopping100k/labels_train.txt")
LABEL_TEST=os.path.join(ROOT_DIR,"splits/Shopping100k/labels_test.txt")


if __name__=="__main__":
    print(DATA_TRAIN,DATA_TEST)
# %%
