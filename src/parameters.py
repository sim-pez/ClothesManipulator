#%%
import os

MODEL_EVAL= "01-20-11:18" #"01-15-15:05"
NUM_LAYER=2
NUM_EPOCH=100
LR=0.001#0.01 try this
step_decay=6

weight_decay=0.3
VAL_ORIGINAL=True
contin_training=True#!!!#
pretrain_model= "01-17-12:44" #01-13-15:56" #"01-17-12:44" #"01-15-11:38" #"01-13-15:56" #"01-12-09:50" #"01-11-18:19"
name_data_set="couples_N_1_4_6_8_mixed.h5"
all_data={"couples_N_1_amazon.h5":1,"couples_N_4_small.h5":4,"couples_N_6_small.h5":6,"couples_N_8_small.h5":8,"couples_N_1_4_6_8_mixed.h5":8}
N = 8
CREATE_ZERO_MANIP_ONLY = True# sul file 01-13-15:56 era true. 
MOVE_ZERO_MANIP_LAST = True #useless if CREATE_ZERO_MANIP_ONLY == False
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR=lambda x,y :os.path.join(y,"multi_manip/{}".format(x))
DATA_IDS=lambda y: os.path.join(y,name_data_set)

DATA_TRAIN_DIR=DATA_DIR("train",ROOT_DIR)
DATA_TEST_DIR=DATA_DIR("test",ROOT_DIR)
DATA_TRAIN=DATA_IDS(DATA_TRAIN_DIR)
if VAL_ORIGINAL:
    DATA_TEST= os.path.join(DATA_TEST_DIR,"couples_N_6_small.h5")
else:
    DATA_TEST=DATA_IDS(DATA_TEST_DIR)

FILE_CUT_INDEX=DATA_DIR("cut_index.obj",ROOT_DIR)
FILE_SPLIT_INDEX=DATA_DIR("split_index.obj",ROOT_DIR)
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
