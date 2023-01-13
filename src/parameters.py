#%%
import os
MODEL_EVAL="01-12-22:44"
NUM_LAYER=3
NUM_EPOCH=100
LR=0.001#0.01 try this
step_decay=10
weight_decay=0.6
VAL_ORIGINAL=False
contin_training=False#!!!#
pretrain_model= "" #"01-12-09:50" #"01-11-18:19"
name_data_set="couples_N_1_amazon.h5"
N = 8
CREATE_ZERO_MANIP_ONLY = True
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
