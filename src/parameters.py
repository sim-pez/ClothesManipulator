import os

NUM_LAYER = 2
NUM_EPOCH = 100
N = 8
K = 50                # top@k retrieval for evaluation
LR = 0.001            # learning rate
step_decay = 15
weight_decay = 0.3

contin_training = False   # If set True will continue training from pretrained model
pretrain_model = ""       # Name of folder that contains the model resume training

name_data_set = "couples_N_1_4_6_8_mixed.h5" # dataset name
all_data = {"couples_N_1_amazon.h5":1,"couples_N_4_small.h5":4,"couples_N_6_small.h5":6,"couples_N_8_small.h5":8,"couples_N_1_4_6_8_mixed.h5":8}

#Manipulation vectors
CREATE_ZERO_MANIP_ONLY = False  # do zero padding. If false, creates padding with random manips
MOVE_ZERO_MANIP_LAST = False    # do zero padding only at the end. Useless if CREATE_ZERO_MANIP_ONLY == False
Train_variable_legnth = False   # train with variable legnth sequences. CREATE_ZERO_MANIP_ONLY has to be setted to TRUE
Eval_variable_legnth = False    # same as before for evaluation

VAL_ORIGINAL=True           # if set true, will validate on the amazon data set.
MODEL_EVAL= "01-14-04:07"   # model name to evaluate, used in eval.py
EVAL_ALL=False              # evaluate all models

#Other configs
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
