#%%
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR=lambda x,y :os.path.join(y,"multi_manip/{}".format(x))
DATA_IDS=lambda y: os.path.join(y,"triplets_N_8.h5")

DATA_TRAIN_DIR=DATA_DIR("train",ROOT_DIR)
DATA_TEST_DIR=DATA_DIR("test",ROOT_DIR)
DATA_TRAIN=DATA_IDS(DATA_TRAIN_DIR)
DATA_TEST=DATA_IDS(DATA_TEST_DIR)

if __name__=="__main__":
    print(DATA_TRAIN,DATA_TEST)
# %%
