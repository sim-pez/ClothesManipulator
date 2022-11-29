import numpy as np
import parameters as par
from pprint import pprint

def get_attr_num():
    f = open("./splits/Shopping100k/attr_num.txt")
    return [int(line) for line in f.readlines()]

attr_num = get_attr_num() 


def split_manip(manip):
    manip_list = [np.zeros(len(manip), dtype='int') for _ in attr_num]
    offset = 0
    for i, attr_len in enumerate(attr_num):
        for j in range(attr_len):
            manip_list[i][j + offset] = manip[j + offset]
        offset += attr_len
    return manip_list
    

if __name__ == '__main__':

    manip = [int(i) for i in range(151)]

    a = split_manip(manip)

    pprint(a)
