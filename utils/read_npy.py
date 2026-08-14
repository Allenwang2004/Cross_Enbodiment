# read npy file

import numpy as np

def read_npy(file_path):
    return np.load(file_path)   

if __name__ == "__main__":
    file_path = "/home/allen19/crossenbodiment/data/retargeting_z/crawl-0.4-0-d/crawl-0.4-0-d_0.npy"
    data = read_npy(file_path)
    print(data.shape)