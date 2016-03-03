import numpy as np


import hickle as hkl


X = np.zeros((n_examples, nt, nx, nx)).astype(np.float32)

for i in range(n_examples):

    # randomly choose params
    # create_moving_line with params

    params_dict = {'line_len': np.zeros(n_examples), }


X[4] #

X[1,1,1,1]
X[1] # 3D array nt x nx x nx

np.array()

[][][][]


def create_moving_line(nt, line_len, nx, x0, y0, speed):

    X = np.zeros((nt, nx, nx)).astype(np.float32)

    for i in range(nt):

        xt = x0+i*speed

        X[i,y0:y0+line_lin,xt] = 1

    file_name = 'line.hkl'
    hkl.dump(X, open(file_name, 'w'))
    X = hkl.load(open(file_name))
