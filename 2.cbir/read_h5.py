import h5py

f= h5py.File('featureCNN.h5','r')

for key in f.keys():
    # print(f[key].name)
    # print(f[key].shape)
    print(key)

import tensorflow as tf
print(tf.__version__)