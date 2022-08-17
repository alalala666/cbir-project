import h5py

f= h5py.File('C:\cbir\cbir-project\/featureCNN.h5','r')

for key in f.keys():
    print(f[key].name)
    print(f[key].shape)
    print(key)
    
# import os
# from tkinter.font import names
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# os.environ['TF_CPP_MIN_LOG_level'] = '2'

import tensorflow
print(tensorflow.__version__)