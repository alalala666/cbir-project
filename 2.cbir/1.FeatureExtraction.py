#我電腦怪怪的所以貼這幾行 可以註解掉
import os
from tkinter.font import names
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['TF_CPP_MIN_LOG_level'] = '2'


# -*- coding: utf-8 -*-
import h5py
import numpy as np
import argparse
import time
from extract_cnn_keras import DenseNet, VGGNet

#輸入圖片資料夾路徑
path = 'C:/Users/alalala/Downloads/dataset/-62,-17/'
#path = 'C:/Users/alalala/Downloads/dataset/all/'

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("-database", type=str, default=path, help="Path to database which contains images to be indexed")
    # parser.add_argument("-database", type=str, default='./web-server/static/Caltech256', help="Path to database which contains images to be indexed")
    parser.add_argument("-index", type=str, default='featureCNN.h5', help="Name of index file")

    args = parser.parse_args()

    return args

'''
 Returns a list of filenames for all jpg images in a directory. 
'''
def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpeg') | f.endswith('.jpg') | f.endswith('.png')]

'''e
 Extract features and index the images
'''
if __name__ == "__main__":
    opt = parse_opt()
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    db = opt.database
    img_list = get_imlist(db)

    feats = []
    names = []

    #帶入模型
    #model = VGGNet()
    model = DenseNet()
    

    print("--------------------------------------------------")
    print("         feature extraction starts")
    print("--------------------------------------------------")
    start = time.time()

    for i, img_path in enumerate(img_list):
        norm_feat = model.extract_feat(img_path)
        img_name = os.path.split(img_path)[1]
        feats.append(norm_feat)
        names.append(img_name)
        print("extracting feature from image No. %d , %d images in total" %((i+1), len(img_list)))

    feats = np.array(feats)

    # directory for storing extracted features
    output = opt.index
    
    print("--------------------------------------------------")
    print("      writing feature extraction results ...")
    print("--------------------------------------------------")


    h5f = h5py.File(output, 'w')
    h5f.create_dataset('feats', data = feats)
    h5f.create_dataset('names', data = np.string_(names))
    h5f.close()
    end = time.time()
    print("Cost time: ",end-start," (s)")