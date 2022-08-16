import os
from tkinter.font import names
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['TF_CPP_MIN_LOG_level'] = '2'


# -*- coding: utf-8 -*-
# Author: AI算法与图像处理

from extract_cnn_keras import VGGNet,DenseNet
from keras.preprocessing import image
from numpy import linalg as LA

import numpy as np
import h5py
from matplotlib import pyplot as plt
#import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse


ap = argparse.ArgumentParser()
#ap.add_argument("-query", required = False, default='TestImages/0.png',
#	help = "Path to query which contains image to be queried")
# ap.add_argument("-index", required = False,default='LACEfeatureCNN.h5',
# 	help = "Path to index")
# ap.add_argument("-result", required = False,default='lace',
# 	help = "Path for output retrieved images")
# 总数据
ap.add_argument("-index", required = False,default='CNNFeature.h5',
	help = "Path to index")
ap.add_argument("-result", required = False,default='database',
	help = "Path for output retrieved images")

args = vars(ap.parse_args())


# read in indexed images' feature vectors and corresponding image names
h5f = h5py.File('featureCNN.h5','r')
# feats = h5f['dataset_1'][:]
feats = h5f['feats'][:]
print(feats)
imgNames = h5f['names'][:]
print(imgNames)
h5f.close()
        
print("--------------------------------------------------")
print("               searching starts")
print("--------------------------------------------------")

def url_is_correct(index_t):
    
    if index_t >5:
        print('超出请求次数！！！')
        exit()

    try:
        url = input('请输入正确的图片路径：')
        
        queryDir = url
        
        src = mpimg.imread(queryDir)
        return queryDir
            
    except:
        print('有误的图片路径，请重新输入：')
    return url_is_correct(index_t+1)


while True:    
    print('----------**********-------------')
    op = input("退出请输 0，查询请输 enter : ")
    if  op == '0':
        break
    else:
        # read and show query image
        # 设置多张图片共同显示
        SqureShow = 4
        figure,ax=plt.subplots(SqureShow,SqureShow)
        global index_t
        index_t = 1
        queryDir = url_is_correct(index_t)
        queryImg = mpimg.imread(queryDir)
        x=0
        ax[x][x].set_title('QueryImage',fontsize=10, fontname="Times New Roman Bold")
        ax[x][x].imshow(queryImg,cmap=plt.cm.gray)
        ax[x][x].axis('off') # 显示第一张测试图片
        #plt.title("Query Image")
        #plt.imshow(queryImg)
        #plt.show()

        #model = VGGNet()
        model = DenseNet()

        queryVec = model.extract_feat(queryDir)
        #queryVec = model.extract_feat(queryDir)
        scores = np.dot(queryVec, feats.T)
        rank_ID = np.argsort(scores)[::-1]
        rank_score = scores[rank_ID]
        #print rank_ID
        #print(rank_score)


        # number of top retrieved images to show
        maxres = SqureShow*SqureShow-1
        imlist = [imgNames[index] for i,index in enumerate(rank_ID[0:maxres])]
        print("top %d images in order are: " %maxres, imlist)

        # # show top #maxres retrieved result one by one
        # for i,im in enumerate(imlist):
        #     image = mpimg.imread(args["result"]+"/"+str(im, 'utf-8'))
        #     plt.title("search output %d" %(i+1))
        #     plt.imshow(image)
        #     plt.show()

        # 显示多张图片

        for i,im in enumerate(imlist):
            #image = mpimg.imread("C:/Users/alalala/Downloads/dataset/-62,-17/" + str(im, 'utf-8'))
            image = mpimg.imread("C:/Users/alalala/Downloads/dataset/all/" + str(im, 'utf-8'))
            print(i, im)
            #im_name = str(im).split('/')[1]
            Year = str(im).split('.')[0]
            # Issue = str(im_name).split('_')[1]
            # Page = int(str(im_name).split('_')[2][-1])+1
            # ShowName = Year + '_' + Issue + '_' + str(Page)
            ShowName = Year
            ax[int((i+1)/SqureShow)][(i+1)%SqureShow].set_title('%d:%s(%.3f)' % (i+1,ShowName,rank_score[i]),fontsize=10)
            #ax[int(i/maxres)][i%maxres].set_title('Image_name is %s' % im,fontsize=2)
            ax[int((i+1)/SqureShow)][(i+1)%SqureShow].imshow(image,cmap=plt.cm.gray)
            ax[int((i+1)/SqureShow)][(i+1)%SqureShow].axis('off')
        plt.show()

    