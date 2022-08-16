#from curses import echo
import os
import argparse
from flask import Flask, render_template, request, jsonify
import h5py
from pyimagesearch.extract_cnn_keras import DenseNet
import numpy as np
import pandas as pd
import os
import matplotlib.image as mpimg
from matplotlib import pyplot as plt


# create flask instance
app = Flask(__name__)

INDEX = os.path.join(os.path.dirname(__file__), 'index.csv')
cwd = os.getcwd() + '/'
    
# main route
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():


    if request.method == "POST":

        RESULTS_ARRAY = []

        # get url
        queryImagePath = request.form.get('img')
        queryImagePath = cwd  + queryImagePath
        print(queryImagePath)

        # 总数据
        ap = argparse.ArgumentParser()

        ap.add_argument("-index", required = False,default='C:/Users/alalala/Desktop/ai/featureCNN.h5',help = "Path to index")
        ap.add_argument("-result", required = False,default='database',help = "Path for output retrieved images")
        args = vars(ap.parse_args())


        # read in indexed images' feature vectors and corresponding image names
        h5f = h5py.File('C:/Users/alalala/Desktop/ai/featureCNN.h5','r')

        # feats = h5f['dataset_1'][:]
        feats = h5f['feats'][:]

        #print(feats)
        imgNames = h5f['names'][:]

        #print(imgNames)
        h5f.close()

        #load feature 
        featureListPath = 'C:/cbir/cbir-project/featureCNN.h5'

        # define feature 
        model = DenseNet()
        queryVec = model.extract_feat(queryImagePath)
        
        #queryVec = model.extract_feat(queryDir)
        scores = np.dot(queryVec, feats.T)
        rank_ID = np.argsort(scores)[::-1]
        rank_score = scores[rank_ID]

        # def url_is_correct(index_t):
        
        #     if index_t >5:
        #         print('超出请求次数！！！')
        #         exit()

        #     try:
        #         url = input('请输入正确的图片路径：')
                
        #         queryDir = url
                
        #         src = mpimg.imread(queryDir)
        #         return queryDir
                    
        #     except:
        #         print('有误的图片路径，请重新输入：')
        #     return url_is_correct(index_t+1)

        # while True:    
        #     print('----------**********-------------')
        #     op = input("退出请输 0，查询请输 enter : ")
        #     if  op == '0':
        #         break
        #     else:
        #         # read and show query image
        #         # 设置多张图片共同显示 
        #         SqureShow = 4
        #         figure,ax=plt.subplots(SqureShow,SqureShow)
        #         global index_t
        #         index_t = 1
        #         queryDir = url_is_correct(index_t)
        #         queryImg = mpimg.imread(queryDir)
        #         x=0
        #         ax[x][x].set_title('QueryImage',fontsize=10, fontname="Times New Roman Bold")
        #         ax[x][x].imshow(queryImg,cmap=plt.cm.gray)
        #         ax[x][x].axis('off') # 显示第一张测试图片
        #         #plt.title("Query Image")
        #         #plt.imshow(queryImg)
        #         #plt.show()

        #         #model = VGGNet()
        #         model = DenseNet()

        #         queryVec = model.extract_feat(queryDir)
        #         #queryVec = model.extract_feat(queryDir)
        #         scores = np.dot(queryVec, feats.T)
        #         rank_ID = np.argsort(scores)[::-1]
        #         rank_score = scores[rank_ID]
        #         #print rank_ID
        #         #print(rank_score)


        #         # number of top retrieved images to show
        #         maxres = SqureShow*SqureShow-1
        #         imlist = [imgNames[index] for i,index in enumerate(rank_ID[0:maxres])]
        #         print("top %d images in order are: " %maxres, imlist)

        #         # # show top #maxres retrieved result one by one
        #         # for i,im in enumerate(imlist):
        #         #     image = mpimg.imread(args["result"]+"/"+str(im, 'utf-8'))
        #         #     plt.title("search output %d" %(i+1))
        #         #     plt.imshow(image)
        #         #     plt.show()

        #         # 显示多张图片

        #         for i,im in enumerate(imlist):
        #             image = mpimg.imread("C:/Users/alalala/Downloads/dataset/-62,-17/" + str(im, 'utf-8'))
        #             print(i, im)
        #             #im_name = str(im).split('/')[1]
        #             #Year = str(im).split('.')[0]
        #             # Issue = str(im_name).split('_')[1]
        #             # Page = int(str(im_name).split('_')[2][-1])+1
        #             # ShowName = Year + '_' + Issue + '_' + str(Page)
        #             #ShowName = 1
        #             #ax[int((i+1)/SqureShow)][(i+1)%SqureShow].set_title('%d:%s(%.3f)' % (i+1,ShowName,rank_score[i]),fontsize=10)
        #             #ax[int(i/maxres)][i%maxres].set_title('Image_name is %s' % im,fontsize=2)
        #             ax[int((i+1)/SqureShow)][(i+1)%SqureShow].imshow(image,cmap=plt.cm.gray)
        #             ax[int((i+1)/SqureShow)][(i+1)%SqureShow].axis('off')
        #         plt.show()

        # try:
        #     # load the query image and describe it

        #     # load feature 
        #     featureListPath = 'C:/cbir/cbir-project/featureCNN.h5'
        #     # define feature 
        #     model = 'densenetModel'
        #     searcher = Searcher(featureListPath,model)
        #     if model == 'densenetModel':
        #         results = searcher.BCNNsearch(queryImagePath)
        #     else:
        #         results = searcher.search(queryImagePath)
            
        #     RESULTS_ARRAY = []
        #     RESULTS_ARRAY = [{"image": str(a), "score": str(b)} for a,b in zip(results['new_path'], results['Mo_part'])]
        #     print(RESULTS_ARRAY)
        #     print(results['Mo_part'].value_counts())
            
        #     classification = ''
        #     for i in range(len(results['Mo_part'].value_counts())):
        #         classification = classification + str(results['Mo_part'].value_counts().index[i]) + ':'
        #         classification = classification + str(results['Mo_part'].value_counts()[i]) + '<br>'
            
        #     return jsonify({'results':(RESULTS_ARRAY[::-1]), 'classification':classification})

        # except:

        #     # return error
        #     return jsonify({"sorry": "Sorry, no results! Please try again."}), 500
        
    return render_template('index.html')
    
# run!
if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)
