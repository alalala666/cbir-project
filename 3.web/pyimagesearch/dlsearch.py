import torch
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models

import numpy as np
import pandas as pd
import os
from PIL import Image
import math
from pyimagesearch import mymodels
import warnings
warnings.filterwarnings('ignore')

CUDA = torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")
M_relabel_map = {'MR': 0,
 'CT': 1,
 'PT': 2,
 'OCT': 3,
 'US': 4,
 'CFP': 5,
 'Elastography': 6,
 'Endoscopy': 7,
 'Dermoscopy': 8,
 'Pathology': 9,
 'MG': 10,
 'Thermography': 11,
 'Doppler': 12,
 'PET-CT': 13}
M_relabel_map = {v:k for k,v in M_relabel_map.items()}

relabel_map = {0: 'PT ALL all',
 1: 'MRI BRAIN GBM',
 2: 'MRI PROSTATE adenocarcinoma',
 3: 'OCT EYES CNV',
 4: 'OCT EYES NORMAL',
 5: 'CT BREAST cancer',
 6: 'CFP Ocular all',
 7: 'US THYROID all',
 8: 'CT LIVER hepatocellular carcinoma',
 9: 'CT STOMACH adenocarcinoma',
 10: 'OCT EYES DME',
 11: 'CT UTERUS corpus endometrial carcinoma',
 12: 'US NERVES all',
 13: 'CT PANCREAS ductal adenocarcinoma',
 14: 'MRI LIVER hepatocellular carcinoma',
 15: 'CT OVARY cancer',
 16: 'OCT EYES DRUSEN',
 17: 'CT LUNG squamous cell carcinoma',
 18: 'US LUNG all',
 19: 'MRI BREAST cancer',
 20: 'US CARDIAC MI',
 21: 'MRI CERVIX cervical squamous cell carcinoma and endocervical adenocarcinoma',
 22: 'US PROSTATE adenocarcinoma',
 23: 'Elastography BREAST all',
 24: 'CT BLADDER urothelial bladder carcinoma',
 25: 'CT BRAIN GBM',
 26: 'CT COLON adenocarcinoma',
 27: 'CT PROSTATE adenocarcinoma',
 28: 'MRI KIDNEY renal clear cell carcinoma',
 29: 'CT KIDNEY renal clear cell carcinoma',
 30: 'CT THYROID cancer',
 31: 'Dermoscopy SKIN nevus',
 32: 'MRI RECTUM adenocarcinoma',
 33: 'Pathology BONE all',
 34: 'MG BREAST all',
 35: 'Thermography BREAST cancer',
 36: 'US BREAST all',
 37: 'MRI CARDIAC all',
 38: 'Dermoscopy SKIN melanoma',
 39: 'Doppler NECK all',
 40: 'Endoscopy GI tract  dyed-lifted-polyps',
 41: 'Endoscopy GI tract  dyed-resection-margins',
 42: 'Endoscopy GI tract  esophagitis',
 43: 'Endoscopy GI tract  normal-cecum',
 44: 'Endoscopy GI tract  normal-pylorus',
 45: 'Endoscopy GI tract  normal-z-line',
 46: 'Endoscopy GI tract  polyps',
 47: 'Endoscopy GI tract  ulcerative-colitis',
 48: 'Dermoscopy SKIN seborrheic_keratosis',
 49: 'PET-CT BREAST cancer',
 50: 'CT RECTUM adenocarcinoma',
 51: 'Pathology BLOOD all'}


########


class Searcher:
    def __init__(self, dfPath, model):
        # store our index path
        self.dfPath = dfPath
        num_classes = 52
        

        if model == 'alexnet':
            model = mymodels.alexnetModel(num_classes)
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif model == 'resnet152':
            model = mymodels.resnetModel(num_classes)
            if CUDA:
                model.load_state_dict(torch.load('pyimagesearch/resnet152_pixel_2.pth'))
            else:
                model.load_state_dict(torch.load('pyimagesearch/resnet152_pixel_2.pth',map_location='cpu'))
            #except Exception as e:
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif model == 'inception_v3':
            model = mymodels.inceptionModel(num_classes)
            
            transform = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif model == 'densenet201':
            model = mymodels.densenetModel(num_classes)
            if CUDA:
                model.load_state_dict(torch.load('pyimagesearch/.pth'))
            else:
                model.load_state_dict(torch.load('pyimagesearch/.pth',map_location='cpu'))
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif model == 'BCNN':
            model = mymodels.BCNN()
            
            if CUDA:
                model.load_state_dict(torch.load('C:/Users/00/Downloads/CBMIR/BCNN_pixel_5.pth'))
            else:
                model.load_state_dict(torch.load('C:/Users/00/Downloads/CBMIR/BCNN_pixel_5.pth',map_location='cpu'))
                
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        
        if CUDA:
            model = model.cuda()
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        self.model = model
        self.transform = transform
        
        
    
    def deduplicate(self, de_df):
        
        PT = de_df['Modality']=='PT'
        CT = (de_df['Modality']=='CT') & (de_df['part']!='RECTUM')
        MR = (de_df['Modality']=='MR') & (de_df['part']!='CARDIAC')
        PTCTMR = de_df[(PT | CT | MR)]
        MR_CARDIAC = de_df[(de_df['Modality']=='MR') & (de_df['part']=='CARDIAC')]
        
        de_df.drop(PTCTMR.index, inplace = True)
        de_df.drop(MR_CARDIAC.index, inplace = True)
        
        PTCTMR['series'] = PTCTMR['new_path'].str.rsplit('\\',n = 1).str[0]
        PTCTMR = PTCTMR.drop_duplicates('series')
        
        MR_CARDIAC['series'] = MR_CARDIAC['new_path'].str.rsplit('_',n = 1).str[0]
        MR_CARDIAC = MR_CARDIAC.drop_duplicates('series')
        PTCTMR.drop('series', axis = 1, inplace = True)
        MR_CARDIAC.drop('series', axis = 1, inplace = True)
        
        de_df = pd.concat([de_df, PTCTMR, MR_CARDIAC])
        return de_df
    
	
    def search(self, queryImagePath, limit = 10):
        all_feature = pd.read_csv(self.dfPath,
                                  dtype={'Modality':str, 'part':str},
                                  chunksize = 5000,
                                  iterator = True)
        with torch.no_grad():
            img = Image.open(queryImagePath).convert('RGB')
            img = self.transform(img)
            img = torch.unsqueeze(img, dim=0).to(device)
            img_feature =  np.array(self.model.get_feature(img).tolist())
        retrieval_list = pd.DataFrame(columns = ['Modality','part','new_path','Mo_part','Euclidean'])

        all_feature = pd.read_csv('C:/Users/00/Downloads/CBMIR/.csv', 
                                dtype={'Modality':str, 'part':str},
                                chunksize = 60000,
                                iterator = True)

        for feature_list in all_feature:
            feature_list['Cosine'] = feature_list.iloc[:,4:-1].dot(img_feature)/(np.linalg.norm(feature_list.iloc[:,4:-1], axis=1) * np.linalg.norm(img_feature))
            part_list = feature_list.nlargest(20,'Cosine')
            f = feature_list.drop(part_list.index)
            part_list = self.deduplicate(part_list)
            while len(part_list) < 10:
                p = f.nlargest(20-len(part_list),'Cosine')
                part_list = pd.concat([part_list, p])
                f.drop(p.index,inplace=True)
                part_list = self.deduplicate(part_list)
            retrieval_list = pd.concat([retrieval_list, 
                                        part_list.loc[:,['Modality','part','new_path','Mo_part','Cosine']]],
                                        ignore_index=True)

        retrieval_list['Cosine'] = retrieval_list['Cosine'].astype('float')
        retrieval_list = retrieval_list.nsmallest(limit,'Euclidean')
        retrieval_list['Mo_part'] = retrieval_list['Modality']+' '+retrieval_list['part']
        #retrieval_list['new_path'] = 'static/minidataset/'+retrieval_list['new_path'].str[42:].replace('\\','+')
        #retrieval_list['new_path'] = retrieval_list['new_path'].str.replace('\\','+')
        print(retrieval_list['new_path'][0])
        retrieval_list.to_csv('success.csv')
        
        return retrieval_list


    def BCNNsearch(self, queryImagePath, limit = 10):
        r_list = pd.DataFrame(columns = ['Modality','part','new_path','Mo_part','Cosine'])
        with torch.no_grad():
            img = Image.open(queryImagePath).convert('RGB')
            img =  self.transform(img)
            img = torch.unsqueeze(img, dim=0).to(device)
            img_f1, img_f2, img_f3 = self.model.get_feature(img)#.tolist()
            
            img_f1 = img_f1.tolist()[0]
            img_f2 = img_f2.tolist()[0]
            img_f3 = img_f3.tolist()[0]
    
            #read feature
        all_feature = pd.read_csv('C:/Users/00/Downloads/CBMIR/feature_list_BCNN_pixel301.csv', 
                                dtype={'Modality':str, 'part':str},
                                chunksize = 60000,
                                iterator = True)
        

        for feature_list in all_feature:
            feature_list['Cosine'] = 0
            feature_list['Modality'] = feature_list['Modality'].astype(int).map(M_relabel_map)
            feature_list['Cosine'] = feature_list.iloc[:,1156:-1].dot(img_f3)/(np.linalg.norm(feature_list.iloc[:,1156:-1], axis=1) * np.linalg.norm(img_f3))
            part_list = feature_list.nlargest(limit,'Cosine')
            f = feature_list.drop(part_list.index)
            
            part_list = self.deduplicate(part_list)
            print(len(part_list))
            while len(part_list) < limit:
                p = f.nlargest(limit-len(part_list),'Cosine')
                part_list = pd.concat([part_list, p])
                f.drop(p.index,inplace=True)
                part_list = self.deduplicate(part_list)
            r_list = pd.concat([r_list, 
                                part_list.loc[:,['Modality','part','new_path','Mo_part','Cosine']]],
                            ignore_index=True)
            
        r_list['Cosine'] = r_list['Cosine'].astype('float')
        r_list = r_list.nlargest(10,'Cosine')
        
        r_list['Mo_part'] = r_list['Mo_part'].astype(int).map(relabel_map)
        #r_list['new_path'] = 'static/minidataset/'+r_list['new_path'].str[42:].replace('\\','+')
        #r_list['new_path'] = r_list['new_path'].str.replace('\\','+')
        r_list['new_path'] = 'static/' + r_list['new_path'].str[28:]
        print(r_list['new_path'])
        r_list.to_csv('success.csv')
        return r_list

        

