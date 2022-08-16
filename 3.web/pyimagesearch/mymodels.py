import torch
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models

CUDA = torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")

class alexnetModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(alexnetModel, self).__init__()
        alexnet = models.alexnet(pretrained=True)
        for param in alexnet.parameters():
            param.requires_grad = False
        alexnet.classifier[6] = nn.Linear(4096, 4096)
        self.feature = alexnet
        self.relu = nn.ReLU(inplace = True)
        self.fc2 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.feature(x)
        x = self.relu(x)
        x = self.fc2(x)        
        return x
    
    def get_feature(self, x):
        x = self.feature(x)    
        return x

class resnetModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(resnetModel, self).__init__()
        resnet152 = models.resnet152(pretrained=True)
        for param in resnet152.parameters():
            param.requires_grad = False
        resnet152.fc = nn.Linear(2048, 2048)
        self.feature = resnet152
        self.relu = nn.ReLU(inplace = True)
        self.fc2 = nn.Linear(2048, num_classes)
        

    def forward(self, x):
        x = self.feature(x)
        x = self.relu(x)
        x = self.fc2(x)        
        return x
    
    def get_feature(self, x):
        x = self.feature(x)    
        return x


class inceptionModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(inceptionModel, self).__init__()
        inception_v3 = models.inception_v3(pretrained=True, aux_logits=False)
        for param in inception_v3.parameters():
            param.requires_grad = False
        inception_v3.fc = nn.Linear(2048, 2048)
        self.feature = inception_v3
        self.relu = nn.ReLU(inplace = True)
        self.fc2 = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.feature(x)
        x = self.relu(x)
        x = self.fc2(x)        
        return x
    
    def get_feature(self, x):
        x = self.feature(x)    
        return x

class densenetModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(densenetModel, self).__init__()
        densenet201 = models.densenet201(pretrained=True)
        for param in densenet201.parameters():
            param.requires_grad = False
        densenet201.classifier = nn.Linear(1920, 1920)
        self.feature = densenet201
        self.relu = nn.ReLU(inplace = True)
        self.fc2 = nn.Linear(1920, num_classes)

    def forward(self, x):
        x = self.feature(x)
        x = self.relu(x)
        x = self.fc2(x)        
        return x
    
    def get_feature(self, x):
        x = self.feature(x)    
        return x


class BCNN(nn.Module):
    def __init__(self):
        super(BCNN,self).__init__()
        # Load pretrained model
        Densenet201 = models.densenet201(pretrained=True)
        self.begin = nn.Sequential(*list(Densenet201.features.children())[0:4])
        
        self.block1 = nn.Sequential(*list(Densenet201.features.children())[4:6])
        self.block2 = nn.Sequential(*list(Densenet201.features.children())[6:8])
        # Level-1 classifier after second conv block
        self.level_one_feature = nn.Linear(50176,128)
        self.level_one_clf = nn.Linear(128, 14)
        
        self.block3 = nn.Sequential(*list(Densenet201.features.children())[8:10])
        # Level-2 classifier after 4 conv block
        self.level_two_feature = nn.Linear(43904, 1024)
        self.level_two_clf = nn.Linear(1024, 40)
        
        self.block4 = nn.Sequential(*list(Densenet201.features.children())[10:12])
        # Level-3 classifier after fifth conv block
        self.level_three_feature = nn.Linear(1920, 1920)
        self.level_three_clf = nn.Linear(1920, 52)
        
    def forward(self,x):
        x = self.begin(x)
        x = self.block1(x)
        x = self.block2(x)
        lvl_one = torch.flatten(x,1)
        lvl_one = self.level_one_feature(lvl_one)
        lvl_one = F.relu(lvl_one, inplace= True)
        lvl_one = self.level_one_clf(lvl_one)
        
        x = self.block3(x)
        lvl_two = torch.flatten(x,1)
        lvl_two = self.level_two_feature(lvl_two)
        lvl_two = F.relu(lvl_two, inplace= True)
        lvl_two = self.level_two_clf(lvl_two)
        
        x = self.block4(x)
        lvl_three = F.relu(x, inplace= True)
        lvl_three = F.adaptive_avg_pool2d(lvl_three, (1,1))
        lvl_three = torch.flatten(lvl_three,1)
        lvl_three = self.level_three_feature(lvl_three)
        lvl_three = self.level_three_clf(lvl_three)
        return lvl_one, lvl_two, lvl_three
    
    def get_feature(self, x):
        x = self.begin(x)
        x = self.block1(x)
        x = self.block2(x)
        lvl_one = torch.flatten(x,1)
        lvl_one = self.level_one_feature(lvl_one)
        
        x = self.block3(x)
        lvl_two = torch.flatten(x,1)
        lvl_two = self.level_two_feature(lvl_two)
        x = self.block4(x)
        lvl_three = F.relu(x, inplace= True)
        lvl_three = F.adaptive_avg_pool2d(lvl_three, (1,1))
        lvl_three = torch.flatten(lvl_three,1)
        lvl_three = self.level_three_feature(lvl_three)
        return lvl_one, lvl_two, lvl_three
    
    def lock(self, level):
        if level == 0:
            for param in self.parameters():
                param.requires_grad = True
        if level == 1:
            for param in self.begin.parameters():
                param.requires_grad = False
            for param in self.block1.parameters():
                param.requires_grad = False
            for param in self.block2.parameters():
                param.requires_grad = False
            for param in self.level_one_feature.parameters():
                param.requires_grad = False
            for param in self.level_one_clf.parameters():
                param.requires_grad = False
        if level == 2:
            for param in self.begin.parameters():
                param.requires_grad = False
            for param in self.block1.parameters():
                param.requires_grad = False
            for param in self.block2.parameters():
                param.requires_grad = False
            for param in self.level_one_feature.parameters():
                param.requires_grad = False
            for param in self.level_one_clf.parameters():
                param.requires_grad = False
            for param in self.block3.parameters():
                param.requires_grad = False
            for param in self.level_two_feature.parameters():
                param.requires_grad = False
            for param in self.level_two_clf.parameters():
                param.requires_grad = False
        if level == 3:
            for param in self.parameters():
                param.requires_grad = False