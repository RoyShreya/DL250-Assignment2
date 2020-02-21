import torch
import torchvision
import numpy as np
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
#import matplotlib.pyplot as plt
import scipy
import scipy.misc
#from PIL import Image
from torch.utils.data import Dataset,DataLoader
f1=open('multi-layer-net.txt','w')
f2=open('convolution-neural-net.txt','w')
PATHConv='models/MyConvModel.pt'
PATHMNN='models/MyMultilayerNN.pt'
PATH_testdata='FashionMNIST/processed/test.pt'
_numclass=10
BATCH_SIZE=100
FashionMNISTobj=torchvision.datasets.FashionMNIST(root=os.getcwd(), train=True, transform=None, target_transform=None, download=True)
if os.path.exists(PATH_testdata):
     
     print(torch.load(PATH_testdata)[0])
     test_X=torch.load(PATH_testdata)[0]
     test_Y=(torch.load(PATH_testdata)[1])
testY=np.reshape(torch.load(PATH_testdata)[1].cpu().numpy(),(test_Y.size(0)))
trans=torchvision.transforms.ToPILImage(mode='L')
revtrans=torchvision.transforms.ToTensor()
rotationObj=torchvision.transforms.RandomRotation((90,90),expand=False)
HflippedObj=torchvision.transforms.RandomHorizontalFlip(p=1)
VflippedObj=torchvision.transforms.RandomVerticalFlip(p=1)  
print(testY)
class MyDatasetForConvNN(Dataset):

    """
    A customized data loader.
    """
    def __init__(self):
        """ Intialize the dataset
        """
       
        self.input = torch.as_tensor(test_X,dtype=torch.float).unsqueeze(1)
        self.label=  torch.as_tensor(test_Y,dtype=torch.int).unsqueeze(1)
        
        
          
        #print(self.label.size())
        #print(self.input.size())
        
        self.len = (self.input.size(0))
       
    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        if index<self.len :
            
            return self.input[index],self.label[index]
        else:
            return NULL

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

class MyDatasetForMNN(Dataset):

    """
    A customized data loader.
    """
    def __init__(self):
        """ Intialize the dataset
        """
        X=np.reshape(test_X.detach().cpu().numpy().ravel(),(test_X.size(0),28*28,1))
        self.input = torch.as_tensor(X,dtype=torch.float)
        self.label=  torch.as_tensor(test_Y,dtype=torch.int).unsqueeze(1)
        
        print(self.label.size())
        print(self.input.size())
        self.len = (self.input.size(0))
    
    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        if index<self.len :
            
            return self.input[index],self.label[index]
        else:
            return NULL

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
class My_convolution_neural_net(torch.nn.Module):
    def __init__(self,colordim=1 ,out_channel=_numclass,p1=14,p2=8,p3=4,k1=3):
        super(My_convolution_neural_net, self).__init__()
        self.conv1_1 = nn.Conv2d(colordim, p1, k1)  # input of (n,n,1), output of (n-2,n-2,64)
        self.conv1_2 = nn.Conv2d(p1, p1, k1)
        self.bn1 = nn.BatchNorm2d(p1)
        self.maxpool=nn.MaxPool2d(2)
	
        self.conv2_1 = nn.Conv2d(p1, p2, k1) 
        self.conv2_2 = nn.Conv2d(p2, p3, k1)
        #self.bn2 = nn.BatchNorm2d(p2)
        self.conv3_1 = nn.Conv2d(p3, p3, k1)
        #self.conv3_1 = nn.Conv2d(p2, out_channel, 3) #n/4-5
        self.conv4_1 = nn.Conv2d(p3, out_channel, k1)
       
        
    def forward(self, x):
       
        x1 = F.relu(self.bn1(self.conv1_2(F.relu(self.conv1_1(x))))) 
        x2 = F.relu((self.conv2_2(F.relu(self.conv2_1(self.maxpool(x1))))))
        x3 = (self.conv4_1(F.relu(self.maxpool(self.conv3_1(x2)))))
        #print(x3.size())
        return x3.view(x3.size(0),x3.size(1))



'''
class My_multi_layer_net(torch.nn.Module):
    def __init__(self, D_in,_numclass): #D_out is the number of classes
       
        super(My_multi_layer_net, self).__init__()
       
        H1=int(D_in/2) #D_in is the input dimension which is 784 here
        H2=int(D_in/4)
        H3=int(D_in/8)
        H4=int(D_in/16)
        
        #H1=400
        
        
        #H3=25
        self.linear1 = nn.Linear(D_in, H1, bias=True)
        #self.linear2 = nn.Linear(H1,H2, bias=True)
        self.linear3 = nn.Linear(H1, H3, bias=True)
        #self.linear4 = nn.Linear(H3, H4, bias=True)
        #self.linear5 = nn.Linear(H4, 20, bias=True) #49,20
        self.linear6 = nn.Linear(25, _numclass, bias=True)
        self.bn1 = nn.BatchNorm1d(H1)
        #self.bn2 = nn.BatchNorm1d(H2)
        self.bn3 = nn.BatchNorm1d(H3)
        #self.bn4 = nn.BatchNorm1d(H4)        
        self.bn5 = nn.BatchNorm1d(25)    
        self.dropout=nn.Dropout(p=.5,inplace=False)    
    def forward(self, x):
       
        h1_relu = F.relu(self.linear1(x))#.clamp(min=0)
        #h2_relu = F.relu(self.linear2(self.bn1(h1_relu)))
        h3_relu = F.relu(self.linear3(self.bn2(h2_relu)))
        #h4_relu = F.relu(self.linear4(self.bn3(h3_relu)))
        #h5_relu = F.relu((self.linear5(self.bn4(h4_relu))))
        y_pred=(self.linear6(self.bn5(h3_relu)))
        return y_pred
'''
class My_multi_layer_net(torch.nn.Module):
    def __init__(self, D_in,_numclass): #D_out is the number of classes
       
        super(My_multi_layer_net, self).__init__()
        '''
        H1=int(D_in/2) #D_in is the input dimension which is 784 here
        H2=int(D_in/4)
        H3=int(D_in/8)
        H4=int(D_in/16)
        '''
        H1=200
        H2=100
        H3=50
        
        self.linear1 = nn.Linear(D_in, H1, bias=True)
        self.linear2 = nn.Linear(H1,H2, bias=True)
        self.linear3 = nn.Linear(H2, H3, bias=True)
        #self.linear4 = nn.Linear(H3, H4, bias=True)
        #self.linear5 = nn.Linear(H4, 20, bias=True) #49,20
        self.linear6 = nn.Linear(H3, _numclass, bias=True)
        self.bn1 = nn.BatchNorm1d(H1)
        self.bn2 = nn.BatchNorm1d(H2)
        self.bn3 = nn.BatchNorm1d(H3)
        #self.bn4 = nn.BatchNorm1d(H4)        
        #self.bn5 = nn.BatchNorm1d(H3)    
        self.dropout=nn.Dropout(p=.5,inplace=False)    
    def forward(self, x):
       
        h1_relu = F.relu(self.linear1(x))#.clamp(min=0)
        h2_relu = F.relu(self.linear2(self.bn1(h1_relu)))
        h3_relu = F.relu(self.linear3(self.bn2(h2_relu)))
        #h4_relu = F.relu(self.linear4(self.bn3(h3_relu)))
        #h5_relu = F.relu((self.linear5(self.bn4(h4_relu))))
        y_pred=(self.linear6(self.bn3(h3_relu)))
        return y_pred


test_datasetMNN=MyDatasetForMNN()
test_datasetConv=MyDatasetForConvNN()
test_loader_Conv = DataLoader(dataset=test_datasetConv, batch_size=BATCH_SIZE, shuffle=True)
test_loader_MNN =DataLoader(dataset=test_datasetMNN, batch_size=BATCH_SIZE, shuffle=True)
test_len=len(test_datasetMNN)
print("TestSize="+str(test_len))
labelConvListBatch=[]
labelMNNListBatch=[]
truelabelConvListBatch=[]
truelabelMNNListBatch=[]
#if torch.cuda.is_available():
if 1==0:
    print("Running on GPU")
    netMNN=My_multi_layer_net(784,10) 
    netConv=My_convolution_neural_net() #inp dimension and hidden dimension and output dsimension ie. numof classes here
    print("Total number of parameters in the MNN")
    MNN_total_params = sum(p.numel() for p in netMNN.parameters() if p.requires_grad)
    print(MNN_total_params)
    print("Total number of parameters in the ConvNet")
    Conv_total_params = sum(q.numel() for q in netConv.parameters() if q.requires_grad)
    print(Conv_total_params)
   
    netMNN=netMNN.float()
    netConv=netConv.float()
    netMNN=netMNN.cuda()
    netConv=netConv.cuda()
    netMNN.load_state_dict(torch.load(PATHMNN)['model_state_dict'])  
    netConv.load_state_dict(torch.load(PATHConv)['model_state_dict']) 
    
    for batch_index,(inp,label)  in enumerate(test_loader_MNN,0) :
        #X=np.reshape(test_X.detach().cpu().numpy().ravel(),(test_X.size(0),28*28,1))
        #inp = torch.as_tensor(X,dtype=torch.float)
        out=netMNN(inp.view(BATCH_SIZE,784).cuda())
        #print(label.size())
        labelMNN=np.argmax(np.array(out.detach().cpu().numpy().reshape(out.size(0),out.size(1))),axis=1)
        labelMNNListBatch.append(labelMNN)
        truelabelMNN=label.detach().cpu().numpy()
        truelabelMNNListBatch.append(truelabelMNN)
    
    for batch_index,(inp,label)  in enumerate(test_loader_Conv,0) :
       
        out=netConv(torch.as_tensor(inp).cuda())
        
        labelConv=np.argmax(np.array(out.detach().cpu().numpy().reshape(out.size(0),out.size(1))),axis=1)
        labelConvListBatch.append(labelConv)
        truelabelConv=label.detach().cpu().numpy()
        truelabelConvListBatch.append(truelabelConv)
    
else:
    print("Running On CPU")
    netMNN=My_multi_layer_net(784,10) 
    netConv=My_convolution_neural_net() #inp dimension and hidden dimension and output dsimension ie. numof classes here
    print("Total number of parameters in the MNN")
    MNN_total_params = sum(p.numel() for p in netMNN.parameters() if p.requires_grad)
    print(MNN_total_params)
    print("Total number of parameters in the ConvNet")
    Conv_total_params = sum(q.numel() for q in netConv.parameters() if q.requires_grad)
    print(Conv_total_params)
    netMNN=netMNN.float()
    netConv=netConv.float()
    
    netMNN.load_state_dict(torch.load(PATHMNN)['model_state_dict'])  
    netConv.load_state_dict(torch.load(PATHConv)['model_state_dict']) 
    for batch_index,(inp,label)  in enumerate(test_loader_MNN,0) :
        #X=np.reshape(test_X.detach().cpu().numpy().ravel(),(test_X.size(0),28*28,1))
        #inp = torch.as_tensor(X,dtype=torch.float)
        out=netMNN(inp.view(BATCH_SIZE,784))
        
        labelMNN=np.argmax(np.array(out.detach().cpu().numpy().reshape(out.size(0),out.size(1))),axis=1)
        labelMNNListBatch.append(labelMNN)
        truelabelMNN=label.detach().cpu().numpy()
        truelabelMNNListBatch.append(truelabelMNN)
    
    for batch_index,(inp,label)  in enumerate(test_loader_Conv,0) :
       
        out=netConv(torch.as_tensor(inp))
        
        labelConv=np.argmax(np.array(out.detach().cpu().numpy().reshape(out.size(0),out.size(1))),axis=1)
        labelConvListBatch.append(labelConv)
        truelabelConv=label.detach().cpu().numpy()
        truelabelConvListBatch.append(truelabelConv)
PredictedWithConv=np.reshape(np.array(labelConvListBatch),(test_len)).astype(np.uint8)
PredictedWithMNN=np.reshape(np.array(labelMNNListBatch),(test_len)).astype(np.uint8)
trueWithConv=np.reshape(np.array(truelabelConvListBatch),(test_len)).astype(np.uint8)
trueWithMNN=np.reshape(np.array(truelabelMNNListBatch),(test_len)).astype(np.uint8)

np.savetxt(f1,PredictedWithMNN)
np.savetxt(f2,PredictedWithConv)
f1.close()
f2.close()

MultiLayerNetTestAcc=np.sum(PredictedWithMNN==trueWithMNN)*100.0/test_len
ConvNetTestAcc=np.sum(PredictedWithConv==trueWithConv)*100.0/test_len
print("Test Accuracy by Multi-Layer-Net in percentage:")
print(MultiLayerNetTestAcc)
print("Test Accuracy by ConvNet in percentage:")
print(ConvNetTestAcc)
confMatCNN=np.zeros((10,10))
confMatMNN=np.zeros((10,10))
print(PredictedWithMNN.shape[0])
for m in range(0,PredictedWithMNN.shape[0]):
    n=m
    #if trueWithMNN[m]==PredictedWithMNN[n]:
    confMatMNN[trueWithMNN[m],PredictedWithMNN[n]]=confMatMNN[trueWithMNN[m],PredictedWithMNN[n]]+1
    #if trueWithConv[m]==PredictedWithConv[n]:
    confMatCNN[trueWithConv[m],PredictedWithConv[n]]=confMatCNN[trueWithConv[m],PredictedWithConv[n]]+1
print("confusion matrix for MNN")
print(confMatMNN)
print("confusion matrix for CNN")
print(confMatCNN)

np.save('ConfusionMatrixOfMNN.npy',confMatMNN)
np.save('ConfusionMatrixOfCNN.npy',confMatCNN)
scipy.misc.toimage(confMatMNN,high=255,low=0).save('ConfusionMatrixOfMNN.jpeg')
scipy.misc.toimage(confMatCNN,high=255,low=0).save('ConfusionMatrixOfCNN.jpeg')
