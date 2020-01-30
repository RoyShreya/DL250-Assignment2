import torch
import torchvision
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
PATH1='FashionMNIST/processed/training.pt'
PATH2='FashionMNIST/processed/test.pt'
PATHConv='MyConvModel.pt'
PATHMNN='MyMultilayerNN.pt'
colordim=1
_numclass=10
epoch=200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#FashionMNISTobj=torchvision.datasets.FashionMNIST(root=os.getcwd(), train=True, transform=None, target_transform=None, download=True)
if os.path.exists(PATH1):
     print("a")
     print(torch.load(PATH1)[0])
     train_X=torch.load(PATH1)[0]
    
     train_label=torch.load(PATH1)[1]
    
class MyDatasetForConvNN(Dataset):

    """
    A customized data loader.
    """
    def __init__(self):
        """ Intialize the dataset
        """
        self.input = torch.as_tensor(train_X,dtype=torch.float).unsqueeze(1)
        self.label=  torch.as_tensor(train_label,dtype=torch.int).unsqueeze(1)
        
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
class MyDatasetForMNN(Dataset):

    """
    A customized data loader.
    """
    def __init__(self):
        """ Intialize the dataset
        """
        X=np.reshape(train_X.detach().cpu().numpy().ravel(),(train_X.size(0),28*28,1))
        self.input = torch.as_tensor(X,dtype=torch.float)
        self.label=  torch.as_tensor(train_label,dtype=torch.int).unsqueeze(1)
        
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
class My_multi_layer_net(torch.nn.Module):
    def __init__(self, D_in,_numclass): #D_out is the number of classes
       
        super(My_multi_layer_net, self).__init__()
        H1=int(D_in/2)
        H2=int(D_in/4)
        H3=int(D_in/8)
        H4=int(D_in/16)
        self.linear1 = nn.Linear(D_in, H1, bias=True)
        self.linear2 = nn.Linear(H1,H2, bias=True)
        self.linear3 = nn.Linear(H2, H3, bias=True)
        self.linear4 = nn.Linear(H3, H4, bias=True)
        self.linear5 = nn.Linear(H4, 20, bias=True) #49,20
        self.linear6 = nn.Linear(20, _numclass, bias=True)
        self.bn1 = nn.BatchNorm1d(H1)
        self.bn2 = nn.BatchNorm1d(H2)
        self.bn3 = nn.BatchNorm1d(H3)
        self.bn4 = nn.BatchNorm1d(H4)        
        self.bn5 = nn.BatchNorm1d(20)        
    def forward(self, x):
       
        h1_relu = F.relu(self.linear1(x))#.clamp(min=0)
        h2_relu = F.relu(self.linear2(self.bn1(h1_relu)))
        h3_relu = F.relu(self.linear3(self.bn2(h2_relu)))
        h4_relu = F.relu(self.linear4(self.bn3(h3_relu)))
        h5_relu = F.relu(self.linear5(self.bn4(h4_relu)))
        y_pred=self.linear6(self.bn5(h5_relu))
        return y_pred


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

def train_with_Conv(BATCH_SIZE):
    full_dataset=MyDatasetForConvNN()
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size]) 
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_loader.len=train_size
    val_loader.len=val_size
    train_loss_list=[]
    val_loss_list=[]
    acc_list=[]
    #001
    print(type(device))
    if torch.cuda.is_available():
    #if 1==0:
        net=My_convolution_neural_net()  #inp dimension and hidden dimension and output dsimension ie. numof classes here
        net=net.float()
        net=net.cuda()
        for iteration in range(0,epoch):
                '''
                if(iteration==0):
                    if os.path.exists(PATH):
                        net.load_state_dict(torch.load(PATH)['model_state_dict'])  
                '''
                #learning_rate = .1#00000001#001#0001#001#00001#1#00001
                optimizer = torch.optim.Adam(net.parameters(),weight_decay=1e-4)
                
                criterion=nn.CrossEntropyLoss()
                j=0
                for batch_index,(inp,label)  in enumerate(train_loader,0) : 
                    inp=torch.as_tensor(inp, dtype=torch.float)
                    label=torch.as_tensor(label, dtype=torch.long)
                    #print(inp)
                    inp=inp.to(device='cuda')# dtype=torch.double) #double model weight so make all input float tensor to double
                    label=label.to(device='cuda')#, dtype=torch.double)
                    out=net(inp)
                    out=out.unsqueeze(2)
                    optimizer.zero_grad()
                    loss = criterion(out,label)
                    train_loss_list.append(loss.item())
                    np.save('TrainLossWithConvNet.npy',np.array(train_loss_list))
                    loss.backward()
                    optimizer.step()
                val_loss=0.0
                acc=0.0
                
                with torch.no_grad():
                    j=0
                    
                    for batch_index,(inp,label)  in enumerate(val_loader,0):
                         inp=torch.as_tensor(inp, dtype=torch.float)
                         label=torch.as_tensor(label, dtype=torch.long)
                         inp=inp.to(device='cuda')# dtype=torch.double) #double model weight so make all input float tensor to double
                         label=label.to(device='cuda')#, dtype=torch.double)
                         out=net(inp)
                         out=out.unsqueeze(2)
                         batch_acc=0.0
                         for b in range(0,BATCH_SIZE):
                             #print(np.argmax(np.array(out[b,:,0].detach().cpu().numpy().reshape(1,out.size(1),out.size(2))),axis=1))
                             #print(label[b,0].item())
                             #print(np.array(out[b,:,0].detach().cpu().numpy().reshape(1,out.size(1),out.size(2))))
                             # print(np.argmax(np.array(out[b,:,0].detach().cpu().numpy().reshape(1,out.size(1),out.size(2))),axis=1))
                             #print(label[b,0].item())
                             if np.argmax(np.array(out[b,:,0].detach().cpu().numpy().reshape(1,out.size(1),out.size(2))),axis=1)==label[b,0].item():
                                 #  print("T")
                                 batch_acc=batch_acc+1
                             #else:
                                 #print("F")
                         acc=acc+batch_acc
                         val_loss=val_loss+criterion(out,label).item()
                         j=j+1
                    #print(j)
                    #print(test_size)
                    avg_val_loss=val_loss/j 
                  #  print(acc)
                    #print(test_size)
                    avg_acc=acc/val_size
                    val_loss_list.append(avg_val_loss)
                    np.save('ValLossWithConvNet.npy',np.array(val_loss_list)) 
                   # print('iter no:'+str(epoch))
                   # print('Validation Loss:')
                    #print(avg_val_loss) 
                    #print('accuracy') 
                    acc_list.append(avg_acc)
                    print("At"+str(iteration)+"th"+str(avg_acc)) 
                  
                  
                    np.save('ValAccWithConvNet.npy',np.array(acc_list)) 
                torch.save({'epoch': iteration,'model_state_dict':net.state_dict(),'optimizer_state_dict': optimizer.state_dict()}, PATHConv)  
                if avg_acc==1.0:
                   break 

#train_with_Conv(1000)
def train_with_MNN(BATCH_SIZE):
    full_dataset=MyDatasetForMNN()
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size]) 
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_loader.len=train_size
    val_loader.len=val_size
    train_loss_list=[]
    val_loss_list=[]
    acc_list=[]
    #001
    print(type(device))
    if torch.cuda.is_available():
    #if 1==0:
        net=My_multi_layer_net(784,10)  #inp dimension and hidden dimension and output dsimension ie. numof classes here
        net=net.float()
        net=net.cuda()
        for iteration in range(0,epoch):
                '''
                if(iteration==0):
                    if os.path.exists(PATH):
                        net.load_state_dict(torch.load(PATH)['model_state_dict'])  
                '''
                #learning_rate = .1#00000001#001#0001#001#00001#1#00001
                optimizer = torch.optim.Adam(net.parameters(),weight_decay=1e-4)
                
                criterion=nn.CrossEntropyLoss()
                j=0
                for batch_index,(inp,label)  in enumerate(train_loader,0) : 
                    inp=torch.as_tensor(inp, dtype=torch.float).view(BATCH_SIZE,784)
                    label=torch.as_tensor(label, dtype=torch.long)
                    #print(inp)
                    inp=inp.to(device='cuda')# dtype=torch.double) #double model weight so make all input float tensor to double
                    label=label.to(device='cuda')#, dtype=torch.double)
                    out=net(inp)
                    out=out.unsqueeze(2)
                    optimizer.zero_grad()
                    loss = criterion(out,label)
                    train_loss_list.append(loss.item())
                    np.save('TrainLossWithMultilayerNet.npy',np.array(train_loss_list))
                    loss.backward()
                    optimizer.step()
                val_loss=0.0
                acc=0.0
                
                with torch.no_grad():
                    j=0
                    
                    for batch_index,(inp,label)  in enumerate(val_loader,0):
                         inp=torch.as_tensor(inp, dtype=torch.float).view(BATCH_SIZE,784)
                         label=torch.as_tensor(label, dtype=torch.long)
                         inp=inp.to(device='cuda')# dtype=torch.double) #double model weight so make all input float tensor to double
                         label=label.to(device='cuda')#, dtype=torch.double)
                         out=net(inp)
                         out=out.unsqueeze(2)
                         batch_acc=0.0
                         for b in range(0,BATCH_SIZE):
                             #print(np.argmax(np.array(out[b,:,0].detach().cpu().numpy().reshape(1,out.size(1),out.size(2))),axis=1))
                             #print(label[b,0].item())
                             #print(np.array(out[b,:,0].detach().cpu().numpy().reshape(1,out.size(1),out.size(2))))
                             # print(np.argmax(np.array(out[b,:,0].detach().cpu().numpy().reshape(1,out.size(1),out.size(2))),axis=1))
                             #print(label[b,0].item())
                             if np.argmax(np.array(out[b,:,0].detach().cpu().numpy().reshape(1,out.size(1),out.size(2))),axis=1)==label[b,0].item():
                                 #  print("T")
                                 batch_acc=batch_acc+1
                             #else:
                                 #print("F")
                         acc=acc+batch_acc
                         val_loss=val_loss+criterion(out,label).item()
                         j=j+1
                    #print(j)
                    #print(test_size)
                    avg_val_loss=val_loss/j 
                  #  print(acc)
                    #print(test_size)
                    avg_acc=acc/val_size
                    val_loss_list.append(avg_val_loss)
                    np.save('ValLossWithMultilayerNet.npy',np.array(val_loss_list)) 
                   # print('iter no:'+str(epoch))
                   # print('Validation Loss:')
                    #print(avg_val_loss) 
                    #print('accuracy') 
                    acc_list.append(avg_acc)
                    print("At"+str(iteration)+"th"+str(avg_acc)) 
                  
                  
                    np.save('ValAccWithMultilayerNet.npy',np.array(acc_list)) 
                torch.save({'epoch': iteration,'model_state_dict':net.state_dict(),'optimizer_state_dict': optimizer.state_dict()}, PATHMNN)  
                if avg_acc==1.0:
                   break 

train_with_MNN(1000)
train_with_Conv(1000)
