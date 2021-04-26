"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch深度学习和图神经网络（卷 1）——基础知识>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
Created on Sat Jan 25 23:20:20 2020
"""

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

#引入本地代码库
from code_20_Variational_AutoEncoder import VAE,train,device,test_loader,to_img,imshow

class CondVAE(VAE):
    def __init__(self,hidden_1=256,hidden_2=512,
                      in_decode_dim=2+10,hidden_3=256):
        super(CondVAE, self).__init__(hidden_1,hidden_2,in_decode_dim,hidden_3)
        self.labfc1 = nn.Linear(10, hidden_1)

    def encode(self, x,lab):
        h1 = F.relu(self.fc1(x))
        lab1=F.relu(self.labfc1(lab))
        h1 =torch.cat([h1,lab1],axis=1)
        return self.fc21(h1), self.fc22(h1)


    def decode(self, z,lab):
        h3 = F.relu(self.fc3(torch.cat([z,lab],axis=1)   ))
        return self.fc4(h3)

    def forward(self, x,lab):
        mean, lg_var = self.encode(x,lab)
        z = self.reparametrize(mean, lg_var)
        return self.decode(z,lab), mean, lg_var


if __name__ == '__main__': 
    
    model = CondVAE().to(device)    

    train(model,50)

    
    sample = iter(test_loader)
    images, labels = sample.next()
    
    y_one_hots = torch.zeros(labels.shape[0],
                            10).scatter_(1,labels.view(labels.shape[0],1),1)
#######

    images2 = images.view(images.size(0), -1)
    with torch.no_grad():
        pred, mean, lg_var = model(images2.to(device),y_one_hots.to(device))
        
    pred = to_img(pred.cpu().detach())

 ##########  
    print("标签值：",labels) 
    z_sample = torch.randn(10,2).to(device)
    x_decoded = model.decode(z_sample,y_one_hots.to(device))
    rel = torch.cat([images,pred,to_img(x_decoded.cpu().detach())],axis = 0)
    imshow(torchvision.utils.make_grid(rel,nrow=10))
    plt.show()

