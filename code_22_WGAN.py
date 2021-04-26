# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch深度学习和图神经网络（卷 1）——基础知识>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
Created on Sat Jan 25 23:20:20 2020
"""

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
import torch.autograd as autograd
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib
 

#引入本地代码库
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

def imshow(img,filename=None):
    npimg = img.numpy()
    plt.axis('off')
    array = np.transpose(npimg, (1, 2, 0))    
    if filename!=None:
        matplotlib.image.imsave(filename, array)       
    else:
        plt.imshow(array  ) 
#        plt.savefig(filename) 保存图片
        plt.show()

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  ])

data_dir = './fashion_mnist/'
train_dataset = torchvision.datasets.FashionMNIST(data_dir, train=True, 
                                                  transform=img_transform,download=True)
train_loader = DataLoader(train_dataset,batch_size=1024, shuffle=True)

val_dataset = torchvision.datasets.FashionMNIST(data_dir, train=False, 
                                                transform=img_transform)
test_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
#指定设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



class WGAN_D(nn.Module):
    def __init__(self,inputch=1):
        super(WGAN_D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inputch, 64,4, 2, 1),  # batch, 64, 28, 28
            nn.LeakyReLU(0.2, True),
            nn.InstanceNorm2d(64, affine=True)   )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128,4, 2, 1),  # batch, 64, 14, 14
            nn.LeakyReLU(0.2, True),
            nn.InstanceNorm2d(128, affine=True)   )
        self.fc = nn.Sequential(
            nn.Linear(128*7*7, 1024),
            nn.LeakyReLU(0.2, True),            )
        self.fc2 =nn.Sequential(
                nn.InstanceNorm1d(1, affine=True),
                nn.Flatten(),
                nn.Linear(1024, 1)  )
        

               
    def forward(self, x,*arg):#batch, width, height, channel=1
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.reshape(x.size(0),1, -1)        
        x = self.fc2(x)
        return x.view(-1, 1).squeeze(1)

class WGAN_G(nn.Module):
    def __init__(self, input_size,input_n=1):
        super(WGAN_G, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(input_size*input_n, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024)  )

        self.fc2 = nn.Sequential(
            nn.Linear(1024,7*7*128),
            nn.ReLU(True),
            nn.BatchNorm1d(7*7*128)   )
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False),  # batch, 64, 14, 14
            nn.ReLU(True),
            nn.BatchNorm2d(64)   )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 4, 2, padding=1, bias=False),  # batch, 64, 28, 28
            nn.Tanh(),  )
        

    def forward(self, x,*arg):
        x = self.fc1(x)
        x = self.fc2(x)
        
        x = x.view(x.size(0), 128, 7, 7)
        x = self.upsample1(x)
        img = self.upsample2(x)

        return img

# Loss weight for gradient penalty
lambda_gp = 10
def compute_gradient_penalty(D, real_samples, fake_samples,y_one_hot):

    eps = torch.FloatTensor(real_samples.size(0),1,1,1).uniform_(0,1).to(device)
    # Get random interpolation between real and fake samples
    X_inter = (eps * real_samples + ((1 - eps) * fake_samples)).requires_grad_(True)
    d_interpolates = D(X_inter,y_one_hot)
    fake = torch.full((real_samples.size(0), ), 1, device=device)
    
    # Get gradient
    gradients = autograd.grad( outputs=d_interpolates,
            inputs=X_inter,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penaltys = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penaltys
    


def train(D,G,outdir,z_dimension ,num_epochs = 30):
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.001)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.001)
    
    os.makedirs(outdir, exist_ok=True)

    # train
    for epoch in range(num_epochs):
        for i, (img, lab) in enumerate(train_loader):
            num_img = img.size(0)
            # =================train discriminator

            real_img = img.to(device)
            y_one_hot = torch.zeros(lab.shape[0],10).scatter_(1,
                                   lab.view(lab.shape[0],1),1).to(device)
            for ii in range(5):
                d_optimizer.zero_grad()
        
                # compute loss of real_img
                real_out = D(real_img,y_one_hot)# closer to 1 means better
                # compute loss of fake_img
                z = torch.randn(num_img, z_dimension).to(device)
                fake_img = G(z,y_one_hot)
                fake_out = D(fake_img,y_one_hot)# closer to 0 means better
                
                gradient_penalty = compute_gradient_penalty(D, 
                                        real_img.data, fake_img.data,y_one_hot)
    
                # Loss measures generator's ability to fool the discriminator
                d_loss = -torch.mean(real_out) + torch.mean(fake_out) + gradient_penalty
                d_loss.backward()
                d_optimizer.step()
    
            # ===============train generator
            # compute loss of fake_img
            for ii in range(1):
                g_optimizer.zero_grad()
                z = torch.randn(num_img, z_dimension).to(device)
                fake_img = G(z,y_one_hot)
                fake_out = D(fake_img,y_one_hot)
                g_loss = -torch.mean(fake_out)
                g_loss.backward()
                g_optimizer.step()
                
        fake_images = to_img(fake_img.cpu().data)
        real_images = to_img(real_img.cpu().data)
        rel = torch.cat([to_img(real_images[:10]),fake_images[:10]],axis = 0)
        imshow(torchvision.utils.make_grid(rel,nrow=10),
              os.path.join(outdir, 'fake_images-{}.png'.format(epoch+1) ) )

        
        print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
                      'D real: {:.6f}, D fake: {:.6f}'
                      .format(epoch, num_epochs, d_loss.data, g_loss.data,
                              real_out.data.mean(), fake_out.data.mean()))
        
    torch.save(G.state_dict(), os.path.join(outdir, 'generator.pth'  ) )
    torch.save(D.state_dict(), os.path.join(outdir, 'discriminator.pth'  ) )   

def displayAndTest(D,G,z_dimension):
    # 可视化结果
    sample = iter(test_loader)
    images, labels = sample.next()
    y_one_hot = torch.zeros(labels.shape[0],10).scatter_(1,
                                   labels.view(labels.shape[0],1),1).to(device)
    
    num_img = images.size(0)
    with torch.no_grad():
        z = torch.randn(num_img, z_dimension).to(device)
        fake_img = G(z,y_one_hot)
    fake_images = to_img(fake_img.cpu().data)
    rel = torch.cat([to_img(images[:10]),fake_images[:10]],axis = 0)
    imshow(torchvision.utils.make_grid(rel,nrow=10))
    print(labels[:10])     
    
if __name__ == '__main__': 
    
    z_dimension = 40  # noise dimension
    
    D = WGAN_D().to(device)  # discriminator model
    G = WGAN_G(z_dimension).to(device)  # generator model
    train(D,G,'./w_img',z_dimension)
    
    displayAndTest(D,G,z_dimension)


    
 
    
    
    