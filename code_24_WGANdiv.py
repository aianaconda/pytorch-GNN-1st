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
import torch.autograd as autograd
import os


#引入本地代码库
from code_22_wGan import ( train_loader,to_img,
                          device,displayAndTest,imshow, WGAN_G,WGAN_D)


#计算W散度
def compute_w_div(real_samples,real_out, fake_samples,fake_out):
    #定义参数
    k = 2
    p = 6
    
    #计算真实空间的梯度
    weight = torch.full((real_samples.size(0), ), 1, device=device)    
    real_grad = autograd.grad(outputs=real_out,
                              inputs=real_samples, 
                              grad_outputs=weight, 
                              create_graph=True, 
                              retain_graph=True, only_inputs=True)[0]
    #L2范数
    real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) 

    #计算模拟空间的梯度
    fake_grad = autograd.grad(outputs=fake_out, 
                              inputs=fake_samples, 
                              grad_outputs=weight, 
                              create_graph=True, 
                              retain_graph=True, only_inputs=True)[0]
    #L2范数
    fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) 
    #计算W散度距离
    div_gp = torch.mean(real_grad_norm** (p / 2) + fake_grad_norm** (p / 2)) * k / 2
    return div_gp
    
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
                real_img= real_img.requires_grad_(True)#为了求梯度
                # compute loss of real_img
                real_out = D(real_img,y_one_hot)# closer to 1 means better
                # compute loss of fake_img
                z = torch.randn(num_img, z_dimension).to(device)
                fake_img = G(z,y_one_hot)
                fake_out = D(fake_img,y_one_hot)# closer to 0 means better
                
               
                gradient_penalty_div = compute_w_div(real_img,real_out, 
                                                     fake_img,fake_out,y_one_hot)
    
                # Loss measures generator's ability to fool the discriminator
                d_loss = -torch.mean(real_out) + torch.mean(fake_out) + gradient_penalty_div
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


    
if __name__ == '__main__': 
    
    z_dimension = 40  # noise dimension
    
    D = WGAN_D().to(device)  # discriminator model
    G = WGAN_G(z_dimension).to(device)  # generator model
    train(D,G,'./wdiv_img',z_dimension)
    
    displayAndTest(D,G,z_dimension)

    
    