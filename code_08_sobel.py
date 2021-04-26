# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch深度学习和图神经网络（卷 1）——基础知识>配套代码 
@配套代码技术支持：bbs.aianaconda.com     
Created on Sat Apr 27 04:59:15 2019
"""

import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import torch
import torchvision.transforms as transforms

myimg = mpimg.imread('img.jpg') # 读取和代码处于同一目录下的图片
plt.imshow(myimg) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()
print(myimg.shape)


pil2tensor = transforms.ToTensor()
rgb_image = pil2tensor(myimg)
print(rgb_image[0][0])
print(rgb_image.shape)



sobelfilter =  torch.tensor([[-1.0,0,1],  [-2,0,2],  [-1.0,0,1.0]]*3).reshape([1,3,3, 3])

print(sobelfilter)
op =torch.nn.functional.conv2d(rgb_image.unsqueeze(0), sobelfilter, stride=3,padding = 1) #3个通道输入，生成1个feature map



ret = (op - op.min()).div(op.max() - op.min())
ret =ret.clamp(0., 1.).mul(255).int()
print(ret)

plt.imshow(ret.squeeze(),cmap='Greys_r') # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()

op=torch.nn.functional.max_pool2d(op,kernel_size =5, stride=5)
op = op.transpose(1,3).transpose(1,2)
print(op.shape)



