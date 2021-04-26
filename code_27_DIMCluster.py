# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch深度学习和图神经网络（卷 1）——基础知识>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
Created on Sun Feb  2 23:50:52 2020

"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import random



#引入本地代码库
from code_26_DIM import ( train_loader,train_dataset,totalepoch,
                          device,batch_size,imshowrow, Encoder)

#加载模型
model_path = r'./DIMmodel/encoder%d.pth'% (totalepoch)
encoder = Encoder().to(device)
encoder.load_state_dict(torch.load(model_path,map_location=device))


# compute the latent space for each image and store in (latent, image)
batchesimg = []
batchesenc = []
batch = tqdm(train_loader, total=len(train_dataset) // batch_size)

for images, target in batch:
    images = images.to(device)
    with torch.no_grad():
        encoded, features = encoder(images)
    batchesimg.append(images)
    batchesenc.append(encoded)
    
#    if len(batchesimg)>2:
#        break

batchesenc = torch.cat(batchesenc,axis = 0)
batchesimg = torch.cat(batchesimg,axis = 0)


index = random.randrange(0, len(batchesenc))#随机获取一个索引

batchesenc[index].repeat(len(batchesenc),1)



l2_dis =F.mse_loss(batchesenc[index].repeat(len(batchesenc),1),batchesenc,reduction = 'none').sum(1)


findnum = 10 #查找图片的个数

_,indices = l2_dis.topk(findnum,largest=False)
_,indices_far = l2_dis.topk(findnum,)

indices = torch.cat([torch.tensor([index]).to(device),indices])
indices_far = torch.cat([torch.tensor([index]).to(device),indices_far])

rel = torch.cat([batchesimg[indices],batchesimg[indices_far]],axis = 0)
imshowrow(rel.cpu() ,nrow=len(indices))
#imshow(torchvision.utils.make_grid(rel,nrow=len(indices)))














