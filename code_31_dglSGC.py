# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch深度学习和图神经网络（卷 1）——基础知识>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
Created on Mon Dec  9 21:32:35 2019
"""


from code_30_dglGAT import features,g,n_classes,feats_dim,trainmodel
from dgl.nn.pytorch.conv import SGConv

model = SGConv(feats_dim,
                   n_classes,
                   k=2,
                   cached=True,
                   bias=False)

print(model)
trainmodel(model,'code_39_dglSGC_checkpoint.pt',g,features, lr=0.2, weight_decay=5e-06)


