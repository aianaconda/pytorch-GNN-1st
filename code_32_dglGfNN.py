# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch深度学习和图神经网络（卷 1）——基础知识>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
Created on Mon Dec  9 21:32:35 2019
"""

import torch.nn as nn
from code_30_dglGAT import features,g,n_classes,feats_dim,trainmodel
from dgl.nn.pytorch.conv import SGConv

class GfNN(nn.Module):
    
    def __init__(self,in_feats, n_hidden, n_classes,
                 k, activation, dropout, cached=True,bias=False):
        super(GfNN, self).__init__()
        self.activation = activation
        self.sgc = SGConv(in_feats, n_hidden, k,cached, bias)
        self.fc = nn.Linear(n_hidden, n_classes)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, g,features):
        x = self.activation(self.sgc(g,features))
        x = self.dropout(x)
        return self.fc(x)
    

model = GfNN(feats_dim,n_hidden=512,n_classes=n_classes,
                   k=2,activation= nn.PReLU(512) ,dropout = 0.2)

print(model)
trainmodel(model,'code_32_dglGfNN_checkpoint.pt',g,features, lr=0.2, weight_decay=5e-06)


