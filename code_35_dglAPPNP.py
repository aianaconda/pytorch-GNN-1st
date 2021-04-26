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
from dgl.nn.pytorch.conv import APPNPConv
import torch.nn.functional as F

class APPNP(nn.Module):
    def __init__(self,in_feats,n_classes,n_hidden, n_layers,
                 activation, feat_drop,  edge_drop, alpha,  k):
        super(APPNP, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(in_feats, n_hidden))
        # hidden layers
        for i in range(1, n_layers):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        # output layer
        self.layers.append(nn.Linear(n_hidden, n_classes))
        self.activation = activation
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.propagationconv = APPNPConv(k, alpha, edge_drop)


    def forward(self, g,features):
        # prediction step
        h = features
        h = self.feat_drop(h)
        h = self.activation(self.layers[0](h))
        for layer in self.layers[1:-1]:
            h = self.activation(layer(h))
        h = self.layers[-1](self.feat_drop(h))
        # propagation step
        h = self.propagationconv(g, h)
        return h

model = APPNP(feats_dim, n_classes,  n_hidden=54, n_layers=1, activation=F.relu,
              feat_drop=0.5, edge_drop=0.5,  alpha=0.1,  k=10)

print(model)
trainmodel(model,'code_35_dglAPPNP_checkpoint.pt',g,features, lr=1e-2, weight_decay=5e-6)








