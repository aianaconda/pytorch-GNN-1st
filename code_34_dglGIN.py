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
from dgl.nn.pytorch.conv import GINConv


class GIN(nn.Module):
    def __init__(self,  in_feats,
                 n_classes,
                 n_hidden,
                 n_layers,
                 init_eps,
                 learn_eps):
        
        super(GIN, self).__init__()
        self.layers = nn.ModuleList()
        
        self.layers.append( GINConv(  nn.Sequential(
                                    nn.Dropout(0.6),
                                    nn.Linear(in_feats, n_hidden),
                                    nn.ReLU() ),
                            'max',#'mean',
                            init_eps,
                            learn_eps )   )
        
        for i in range(n_layers - 1):
            self.layers.append( GINConv(nn.Sequential(
                                        nn.Dropout(0.6),
                                        nn.Linear(n_hidden, n_hidden),
                                        nn.ReLU() ),
                                    'sum',#'mean',
                                    init_eps,
                                    learn_eps )  )
        
        self.layers.append(  GINConv( nn.Sequential(
                                        nn.Dropout(0.6),
                                        nn.Linear(n_hidden, n_classes) ),
                                    'mean',#'mean',
                                    init_eps,
                                    learn_eps )  )

    def forward(self, g,features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return h


model = GIN(feats_dim, n_classes,  n_hidden=16, n_layers=1, init_eps=0, learn_eps=True)

print(model)
trainmodel(model,'code_34_dglGIN_checkpoint.pt',g,features, lr=1e-2, weight_decay=5e-6)








