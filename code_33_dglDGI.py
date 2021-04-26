# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch深度学习和图神经网络（卷 1）——基础知识>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
Created on Mon Dec  9 21:32:35 2019
"""

import math
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch.conv import SGConv

from code_30_dglGAT import features,g,n_classes,feats_dim,n_edges,trainmodel

class MSGC(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 k,
                 n_layers,
                 activation,
                 dropout):
        super(MSGC, self).__init__()
        self.layers = nn.ModuleList()
        # input layer

        self.activation = activation
        self.layers.append(SGConv(in_feats, n_hidden, k,cached=False, bias=False))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SGConv(n_hidden, n_hidden, k,cached=False, bias=False))
        # output layer

        self.layers.append(SGConv(n_hidden, n_classes, k,cached=False, bias=False))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g,features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g,h )
            
            if i != len(self.layers)-1:
                h = self.activation(h)

        return h

class Encoder(nn.Module):
    def __init__(self, in_feats, n_hidden,k, n_layers, activation, dropout):
        super(Encoder, self).__init__()
        self.conv = MSGC( in_feats, n_hidden, n_hidden, k,n_layers, activation, dropout)

    def forward(self,g, features, corrupt=False):
        if corrupt:
            perm = torch.randperm(g.number_of_nodes())#返回0--n个随机顺序的整数
            features = features[perm]
        features = self.conv(g,features)
        return features


class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.FC = nn.Linear(n_hidden,n_hidden) 	#定义全连接层
    def forward(self, features, summary):
        features = torch.matmul(features, self.FC(summary) )
        return features


class DGI(nn.Module):
    def __init__(self,  in_feats, n_hidden,k, n_layers, activation, dropout):
        super(DGI, self).__init__()
        self.encoder = Encoder( in_feats, n_hidden, k,n_layers, activation, dropout)
        self.discriminator = Discriminator(n_hidden)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, g,features):

        positive = self.encoder(g,features, corrupt=False)
        negative = self.encoder(g,features, corrupt=True)
        summary = torch.sigmoid(positive.mean(dim=0))

        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)

        l1 = self.loss(positive, torch.ones_like(positive))
        l2 = self.loss(negative, torch.zeros_like(negative))

        return l1 + l2

dgi = DGI(feats_dim, n_hidden=512, k=2,n_layers=1, activation =nn.PReLU(512), dropout=0.1)
dgi.cuda()

dgi_optimizer = torch.optim.Adam(dgi.parameters(), lr=1e-3, weight_decay=5e-06)
cnt_wait = 0
best = 1e9
best_t = 0
dur = []
patience = 20
for epoch in range(300):
    dgi.train()
    if epoch >= 3:
        t0 = time.time()

    dgi_optimizer.zero_grad()
    loss = dgi(g,features)
    loss.backward()
    dgi_optimizer.step()

    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(dgi.state_dict(), 'code_41_dglGDI_best_dgi.pt')
    else:
        cnt_wait += 1

    if cnt_wait == patience:
        print('Early stopping!')
        break

    if epoch >= 3:
        dur.append(time.time() - t0)

    print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | "
          "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                        n_edges / np.mean(dur) / 1000))



class Classifier(nn.Module):
    def __init__(self, n_hidden, n_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(n_hidden, n_classes)

    def forward(self, features):
        features = self.fc(features)
        return torch.log_softmax(features, dim=-1)

classifier = Classifier(n_hidden=512, n_classes=n_classes)

dgi.load_state_dict(torch.load('code_33_dglGDI_best_dgi.pt'))
embeds = dgi.encoder(g,features, corrupt=False)
embeds = embeds.detach()


trainmodel(classifier,'code_33_dglGDI_checkpoint.pt',embeds, lr=1e-2, weight_decay=5e-06,
           loss_fcn = F.nll_loss)


