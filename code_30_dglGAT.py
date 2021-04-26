# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch深度学习和图神经网络（卷 1）——基础知识>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
Created on Mon Dec  9 21:32:35 2019
"""

import dgl
import torch
from torch import nn
from dgl.data import citation_graph 
import torch.nn.functional as F
from dgl.nn.pytorch import  GATConv

data = citation_graph.CoraDataset()

#输出运算资源请况
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

features = torch.FloatTensor(data.features).to(device)
labels = torch.LongTensor(data.labels).to(device)

train_mask = torch.BoolTensor(data.train_mask).to(device)
val_mask = torch.BoolTensor(data.val_mask).to(device)
test_mask = torch.BoolTensor(data.test_mask).to(device)

feats_dim = features.shape[1]
n_classes = data.num_labels
n_edges = data.graph.number_of_edges()
print("""----数据统计------
  #边数 %d
  #样本特征维度 %d
  #类别数 %d 
  #训练样本 %d
  #验证样本 %d
  #测试样本 %d""" % (n_edges, feats_dim,n_classes,
       train_mask.int().sum().item(),val_mask.int().sum().item(),
       test_mask.int().sum().item()))




#邻接矩阵
g = dgl.DGLGraph(data.graph)#将networkx图转成DGL图
g.add_edges(g.nodes(), g.nodes()) #添加自环
n_edges = g.number_of_edges()




class GAT(nn.Module):
    def __init__(self,
                 num_layers,#层数
                 in_dim,    #输入维度
                 num_hidden,#隐藏层维度
                 num_classes,#类别个数
                 heads,#多头注意力的计算次数
                 activation,#激活函数
                 feat_drop,#特征层的丢弃率
                 attn_drop,#注意力分数的丢弃率
                 negative_slope,#LeakyReLU激活函数的负向参数
                 residual):#是否使用残差网络结构
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.gat_layers.append(GATConv(in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        #定义隐藏层
        for l in range(1, num_layers):
            #多头注意力 the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        #输出层
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, g,inputs):
        h = inputs
        for l in range(self.num_layers):#隐藏层
            h = self.gat_layers[l](g, h).flatten(1)
        #输出层
        logits = self.gat_layers[-1](g, h).mean(1)
        return logits


def getmodel( GAT ):
    # create model
    num_heads = 8
    num_layers = 1
    num_out_heads =1

    model = GAT(
                num_layers,
                feats_dim,#输入维度
                num_hidden= 8,
                num_classes = n_classes,
                heads = ([num_heads] * num_layers) + [num_out_heads],#总的注意力头数
                activation = F.elu,
                feat_drop=0.6,
                attn_drop=0.6,
                negative_slope = 0.2,
                residual = True)
    return model


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(model, labels, mask,*modelinput):
    model.eval()
    with torch.no_grad():
        logits = model(*modelinput)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)
    
class EarlyStopping:
    def __init__(self, patience=10,modelname='checkpoint.pt'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.modelname = modelname

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            torch.save(model.state_dict(), self.modelname)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            torch.save(model.state_dict(), self.modelname)
            self.counter = 0
        return self.early_stop

def trainmodel(model,modelname,*modelinput, lr=0.005, weight_decay=5e-4,
               loss_fcn = torch.nn.CrossEntropyLoss()):  
    stopper = EarlyStopping(patience=100,modelname=modelname)
    model.to(device)
    
    optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay)
    import time
    import numpy as np
    
    
    model.train()
    # initialize graph
    dur = []
    for epoch in range(200):
        
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(*modelinput)

        

        loss = loss_fcn(logits[train_mask], labels[train_mask])
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if epoch >= 3:
            dur.append(time.time() - t0)
    
        train_acc = accuracy(logits[train_mask], labels[train_mask])
    
    
        val_acc = accuracy(logits[val_mask], labels[val_mask])
        if stopper.step(val_acc, model):   
            break
    
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
              " ValAcc {:.4f} | ETputs(KTEPS) {:.2f}".
              format(epoch, np.mean(dur), loss.item(), train_acc,
                     val_acc, n_edges / np.mean(dur) / 1000))
    
    model.load_state_dict(torch.load(modelname))
    acc = evaluate(model,labels, test_mask,*modelinput)
    print("\nTest Accuracy {:.4f}".format(acc))

if __name__ == '__main__':
    model = getmodel(GAT)
    print(model)
    trainmodel(model,'code_30_dglGAT_checkpoint.pt',g,features)