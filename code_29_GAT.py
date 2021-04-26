# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch深度学习和图神经网络（卷 1）——基础知识>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
Created on Sat Oct 19 20:03:44 2019
"""

from pathlib import Path #提升路径的兼容性
#引入矩阵运算相关库
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix,csr_matrix,diags,eye

#引入深度学习框架库
import torch
from torch import nn
import torch.nn.functional as F
#引入绘图库
import matplotlib.pyplot as plt

'''
conda install pandas
'''


#输出运算资源请况
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

#输出样本路径
path = Path('data/cora')
print(path)

#读取论文内容数据，并将其转化为数组
paper_features_label = np.genfromtxt(path/'cora.content', dtype=np.str)
print(paper_features_label,np.shape(paper_features_label))

#取出数据的第一列：论文的ID
papers = paper_features_label[:,0].astype(np.int32)
print(papers)
#为论文重新编号，{31336: 0, 1061127: 1,……
paper2idx = {k:v for v,k in enumerate(papers)}


#将数据中间部分的字标签取出，转化成矩阵
features = csr_matrix(paper_features_label[:, 1:-1], dtype=np.float32)
print(np.shape(features))

#将最后一项的论文分类属性取出，并转化为分类索引
labels = paper_features_label[:, -1]
lbl2idx = {k:v for v,k in enumerate(sorted(np.unique(labels)))}
labels = [lbl2idx[e] for e in labels]
print(lbl2idx,labels[:5])


#读取论文关系数据，并将其转化为数组
edges = np.genfromtxt(path/'cora.cites', dtype=np.int32)
print(edges,np.shape(edges))
#转化为新编号节点间的关系
edges = np.asarray([paper2idx[e] for e in edges.flatten()], np.int32).reshape(edges.shape)
print(edges,edges.shape)

# 计算邻接矩阵（Adjacency matrix） ，行列都为论文个数
adj = coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                 shape=(len(labels), len(labels)), dtype=np.float32)

# Symmetric adjacency matrix
#adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
#生成无向图对称矩阵
adj_long = adj.multiply(adj.T < adj)
adj = adj_long+adj_long.T


##############################

def normalize(mx):#定义函数，对矩阵数据进行归一化
    '''Row-normalize sparse matrix'''
    rowsum = np.array(mx.sum(1))#每一篇论文的字数
    r_inv = (rowsum ** -1).flatten() #取总字数的倒数
    r_inv[np.isinf(r_inv)] = 0.#将Nan值设为0
    r_mat_inv = diags(r_inv)#将总字数的倒数做成对角矩阵
    mx = r_mat_inv.dot(mx)#左乘一个矩阵，相当于每个元素除以总数
    return mx

#对 features矩阵进行归一化（每行的总和为1）
features = normalize(features)

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = diags(r_inv)
    return mx.dot(r_mat_inv).transpose().dot(r_mat_inv)
# 对邻接矩阵对角线添加1，将其变为自循环图。同时再对其进行归一化
#adj = normalize(adj + eye(adj.shape[0]))
adj = normalize_adj(adj + eye(adj.shape[0]))
################################################



# Data as tensors
adj = torch.FloatTensor(adj.todense()) #节点间的关系
features = torch.FloatTensor(features.todense())#节点自身的特征
labels = torch.LongTensor(labels) #每个节点的分类标签

#划分数据集
n_train = 200
n_val = 300
n_test = len(features) - n_train - n_val
np.random.seed(34)
idxs = np.random.permutation(len(features))#将原有索引打乱顺序
#计算每个数据集的索引
idx_train = torch.LongTensor(idxs[:n_train])
idx_val   = torch.LongTensor(idxs[n_train:n_train+n_val])
idx_test  = torch.LongTensor(idxs[n_train+n_val:])


#分配运算资源
adj = adj.to(device)
features = features.to(device)
labels = labels.to(device)
idx_train = idx_train.to(device)
idx_val = idx_val.to(device)
idx_test = idx_test.to(device)


def mish(x):					#Mish激活函数
    return x *( torch.tanh(F.softplus(x)))


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout=0.6):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = mish(torch.matmul(a_input, self.a).squeeze(2))#[N,N]
        
        
        
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        print(attention )
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        return mish(h_prime)
    

    
class GAT(nn.Module):
    def __init__(self,nfeat,  nclass,nhid, dropout,  nheads):
        super(GAT, self).__init__()

        self.attentions = [GraphAttentionLayer(nfeat, nhid,dropout) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout)

    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        return self.out_att(x, adj)
 
    
   
    

n_labels = labels.max().item() + 1  #分类个数 7
n_features = features.shape[1]   #节点个数 1433
print(n_labels, n_features)


def accuracy(output,y):
    return (output.argmax(1) == y).type(torch.float32).mean().item()

def step():
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss = F.cross_entropy(output[idx_train], labels[idx_train])
    acc = accuracy(output[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()
    return loss.item(), acc

def evaluate(idx):
    model.eval()
    output = model(features, adj)
    loss = F.cross_entropy(output[idx], labels[idx]).item()
    return loss, accuracy(output[idx], labels[idx])




model = GAT(n_features, n_labels, 16,0.1,8).to(device)




from ranger import *
optimizer = Ranger(model.parameters())




from tqdm import tqdm  #pip install tqdm
#训练模型
epochs = 3000

print_steps = 50
train_loss, train_acc = [], []
val_loss, val_acc = [], []
for i in tqdm(range(epochs)):
    tl, ta = step()
    train_loss += [tl]
    train_acc += [ta]
    if (i+1)%print_steps == 0 or i == 0:
        tl, ta = evaluate(idx_train)
        vl, va = evaluate(idx_val)
        val_loss += [vl]
        val_acc += [va]
        print(f'{i+1:6d}/{epochs}: train_loss={tl:.4f}, train_acc={ta:.4f}'+
              f', val_loss={vl:.4f}, val_acc={va:.4f}')

#输出最终结果
final_train, final_val, final_test = evaluate(idx_train), evaluate(idx_val), evaluate(idx_test)
print(f'Train     : loss={final_train[0]:.4f}, accuracy={final_train[1]:.4f}')
print(f'Validation: loss={final_val[0]:.4f}, accuracy={final_val[1]:.4f}')
print(f'Test      : loss={final_test[0]:.4f}, accuracy={final_test[1]:.4f}')


#可视化训练过程
fig, axes = plt.subplots(1, 2, figsize=(15,5))
ax = axes[0]
axes[0].plot(train_loss[::print_steps] + [train_loss[-1]], label='Train')
axes[0].plot(val_loss, label='Validation')
axes[1].plot(train_acc[::print_steps] + [train_acc[-1]], label='Train')
axes[1].plot(val_acc, label='Validation')
for ax,t in zip(axes, ['Loss', 'Accuracy']): ax.legend(), ax.set_title(t, size=15)



#输出模型预测结果
output = model(features, adj)

samples = 10
idx_sample = idx_test[torch.randperm(len(idx_test))[:samples]]

idx2lbl = {v:k for k,v in lbl2idx.items()}
df = pd.DataFrame({'Real': [idx2lbl[e] for e in labels[idx_sample].tolist()],
                   'Pred': [idx2lbl[e] for e in output[idx_sample].argmax(1).tolist()]})
print(df)



