# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch深度学习和图神经网络（卷 1）——基础知识>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
Created on Sun Feb  2 09:22:59 2020
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

#生成模拟数据
def gen_x():
    return np.sign(np.random.normal(0.,1.,[data_size,1]))

def gen_y(x):
    return x+np.random.normal(0.,0.5,[data_size,1])

data_size = 1000
x_sample=gen_x()
y_sample=gen_y(x_sample)
plt.scatter(np.arange(len(x_sample)), x_sample, s=10,c='b',marker='o')  
plt.scatter(np.arange(len(y_sample)), y_sample, s=10,c='y',marker='o')
plt.show()





class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(1, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x, y):
        h1 = F.relu(self.fc1(x)+self.fc2(y))
        h2 = self.fc3(h1)
        return h2    

model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


n_epoch = 500
plot_loss = []
for epoch in tqdm(range(n_epoch)):
    x_sample=gen_x()
    y_sample=gen_y(x_sample)
    y_shuffle=np.random.permutation(y_sample)
    

    x_sample = torch.from_numpy(x_sample).type(torch.FloatTensor)
    y_sample = torch.from_numpy(y_sample).type(torch.FloatTensor)
    y_shuffle = torch.from_numpy(y_shuffle).type(torch.FloatTensor)    

    model.zero_grad()
    pred_xy = model(x_sample, y_sample)#联合分布
    pred_x_y = model(x_sample, y_shuffle)#边缘分布

    ret = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
    loss = - ret  # maximize
    plot_loss.append(loss.data)
    
    loss.backward()
    optimizer.step()
    

plot_y = np.array(plot_loss).reshape(-1,)
plt.plot(np.arange(len(plot_loss)), -plot_y, 'r') 








