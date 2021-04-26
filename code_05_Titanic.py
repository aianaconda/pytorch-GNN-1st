# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch深度学习和图神经网络（卷 1）——基础知识>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
Created on Sun Nov  3 15:36:39 2019

"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from scipy import stats
import pandas as pd


titanic_data = pd.read_csv("titanic3.csv")
print(titanic_data.columns )




#用哑变量将指定字段转成one-hot
titanic_data = pd.concat([titanic_data,
                          pd.get_dummies(titanic_data['sex']),
                          pd.get_dummies(titanic_data['embarked'],prefix="embark"),
                          pd.get_dummies(titanic_data['pclass'],prefix="class")], axis=1)

print(titanic_data.columns )
print(titanic_data['sex'])
print(titanic_data['female'])

#处理None值
titanic_data["age"] = titanic_data["age"].fillna(titanic_data["age"].mean())
titanic_data["fare"] = titanic_data["fare"].fillna(titanic_data["fare"].mean())#乘客票价

#删去无用的列
titanic_data = titanic_data.drop(['name','ticket','cabin','boat','body','home.dest','sex','embarked','pclass'], axis=1)
print(titanic_data.columns )
#
####################################


#分离样本和标签
labels = titanic_data["survived"].to_numpy()

titanic_data = titanic_data.drop(['survived'], axis=1)
data = titanic_data.to_numpy()

#样本的属性名称
feature_names = list(titanic_data.columns)


#将样本分为训练和测试两部分
np.random.seed(10)#设置种子，保证每次运行所分的样本一致
train_indices = np.random.choice(len(labels), int(0.7*len(labels)), replace=False)
test_indices = list(set(range(len(labels))) - set(train_indices))
train_features = data[train_indices]
train_labels = labels[train_indices]
test_features = data[test_indices]
test_labels = labels[test_indices]
len(test_labels)#393
###########################################

class Mish(nn.Module):#Mish激活函数
    def __init__(self):
        super().__init__()
        print("Mish activation loaded...")
    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


torch.manual_seed(0)  #设置随机种子

class ThreelinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(12, 12)
        self.mish1 = Mish()
        self.linear2 = nn.Linear(12, 8)
        self.mish2 = Mish()
        self.linear3 = nn.Linear(8, 2)
        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss() #定义交叉熵函数

    def forward(self, x): #定义一个全连接网络
        lin1_out = self.linear1(x)
        out1 = self.mish1(lin1_out)
        out2 = self.mish2(self.linear2(out1))

        return self.softmax(self.linear3(out2))
    

    def getloss(self,x,y): #实现LogicNet类的损失值计算接口
        y_pred = self.forward(x)
        loss = self.criterion(y_pred,y)#计算损失值得交叉熵
        return loss

##############################
        
net = ThreelinearModel()


num_epochs = 200

optimizer = torch.optim.Adam(net.parameters(), lr=0.04)



input_tensor = torch.from_numpy(train_features).type(torch.FloatTensor)
label_tensor = torch.from_numpy(train_labels)

losses = []#定义列表，用于接收每一步的损失值
for epoch in range(num_epochs): 
    loss = net.getloss(input_tensor,label_tensor)
    losses.append(loss.item())
    optimizer.zero_grad()#清空之前的梯度
    loss.backward()#反向传播损失值
    optimizer.step()#更新参数
    if epoch % 20 == 0:
        print ('Epoch {}/{} => Loss: {:.2f}'.format(epoch+1, num_epochs, loss.item()))


os.makedirs('models', exist_ok=True)
torch.save(net.state_dict(), 'models/titanic_model.pt')    

from code_02_moons_fun import plot_losses
plot_losses(losses)

#输出训练结果
out_probs = net(input_tensor).detach().numpy()
out_classes = np.argmax(out_probs, axis=1)
print("Train Accuracy:", sum(out_classes == train_labels) / len(train_labels))

#测试模型
test_input_tensor = torch.from_numpy(test_features).type(torch.FloatTensor)
out_probs = net(test_input_tensor).detach().numpy()
out_classes = np.argmax(out_probs, axis=1)
print("Test Accuracy:", sum(out_classes == test_labels) / len(test_labels))

#####################################
