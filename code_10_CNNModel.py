"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch深度学习和图神经网络（卷 1）——基础知识>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
Created on Tue Mar 19 22:24:58 2019
"""

import torchvision
import torchvision.transforms as tranforms
data_dir = './fashion_mnist/'
tranform = tranforms.Compose([tranforms.ToTensor()])
train_dataset = torchvision.datasets.FashionMNIST(data_dir, train=True, transform=tranform,download=True)

print("训练数据集条数",len(train_dataset))
val_dataset  = torchvision.datasets.FashionMNIST(root=data_dir, train=False, transform=tranform)
print("测试数据集条数",len(val_dataset))
import pylab
im = train_dataset[0][0]
im = im.reshape(-1,28)
pylab.imshow(im)
pylab.show()
print("该图片的标签为：",train_dataset[0][1])

############数据集的制作
import torch
batch_size = 10
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


from matplotlib import pyplot as plt
import numpy as np
def imshow(img):
    print("图片形状：",np.shape(img))
    npimg = img.numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

classes = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle_Boot')
sample = iter(train_loader)
images, labels = sample.next()
print('样本形状：',np.shape(images))
print('样本标签：',labels)
imshow(torchvision.utils.make_grid(images,nrow=batch_size))
print(','.join('%5s' % classes[labels[j]] for j in range(len(images))))

############数据集的制作




#########################################################################################################################
#定义myConNet模型类，该模型包括 2个卷积层和3个全连接层
from torch.nn import functional as F

class myConNet(torch.nn.Module):
    def __init__(self):
        super(myConNet, self).__init__()
        #定义卷积层
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3)
        self.conv3 = torch.nn.Conv2d(in_channels=12, out_channels=10, kernel_size=3)

    def forward(self, t):#搭建正向结构
        #第一层卷积和池化处理
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        #第二层卷积和池化处理
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        #第三层卷积和池化处理
        t = self.conv3(t)
        t = F.avg_pool2d(t, kernel_size=t.shape[-2:], stride=t.shape[-2:])

        return t.reshape(t.shape[:2])

if __name__ == '__main__':  

    #
    network = myConNet()
    #指定设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    network.to(device)
    print(network)#打印网络
    
    criterion = torch.nn.CrossEntropyLoss()  #实例化损失函数类
    optimizer = torch.optim.Adam(network.parameters(), lr=.01)
    
    for epoch in range(2): #数据集迭代2次
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0): #循环取出批次数据
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) #
            optimizer.zero_grad()#清空之前的梯度
            outputs = network(inputs)
            loss = criterion(outputs, labels)#计算损失
            loss.backward()  #反向传播
            optimizer.step() #更新参数
    
            running_loss += loss.item()
            if i % 1000 == 999:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    
    # 保存模型
    torch.save(network.state_dict(), './CNNFashingMNIST.pth')
    print('Finished Training')
    
    
    
    
    #使用模型
    network.load_state_dict(torch.load( './CNNFashingMNIST.pth'))#加载模型
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    inputs, labels = images.to(device), labels.to(device)
    
    imshow(torchvision.utils.make_grid(images,nrow=batch_size))
    print('真实标签: ', ' '.join('%5s' % classes[labels[j]] for j in range(len(images))))
    outputs = network(inputs)
    _, predicted = torch.max(outputs, 1)
    
    
    print('预测结果: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(len(images))))
    
    
    #测试模型
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            inputs, labels = images.to(device), labels.to(device)
            outputs = network(inputs)
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.to(device)
            c = (predicted == labels).squeeze()
            for i in range(10):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    
    sumacc = 0
    for i in range(10):
        Accuracy = 100 * class_correct[i] / class_total[i]
        print('Accuracy of %5s : %2d %%' % (classes[i], Accuracy ))
        sumacc =sumacc+Accuracy
    print('Accuracy of all : %2d %%' % ( sumacc/10. ))

