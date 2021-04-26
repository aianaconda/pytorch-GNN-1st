"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch深度学习和图神经网络（卷 1）——基础知识>配套代码
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
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
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        #定义全连接层
        self.fc1 = torch.nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = torch.nn.Linear(in_features=120, out_features=60)
        self.out = torch.nn.Linear(in_features=60, out_features=10)

    def forward(self, t):#搭建正向结构
        #第一层卷积和池化处理
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        #第二层卷积和池化处理
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        #搭建全连接网络，第一层全连接
        t = t.reshape(-1, 12 * 4 * 4)#将卷积结果由4维变为2维
        t = self.fc1(t)
        t = F.relu(t)
        #第二层全连接
        t = self.fc2(t)
        t = F.relu(t)
        #第三层全连接
        t = self.out(t)
        return t

if __name__ == '__main__':  
#
    network = myConNet()
    print(network)#打印网络
    
    #
    
    #print(network.parameters())
    
    ##指定设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    network.to(device)
    #print(network.parameters())
    
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
    
    print('Finished Training')
    # 保存模型
    torch.save(network.state_dict(), './CNNFashionMNIST.pth')
    
    
    #from sklearn.metrics import accuracy_score
    #outputs = network(inputs)
    #_, predicted = torch.max(outputs, 1)
    #print("训练时的准确率：",accuracy_score(predicted.cpu().numpy(),labels.cpu().numpy()))
    
    
    network.load_state_dict(torch.load( './CNNFashionMNIST.pth'))#加载模型
    
    #使用模型
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



'''

当你需要输出tensor查看的时候，或许需要设置一下默认的输出选项：
torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None)

其中precision是每一个元素的输出精度，默认是八位；threshold是输出时的阈值，当tensor中元素的个数大于该值时，进行缩略输出，默认时1000；edgeitems是输出的维度，默认是3；linewidth字面意思，每一行输出的长度；profile=None，修正默认设置（不太懂，感兴趣的可以试试）)
为了防止一些不正常的元素产生，比如特别小的数，pytorch支持如下设置：
torch.set_flush_denormal(mode)

mode中可以填true或者false
例子如下：
>>> torch.set_flush_denormal(True)
True
>>> torch.tensor([1e-323], dtype=torch.float64)
tensor([ 0.], dtype=torch.float64)
>>> torch.set_flush_denormal(False)
True
>>> torch.tensor([1e-323], dtype=torch.float64)
tensor(9.88131e-324 *
       [ 1.0000], dtype=torch.float64)

可以看出设置了之后，当出现极小数时，直接置为0了。文档中提出该功能必须要系统支持。
'''

