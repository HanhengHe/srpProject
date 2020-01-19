import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import PIL.Image as Image


###############################################################################
# fine-tuning base on LeNet-5                                                 #
# since we got a ideal model which is easy to train                           #
###############################################################################

########################################
#  define the model and train it       #
########################################
"""
class LeNet_5(nn.Module):
    def __init__(self):
        super().__init__()

        # 定义卷积层
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)  # 非对称连接，利于提取多种组合特征

        # 全连接层
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # 定义正向传播
    def forward(self, X):
        # Mnist 数据集的size为batch*28*28
        X = self.conv1(X)  # batch*1*28*28-(5*5conv)->batch*6*24*24
        X = F.relu(X)
        X = F.max_pool2d(X, (2, 2), stride=2)  # batch*6*24*24-(max_pooling)->batch*6*12*12
        X = F.relu(X)
        X = self.conv2(X)  # batch*12*12-(5*5conv)->batch*16*8*8
        X = F.relu(X)
        X = F.max_pool2d(X, (2, 2), stride=2)  # batch*16*8*8-(max_pooling)->batch*16*4*4
        X = F.relu(X)
        X = X.view(X.size()[0], -1)
        X = self.fc1(X)
        X = F.relu(X)
        X = self.fc2(X)
        X = F.relu(X)
        X = self.fc3(X)
        X = F.log_softmax(X, dim=1)
        return X


# 读取本地数据
sourceFilePath = 'D:\\WINTER\\PycharmProjects\\data\\Mnist\\train'

sourceSet = []
sourceLabels = []

sourceFile = open(sourceFilePath)

for line in sourceFile.readlines():
    #  修改格式
    data = line.split(')')[0]
    label = line.split(')')[1].replace('\n', '')

    data = data.split(',')
    data[0] = data[0].replace('(', '')

    #  data set调整为float类型，label调整为int类型

    sourceLabels.append(int(label))

    #  置入数据结构中
    sourceSet.append(np.array([int(d) / 255 for d in data]).reshape((28, 28)))

sourceFile.close()

trainSize = len(sourceSet)

sourceSet = torch.from_numpy(np.array([sourceSet])).permute(1, 0, 2, 3)
sourceLabels = torch.from_numpy(np.array(sourceLabels))

model = LeNet_5()
optimizer = optim.Adam(model.parameters())

Epoch = 100
for epoch in range(Epoch):
    model.train()  # 训练模式
    optimizer.zero_grad()
    output = model(sourceSet.float())
    loss = F.nll_loss(output, sourceLabels.long())
    loss.backward()
    optimizer.step()
    print("epoch %s: Loss %s" % (epoch + 1, loss.item()))
    if float(loss.item()) <= 0.05:
        break
"""
################################################################
#  we consider it an ideal model, thus we use it without test  #
################################################################

# read USPS data
targetFilePath = 'D:\\WINTER\\PycharmProjects\\data\\USPS\\USPS'
targetSet = []
targetLabels = []
targetFile = open(targetFilePath)

counter = 1

for i, line in enumerate(targetFile.readlines()):
    #  修改格式
    data = line.split(' ')
    label = int(i/1100) + 1
    if label == 11:
        label = 0
    targetLabels.append(int(label))

    #  reshape image from USPS using openCV
    temp = np.array([int(d) for d in data]).reshape((16, 16))
    image = Image.fromarray(temp)
    image = image.resize((28, 28), Image.ANTIALIAS)

    #  置入数据结构中
    targetSet.append(np.asarray(image))

targetFile.close()
