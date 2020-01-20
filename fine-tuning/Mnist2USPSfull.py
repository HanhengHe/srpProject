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


modelPath = "D:\\WINTER\\PycharmProjects\\PyTorch\\LeNet-5\\model"

model = LeNet_5()
model.load_state_dict(torch.load(modelPath))
model.eval()

# read USPS data
targetFilePath = 'D:\\WINTER\\PycharmProjects\\data\\USPS\\USPS'
targetTrainSet = []
targetTrainLabels = []
targetTestSet = []
targetTestLabels = []
targetFile = open(targetFilePath)

counter = 1

for i, line in enumerate(targetFile.readlines()):
    #  修改格式
    data = line.split(' ')
    label = int(i / 1100) + 1
    if label == 10:
        label = 0

    #  reshape image from USPS using openCV
    temp = np.array([int(d) for d in data]).reshape((16, 16))
    image = Image.fromarray(temp)
    image = image.resize((28, 28), Image.ANTIALIAS)

    if counter <= 100:
        targetTrainLabels.append(int(label))

        #  置入数据结构中
        targetTrainSet.append(np.asarray(image))
    else:
        targetTestLabels.append(int(label))

        #  置入数据结构中
        targetTestSet.append(np.asarray(image))

    counter += 1
    if counter == 1101:
        counter = 1

targetFile.close()

trainSet = torch.from_numpy(np.array([targetTrainSet])).permute(1, 0, 2, 3)
trainLabels = torch.from_numpy(np.array(targetTrainLabels))
testSet = torch.from_numpy(np.array([targetTestSet])).permute(1, 0, 2, 3)
testLabels = torch.from_numpy(np.array(targetTestLabels))

optimizer = optim.Adam([{'params': model.fc1.parameters(), 'lr': 1e-2}, {'params': model.fc2.parameters(), 'lr': 1e-2}, {'params': model.fc3.parameters(), 'lr': 1e-2},
                        {'params': model.conv1.parameters(), 'lr': 1e-4}, {'params': model.conv2.parameters(), 'lr': 1e-4}])

Epoch = 1200
for epoch in range(Epoch):
    model.train()  # 训练模式
    optimizer.zero_grad()
    output = model(trainSet.float())
    loss = F.nll_loss(output, trainLabels.long())
    loss.backward()
    optimizer.step()
    print("epoch %s: Loss %s" % (epoch + 1, loss.item()))
    if float(loss.item()) <= 0.05:
        break

model.eval()  # 测试模式
test_loss = 0
correct = 0
with torch.no_grad():
    output = model(testSet.float())
    test_loss += F.nll_loss(output, testLabels.long(), reduction='sum').item()  # 将一批的损失相加
    predict = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
    correct += predict.eq(testLabels.view_as(predict)).sum().item()

test_loss /= 10000
print("Test: Average loss:%s, Accuracy: %s/%s (%s)"
      % (test_loss, correct, 10000, correct / 10000))
