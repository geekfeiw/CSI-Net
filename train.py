import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import time
from tqdm import tqdm

# from resnet_upsample import ResNet, ResidualBlock, Bottleneck
# from resnet_upsample3x3 import ResNet, ResidualBlock, Bottleneck
# from inception3 import *
# from inceptionv4 import *
# from vgg_net import *
# from alex_net import AlexNet
# 
# res_net_use_this, csinet1.0
from model.res_net_use_this import ResNet, ResidualBlock, Bottleneck
# resnet_generation_upsample, csinet1.5
# from resnet_generation_upsample import ResNet, ResidualBlock, Bottleneck


batch_size = 20
num_epochs = 20
learning_rate = 0.001


# load data
data = sio.loadmat('data/aug_train_data.mat')
train_data = data['aug_train_data']
train_label = data['aug_train_label']

# label matrix organized as nSamplex5, where the 1st coloum is the index of personID, the latter 4 are 4 biometrcs
train_label[:, 0] = train_label[:, 0] - 1 # 1--30 -> 0--29

num_train_instances = len(train_data)
# prepare data, nSample x nChannel x width x height
# reshape train data size to nSample x nSubcarrier x 1 x 1
train_data = torch.from_numpy(train_data).type(torch.FloatTensor).view(num_train_instances, 30, 1, 1)
train_label = torch.from_numpy(train_label).type(torch.FloatTensor)
train_dataset = TensorDataset(train_data, train_label)
train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


# load test data
data = sio.loadmat('data/test.mat')
test_data = data['test_data']
test_label = data['test_label']
test_label[:, 0] = test_label[:, 0] - 1

num_test_instances = len(test_data)
# prepare data, nSample x nChannel x width x height
# reshape test data size to nSample x nSubcarrier x 1 x 1
test_data = torch.from_numpy(test_data).type(torch.FloatTensor).view(num_test_instances, 30, 1, 1)
test_label = torch.from_numpy(test_label).type(torch.FloatTensor)
test_dataset = TensorDataset(test_data, test_label)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


resnet = ResNet(ResidualBlock, [2, 2, 2, 2], 30)
# resnet = ResNet(ResidualBlock, [3, 4, 6, 3], 30)
# resnet = ResNet(Bottleneck, [3, 4, 6, 3], 10)
# resnet = ResNet(Bottleneck, [3, 4, 23, 3], 30)
# inception = InceptionV4(30)
# vgg = VGG(make_layers(cfg['E'], batch_norm=True))
# alexnet = AlexNet().cuda()
# vgg = vgg.cuda()
# alexnet = alexnet.cuda()
# alexnet.eval()

resnet = resnet.cuda()

criterion1 = nn.CrossEntropyLoss().cuda()
criterion2 = nn.L1Loss().cuda()
optimizer = torch.optim.Adam(resnet.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9, 12, 15, 18], gamma=0.3)

for epoch in range(num_epochs):
    print('Epoch:', epoch)
    resnet.train()

    scheduler.step()
    # trained_num = 0
    for (samples, labels) in tqdm(train_data_loader):

        # sample_len = len(samples)
        # trained_num += sample_len
        # print('Process', 100*trained_num/num_train_instances)

        samplesV = Variable(samples.cuda())
        labels = labels.squeeze()
        labelsV = Variable(labels.cuda())

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        predict_label = resnet(samplesV)

        lossC = criterion1(predict_label[0], labelsV[:, 0].type(torch.LongTensor).cuda())

        lossR1 = criterion2(predict_label[1][:, 0], labelsV[:, 1])
        lossR2 = criterion2(predict_label[1][:, 1], labelsV[:, 2])
        lossR3 = criterion2(predict_label[1][:, 2], labelsV[:, 3])
        lossR4 = criterion2(predict_label[1][:, 3], labelsV[:, 4])

        loss = lossC + (0.0386*lossR1 + 0.0405*lossR2 + 0.0629*lossR3 + 0.0877*lossR4)/4
        # Why 0.0386, 0.0405, 0.06029 and 0.0877: these fours are used to normalize four body biometrics
        # fat/muscle/water/bone rates, for example, if looking paper Table 6, where we showed the information of 
        # 30 recruited subjects. The minimal fat rate is 5, the maximum is 30.9, we decided to normarlize the fat rate
        # by dividing (31-5), resulting in 0.0386.  0.0405->[65,90]muscle rate, 0.0629->[49,65]water rate, 0.0877->[1.5 13.0]  
        # We doing this was spired by Faster RCNN loss, which has a object classfication and a bounding box regression. As paper
        # said, they normalized the regression loss.
        #
        # print(loss.item())
        # loss_every += loss.item()
        loss.backward()
        optimizer.step()
# #
    resnet.eval()
    correct_t = 0
    for (samples, labels) in tqdm(train_data_loader):
        with torch.no_grad():
            samplesV = Variable(samples.cuda())
            labelsV = Variable(labels.cuda())
            # labelsV = labelsV.view(-1)

            predict_label = resnet(samplesV)
            prediction = predict_label[0].data.max(1)[1]
            correct_t += prediction.eq(labelsV[:, 0].data.long()).sum()

    print("Training accuracy:", (100*float(correct_t)/num_train_instances))

    trainacc = str(100*float(correct_t)/num_train_instances)[0:6]

    correct_t = 0
    for (samples, labels) in tqdm(test_data_loader):
        with torch.no_grad():
            samplesV = Variable(samples.cuda())
            labelsV = Variable(labels.cuda())
            # labelsV = labelsV.view(-1)

        predict_label = resnet(samplesV)
        prediction = predict_label[0].data.max(1)[1]
        correct_t += prediction.eq(labelsV[:, 0].data.long()).sum()

    print("Test accuracy:", (100 * float(correct_t) / num_test_instances))

    testacc = str(100 * float(correct_t) / num_test_instances)[0:6]

    torch.save(resnet, 'weights/resnet18_Train' + trainacc + 'Test' + testacc + '.pkl')









#

#
# batch_size = 20
# # load data
# data = sio.loadmat('../dataset/test.mat')
# test_data = data['test_data']
# # test_label = data['test_label_bin']
# test_label = data['test_label']
#
# test_label = test_label - 1
#
# # prepare data, nSample x nChannel x width x height
#
# # reshape train data size to nSample x nSubcarrier x 1 x 1
# num_test_instances = len(test_data)
#
# test_data = torch.from_numpy(test_data).type(torch.FloatTensor).view(num_test_instances, 30, 1, 1)
# test_label = torch.from_numpy(test_label).type(torch.FloatTensor)
# test_dataset = TensorDataset(test_data, test_label)
# test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
#
# resnet = torch.load('model/res34.pkl')
# resnet = resnet.cuda()
#
# resnet.eval()
#
# correct = 0
#
# for i, (samples, labels) in enumerate(test_data_loader):
#     with torch.no_grad():
#         samplesV = Variable(samples.cuda())
#         labelsV = Variable(labels.cuda())
#
#     predict_label = resnet(samplesV)
#
#     pred = predict_label.data.max(1)[1]
#     correct += pred.eq(labelsV[:, 0].data.long()).sum()

#     if i == 0:
#         temp = predict_label.data
#         fallen_pred = temp
#
#     elif i > 0:
#         temp = predict_label.data
#         fallen_pred = np.concatenate((fallen_pred, temp), axis=0)
#
# sio.savemat('results/test_result.mat', {'fallen_pred': fallen_pred})

# print('Accuracy:', 100 * correct / num_test_instances)
