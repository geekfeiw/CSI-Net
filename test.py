import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import pandas as pd
from model/res_net_use_this import ResNet, Bottleneck

#
batch_size = 100

# prepare data, nSample x nChannel x width x height
data = sio.loadmat('data/test.mat')
test_data = data['test_data']
test_label = data['test_label']
test_label[:, 0] = test_label[:, 0] - 1

num_test_instances = len(test_data)

test_data = torch.from_numpy(test_data).type(torch.FloatTensor).view(num_test_instances, 30, 1, 1)
test_label = torch.from_numpy(test_label).type(torch.FloatTensor)
test_dataset = TensorDataset(test_data, test_label)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

resnet = torch.load('weights/res18_Train100.0Test93.001.pkl')
resnet = resnet.cuda()

resnet.eval()

correct = 0

for i, (samples, labels) in enumerate(test_data_loader):
    with torch.no_grad():
        samplesV = Variable(samples.cuda())
        labelsV = Variable(labels.cuda())

        predict_label = resnet(samplesV)

    pred = predict_label[0][:, 0:30].data.max(1)[1]
    correct += pred.eq(labelsV[:, 0].data.long()).sum()

    if i == 0:
        tempC = predict_label[0].data
        tempR = predict_label[1].data
        temp = np.concatenate((tempC, tempR), axis=1)
        human_prediction = temp

    elif i > 0:
        tempC = predict_label[0].data
        tempR = predict_label[1].data
        temp = np.concatenate((tempC, tempR), axis=1)
        human_prediction = np.concatenate((human_prediction, temp), axis=0)

print('Accuracy:', 100 * correct / len(test_data_loader.dataset))
sio.savemat('results/test_93.mat', {'human_prediction': human_prediction})

