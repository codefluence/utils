
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from pytorch_lightning import LightningModule

def __init__(self, num_features):

    super(RegimeTrader, self).__init__()

    hidden_size = 1024
    num_targets = 1

    cha_1 = 256//8
    cha_2 = 256//4
    cha_3 = 256//4

    cha_1_reshape = int(hidden_size/cha_1)
    cha_po_1 = int(hidden_size/cha_1/2)
    cha_po_2 = int(hidden_size/cha_1/2/2) * cha_3

    self.cha_1 = cha_1
    self.cha_2 = cha_2
    self.cha_3 = cha_3
    self.cha_1_reshape = cha_1_reshape
    self.cha_po_1 = cha_po_1
    self.cha_po_2 = cha_po_2

    self.batch_norm1 = nn.BatchNorm1d(num_features)
    self.dropout1 = nn.Dropout(0.1)
    self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

    self.batch_norm_c1 = nn.BatchNorm1d(cha_1)
    self.dropout_c1 = nn.Dropout(0.1)
    self.conv1 = nn.utils.weight_norm(nn.Conv1d(cha_1,cha_2, kernel_size = 5, stride = 1, padding=2,  bias=False),dim=None)

    self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size = cha_po_1)

    self.batch_norm_c2 = nn.BatchNorm1d(cha_2)
    self.dropout_c2 = nn.Dropout(0.1)
    self.conv2 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_2, kernel_size = 3, stride = 1, padding=1, bias=True),dim=None)

    self.batch_norm_c2_1 = nn.BatchNorm1d(cha_2)
    self.dropout_c2_1 = nn.Dropout(0.3)
    self.conv2_1 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_2, kernel_size = 3, stride = 1, padding=1, bias=True),dim=None)

    self.batch_norm_c2_2 = nn.BatchNorm1d(cha_2)
    self.dropout_c2_2 = nn.Dropout(0.2)
    self.conv2_2 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_3, kernel_size = 5, stride = 1, padding=2, bias=True),dim=None)

    self.max_po_c2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

    self.flt = nn.Flatten()

    # self.batch_norm3 = nn.BatchNorm1d(cha_po_2)
    # self.dropout3 = nn.Dropout(0.2)
    # self.dense3 = nn.utils.weight_norm(nn.Linear(cha_po_2, num_targets))

    self.linear = nn.Linear(512, num_targets)

def forward(self, x):

    x = self.batch_norm1(x)
    x = self.dropout1(x)
    x = F.celu(self.dense1(x), alpha=0.06)

    x = x.reshape(x.shape[0],self.cha_1, self.cha_1_reshape)

    x = self.batch_norm_c1(x)
    x = self.dropout_c1(x)
    x = F.relu(self.conv1(x))

    x = self.ave_po_c1(x)

    x = self.batch_norm_c2(x)
    x = self.dropout_c2(x)
    x = F.relu(self.conv2(x))
    x_s = x

    x = self.batch_norm_c2_1(x)
    x = self.dropout_c2_1(x)
    x = F.relu(self.conv2_1(x))

    x = self.batch_norm_c2_2(x)
    x = self.dropout_c2_2(x)
    x = F.relu(self.conv2_2(x))
    x =  x * x_s

    x = self.max_po_c2(x)

    x = self.flt(x)

    # x = self.batch_norm3(x)
    # x = self.dropout3(x)
    # x = self.dense3(x)

    x = self.linear(x)

    return x


