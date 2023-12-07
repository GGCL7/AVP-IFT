import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from simple_model.multi_label.feature2 import generate_features
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import time
import pandas as pd
import sys
from model1.BINARY import *
from model1.BLOSUM62 import *
from model1.ZSCALE import *
from get_data import collate, get_prelabel, MyDataSet
from original_data import get_original_data_Anti
from model1.feature_generator import feature_generator
from MyUtils.util_eval import evaluate_accuracy
from MyUtils.util_cal import caculate_metric
from ContrastModel import newModel, ContrastiveLoss
from configuration import config as cf
import torch.nn as nn
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer_encoder(x)


class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()

        d_model = 566  # Assuming the input feature size is 566
        nhead = 2  # Number of self-attention heads
        num_layers = 1  # Number of transformer layers

        self.transformer = TransformerEncoder(d_model, nhead, num_layers)

        self.fc1 = nn.Linear(d_model, 128)
        # self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        # self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x, return_hidden=False):
        x = self.transformer(x)

        x = self.fc1(x)
        # x = self.bn1(x)
        x = torch.relu(x)

        hidden = x  # This is the 128-dimensional hidden layer

        x = self.fc2(x)
        # x = self.bn2(x)
        x = torch.relu(x)

        if return_hidden:
            return x  # This returns the 64-dimensional hidden layer if return_hidden is True

        x = self.fc3(x)

        return x

train_data, train_label, train_seq = feature_generator('/Users/ggcl7/Desktop/balanced/non_AVP/train.txt')
test_data, test_label, test_seq = feature_generator('/Users/ggcl7/Desktop/balanced/non_AVP/test.txt')
train_dataset = MyDataSet(train_data, train_label, train_seq)
test_dataset = MyDataSet(test_data, test_label, test_seq)
# print(train_data.shape)
# print(test_data.shape)
batch_size = 64
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



train_df = generate_features('/Users/ggcl7/Desktop/balanced/non_AVP/train.txt')
test_df = generate_features('/Users/ggcl7/Desktop/balanced/non_AVP/test.txt')
X_train = torch.tensor(train_df.iloc[:, 1:].values, dtype=torch.float32)
y_train = torch.tensor(train_df['Label'].values, dtype=torch.long)
X_test = torch.tensor(test_df.iloc[:, 1:].values, dtype=torch.float32)
y_test = torch.tensor(test_df['Label'].values, dtype=torch.long)
train_dataset2 = TensorDataset(X_train, y_train)
train_loader2 = DataLoader(train_dataset2, batch_size=32, shuffle=False)
test_dataset2 = TensorDataset(X_test, y_test)
test_loader2 = DataLoader(test_dataset2, batch_size=32, shuffle=False)





device = torch.device("cpu")
model1 = newModel().to(device)
checkpoint1 = torch.load('balanced1.pl', map_location=torch.device('cpu'))
model1.load_state_dict(checkpoint1['model'])

model1.eval()  # 设置模型为评估模式

model2 = SimpleClassifier().to(device)
checkpoint2 = torch.load('best_model_balanced_avp.pl')
model2.load_state_dict(checkpoint2['model_state_dict'])
model2.eval()


hidden_features_list_train = []
with torch.no_grad():
    for x, y, z in train_iter:
        x, y = x.to(device), y.to(device)
        hidden_features = model1.forward(x)  # 获取64维度的隐藏层特征
        hidden_features_list_train.append(hidden_features)

all_hidden_features_train = torch.cat(hidden_features_list_train, dim=0)
print(all_hidden_features_train.shape)

hidden_features_list_test = []
with torch.no_grad():
    for x, y, z in test_iter:
        x, y = x.to(device), y.to(device)
        hidden_features = model1.forward(x)  # 获取64维度的隐藏层特征
        hidden_features_list_test.append(hidden_features)

all_hidden_features_test = torch.cat(hidden_features_list_test, dim=0)
print(all_hidden_features_test.shape)

hidden_features_list2_train = []
with torch.no_grad():
    for data, target in train_loader2:
        hidden_features = model2(data, return_hidden=True)  # Get the 64-dimensional hidden features
        hidden_features_list2_train.append(hidden_features)

all_hidden_features2_train = torch.cat(hidden_features_list2_train, dim=0)
print(all_hidden_features2_train.shape)

hidden_features_list2_test = []
with torch.no_grad():
    for data, target in test_loader2:
        hidden_features = model2(data, return_hidden=True)  # Get the 64-dimensional hidden features
        hidden_features_list2_test.append(hidden_features)

all_hidden_features2_test = torch.cat(hidden_features_list2_test, dim=0)
# print(all_hidden_features2_test.shape)

combined_train_features = torch.cat((all_hidden_features_train, all_hidden_features2_train), dim=1)
print(combined_train_features.shape)
combined_test_features = torch.cat((all_hidden_features_test, all_hidden_features2_test), dim=1)
print(combined_test_features.shape)

class CombinedClassifier(nn.Module):
    def __init__(self):
        super(CombinedClassifier, self).__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

combined_model = CombinedClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(combined_model.parameters(), lr=0.001)

combined_train_dataset = TensorDataset(combined_train_features, train_label)
combined_train_loader = DataLoader(combined_train_dataset, batch_size=64, shuffle=True)




checkpoint = torch.load('best_concate_model_balanced1-2.pl')
combined_model.load_state_dict(checkpoint['model'])
combined_model.eval()  # 设置为评估模式

# 3. 使用模型进行预测并评估
with torch.no_grad():
    outputs = combined_model(combined_test_features)
    _, predicted = torch.max(outputs, 1)

    accuracy = accuracy_score(test_label, predicted)
    sensitivity = recall_score(test_label, predicted)
    precision = precision_score(test_label, predicted)
    f1 = f1_score(test_label, predicted)
    auc = roc_auc_score(test_label, outputs[:, 1])
    mcc = matthews_corrcoef(test_label, predicted)

    tn, fp, fn, tp = confusion_matrix(test_label, predicted).ravel()
    specificity = tn / (tn + fp)

    metric1 = {
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "F1 Score": f1,
        "AUC": auc,
        "MCC": mcc
    }

    print(f"Metrics after loading model: {metric1}")
