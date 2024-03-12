import torch
import torch.nn as nn
import torch.nn.functional as F
pep2label = list()

import sys
print(sys.path)

device = torch.device("cpu")  # <-- Change here


import torch.nn as nn
import torch.nn.functional as F


import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F

class newModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.filter_sizes = [1, 2, 3, 4, 6, 8, 16, 32]
        filter_num = 64

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filter_num, (fsz, 45)) for fsz in self.filter_sizes])
        self.dropout = nn.Dropout(0.5)

        # Bi-directional LSTM layer
        self.lstm = nn.LSTM(input_size=len(self.filter_sizes) * filter_num,
                            hidden_size=128,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)

        # Adjusting the linear layer's input size to account for BiLSTM
        self.block1 = nn.Sequential(nn.Linear(256, 256),  # 2 * hidden_size for BiLSTM
                                    nn.BatchNorm1d(256),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, 64),
                                    )

        self.classification = nn.Sequential(
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = x.unsqueeze(1)

        x = [F.relu(conv(x)) for conv in self.convs]
        x = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in x]
        x = [x_item.view(x_item.size(0), -1) for x_item in x]
        x = torch.cat(x, 1)

        # Reshape for LSTM
        x = x.view(x.size(0), 1, -1)  # (batch_size, seq_len, features)

        # Pass through LSTM
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Take the output from the last time step

        x = self.dropout(x)
        output = self.block1(x)

        return output

    def trainModel(self, x):
        output = self.forward(x)
        return self.classification(output)


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # euclidean_distance: [128]
        # print('label.shape', label.shape)
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +  # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive
