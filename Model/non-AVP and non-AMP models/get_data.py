import torch
import os
from configuration import config as cf
import torch.utils.data as Data

def get_prelabel(data_iter, net):
    config = cf.get_train_config()
    device = torch.device("cpu")
    prelabel, relabel = [], []
    for x, y, z in data_iter:
        x, y = x.to(device), y.to(device)
        outputs = net.trainModel(x)
        prelabel.append(outputs.argmax(dim=1).cpu().numpy())
        relabel.append(y.cpu().numpy())
        # print()
    return prelabel, relabel



def collate(batch):
    config = cf.get_train_config()
    device = torch.device("cpu")
    seq1_ls = []
    seq2_ls = []
    label1_ls = []
    label2_ls = []
    label_ls = []
    pep1_ls = []
    pep2_ls = []
    batch_size = len(batch)
    for i in range(int(batch_size / 2)):
        seq1, label1, pep_seq1 = batch[i][0], batch[i][1], batch[i][2]
        seq2, label2, pep_seq2 = batch[i + int(batch_size / 2)][0], batch[i + int(batch_size / 2)][1], batch[i + int(batch_size / 2)][2]
        label1_ls.append(label1.unsqueeze(0))
        label2_ls.append(label2.unsqueeze(0))
        pep1_ls.append(pep_seq1)
        pep2_ls.append(pep_seq2)
        label = (label1 ^ label2)
        seq1_ls.append(seq1.unsqueeze(0))
        seq2_ls.append(seq2.unsqueeze(0))
        label_ls.append(label.unsqueeze(0))
    seq1 = torch.cat(seq1_ls).to(device)
    seq2 = torch.cat(seq2_ls).to(device)
    label = torch.cat(label_ls).to(device)
    label1 = torch.cat(label1_ls).to(device)
    label2 = torch.cat(label2_ls).to(device)
    return seq1, seq2, label, label1, label2, pep1_ls, pep2_ls

class MyDataSet(Data.Dataset):
    def __init__(self, data, label, seq):
        self.data = data
        self.label = label
        self.seq = seq

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.seq[idx]
