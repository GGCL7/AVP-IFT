import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix


import pandas as pd
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from FEM_feature import generate_features
from torch.utils.data import DataLoader, TensorDataset
import torch
from metrics import caculate_metric, evaluate_accuracy
from get_data import collate, get_prelabel, MyDataSet
from feature_generator import feature_generator
from ContrastModel import newModel, ContrastiveLoss
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

train_data, train_label, train_seq = feature_generator('/Users/ggcl7/Desktop/balanced/non_AMP/train.txt')
test_data, test_label, test_seq = feature_generator('/Users/ggcl7/Desktop/balanced/non_AMP/test.txt')
train_dataset = MyDataSet(train_data, train_label, train_seq)
test_dataset = MyDataSet(test_data, test_label, test_seq)

batch_size = 64
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
train_iter_cont = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              shuffle=True, collate_fn=collate)


train_df = generate_features('/Users/ggcl7/Desktop/balanced/non_AMP/train.txt')
test_df = generate_features('/Users/ggcl7/Desktop/balanced/non_AMP/test.txt')
X_train = torch.tensor(train_df.iloc[:, 1:].values, dtype=torch.float32)
y_train = torch.tensor(train_df['Label'].values, dtype=torch.long)
X_test = torch.tensor(test_df.iloc[:, 1:].values, dtype=torch.float32)
y_test = torch.tensor(test_df['Label'].values, dtype=torch.long)
train_dataset2 = TensorDataset(X_train, y_train)
train_loader2 = DataLoader(train_dataset2, batch_size=64, shuffle=False)
test_dataset2 = TensorDataset(X_test, y_test)
test_loader2 = DataLoader(test_dataset2, batch_size=64, shuffle=False)

#
device = torch.device("cpu")
model = newModel().to(device)
model_outputs = []
true_labels_batch = []
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
contrastive_loss_fn = ContrastiveLoss()
cross_entropy_loss_fn = nn.CrossEntropyLoss(reduction='sum')
best_acc = 0
EPOCH = 100

for epoch in range(EPOCH):
    total_loss_values = []
    contrastive_loss_values = []
    cross_entropy_loss_values = []

    model.train()

    for sequence_1, sequence_2, binary_label, label_1, label_2, additional_data_1, additional_data_2 in train_iter_cont:  # More descriptive names

        embedding_1 = model(sequence_1)
        embedding_2 = model(sequence_2)
        class_output_1 = model.trainModel(sequence_1)
        class_output_2 = model.trainModel(sequence_2)
        contrastive_loss = contrastive_loss_fn(embedding_1, embedding_2, binary_label)
        class_loss_1 = cross_entropy_loss_fn(class_output_1, label_1)
        class_loss_2 = cross_entropy_loss_fn(class_output_2, label_2)
        total_loss = contrastive_loss + class_loss_1 + class_loss_2

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_loss_values.append(total_loss.item())
        contrastive_loss_values.append(contrastive_loss.item())
        cross_entropy_loss_values.append((class_loss_1 + class_loss_2).item())
        model_outputs.extend([embedding_1, embedding_2])
        true_labels_batch.extend([label_1, label_2])

    model.eval()
    with torch.no_grad():
        train_accuracy = evaluate_accuracy(train_iter, model)
        test_accuracy = evaluate_accuracy(test_iter, model)

        predicted_labels, true_labels = get_prelabel(test_iter, model)
        predicted_labels = np.concatenate(predicted_labels).reshape(-1, 1)
        true_labels = np.concatenate(true_labels).reshape(-1, 1)

        predicted_labels_df = pd.DataFrame(predicted_labels, columns=['PredictedLabel'])
        true_labels_df = pd.DataFrame(true_labels, columns=['TrueLabel'])
        combined_df = pd.concat([predicted_labels_df, true_labels_df], axis=1)

        softmax_outputs = []
        for batch_features, _, _ in test_iter:
            batch_features = batch_features.to(device)
            softmax_output = torch.softmax(model.trainModel(batch_features), dim=1)
            softmax_outputs.append(softmax_output.cpu().numpy())
        softmax_outputs = np.concatenate(softmax_outputs, axis=0)

        predicted_probabilities = softmax_outputs[:, 1].reshape(-1)
        probabilities_df = pd.DataFrame(predicted_probabilities, columns=['PredictedProbability'])

        final_df = pd.concat([combined_df, probabilities_df], axis=1)

        evaluation_metric, roc_curve_data, precision_recall_curve_data = caculate_metric(final_df['PredictedLabel'],
                                                                                           final_df['TrueLabel'],
                                                                                           final_df[
                                                                                               'PredictedProbability'])
    if test_accuracy > best_acc:
        best_acc = test_accuracy
        torch.save({"best_acc": best_acc,"metric":evaluation_metric, "model": model.state_dict()}, f'module1.pl')
        print(f"best_acc: {best_acc},metric:{evaluation_metric}")

 # 设置模型为评估模式

model2 = SimpleClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for epoch in range(EPOCH):
    # Training loop remains unchanged
    for batch_idx, (data, target) in enumerate(train_loader2):
        optimizer2.zero_grad()
        outputs = model2(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer2.step()

    # Modified evaluation loop to use test_loader
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader2:
            test_outputs = model2(data)
            _, predicted = torch.max(test_outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_acc = 100 * correct / total
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save({
            "best_acc": best_acc,
            "model_state_dict": model.state_dict(),
        }, f'module2.pl')
        print(f"New best accuracy: {best_acc:.2f}%")
print(f"Final best accuracy: {best_acc:.2f}%")
model.eval()
model2.eval()
hidden_features_list_train = []
with torch.no_grad():
    for x, y, z in train_iter:
        x, y = x.to(device), y.to(device)
        hidden_features = model.forward(x)
        hidden_features_list_train.append(hidden_features)

all_hidden_features_train = torch.cat(hidden_features_list_train, dim=0)
print(all_hidden_features_train.shape)

hidden_features_list_test = []
with torch.no_grad():
    for x, y, z in test_iter:
        x, y = x.to(device), y.to(device)
        hidden_features = model.forward(x)
        hidden_features_list_test.append(hidden_features)

all_hidden_features_test = torch.cat(hidden_features_list_test, dim=0)
print(all_hidden_features_test.shape)

hidden_features_list2_train = []
with torch.no_grad():
    for data, target in train_loader2:
        hidden_features = model2(data, return_hidden=True)
        hidden_features_list2_train.append(hidden_features)

all_hidden_features2_train = torch.cat(hidden_features_list2_train, dim=0)
print(all_hidden_features2_train.shape)

hidden_features_list2_test = []
with torch.no_grad():
    for data, target in test_loader2:
        hidden_features = model2(data, return_hidden=True)
        hidden_features_list2_test.append(hidden_features)

all_hidden_features2_test = torch.cat(hidden_features_list2_test, dim=0)


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



best_acc = 0.0


for epoch in range(EPOCH):
    for data, target in combined_train_loader:
        optimizer.zero_grad()
        outputs = combined_model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

    combined_model.eval()
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

    if accuracy > best_acc:
        best_acc = accuracy
        torch.save({
            "best_acc": best_acc,
            "metric": metric1,
            "model": combined_model.state_dict()
        }, f'module3.pl')
        print(f"best_acc: {best_acc:.4f}, metric: {metric1}")


