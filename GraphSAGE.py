import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
import time


from sklearn.metrics import precision_score, recall_score, f1_score
import pynvml

pynvml.nvmlInit()
device_count = pynvml.nvmlDeviceGetCount()


class pyGraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(pyGraphSAGE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.04)

    def forward(self, features, adj_matrix):
        x = self.fc1(features)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = torch.spmm(adj_matrix, x)
        return output


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

start_time = time.time()

idx_train = range(50)
idx_val = range(51, 100)
idx_test = range(101, 200)

def load_data(path="../data/", dataset="cve"):


    print('Loading {} dataset...'.format(dataset))
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    raw_features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj_matrix = sparse_mx_to_torch_sparse_tensor(adj)
    # print(adj_matrix)
    return raw_features, labels, adj_matrix


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


criterion = nn.CrossEntropyLoss()


def train(model, features, adj_matrix, labels, idx_train, idx_val, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.004)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj_matrix)
        loss_train = criterion(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

        loss_train.backward()
        optimizer.step()

        model.eval()
        output_val = model(features, adj_matrix)
        loss_val = criterion(output_val[idx_val], labels[idx_val])
        acc_val = accuracy(output_val[idx_val], labels[idx_val])

        print('Epoch: {:03d}, Loss: {:.4f}, Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'
              .format(epoch + 1, loss_train.item(), acc_train.item(), loss_val.item(), acc_val.item()))


def accuracy(output, labels):
    _, preds = torch.max(output, dim=1)
    correct = torch.eq(preds, labels).float()
    acc = correct.sum() / len(correct)
    return acc


def evaluate_test(model, features, adj_matrix, labels, idx_test):
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)


    pynvml.nvmlShutdown()

    model.eval()
    output = model(features, adj_matrix)
    loss_test = criterion(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    preds = output[idx_test].max(1)[1].type_as(labels)
    # print(len(preds))
    # print(len(labels[idx_test]))
    # print(np.array(preds))
    # print(np.array(labels[idx_test]))
    y_pred = np.array(preds)
    y_test = np.array(labels[idx_test])
    # f1_score = calculate_f1_score(y_test, y_pred)
    # print("F1 Score:", f1_score)
    # sklearn.metrics.precision_score(y_test, y_pred, labels=None, pos_label=1,
    #                                 average='binary', sample_weight=None)


    percision_result = precision_score(y_test, y_pred, labels=None, pos_label=1, average='macro',
                                       sample_weight=None)



torch.manual_seed(0)
np.random.seed(0)


train_features, train_labels, train_adj_matrix = load_data(dataset="cve")
val_features, val_labels, val_adj_matrix = load_data(dataset="cve")
test_features, test_labels, test_adj_matrix = load_data(dataset="cve")


input_dim = train_features.shape[1]
hidden_dim = 32
output_dim = train_labels.max().item() + 1



model = pyGraphSAGE(input_dim, hidden_dim, output_dim)




epochs =10
train(model, train_features, train_adj_matrix, train_labels, idx_train, idx_val, epochs)
end_time = time.time()
t_total = end_time - start_time


test_loss, test_accuracy = evaluate_test(model, test_features, test_adj_matrix, test_labels, idx_test)

print('Test set results: loss= {:.4f} accuracy= {:.4f}'.format(test_loss, test_accuracy))
print("Total time elapsed: {:.4f}s".format(t_total))
