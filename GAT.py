import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
import random
import scipy.sparse as sp
from sklearn.metrics import confusion_matrix
import csv
from sklearn.metrics import confusion_matrix
import pynvml
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_score, recall_score, f1_score

pynvml.nvmlInit()
device_count = pynvml.nvmlDeviceGetCount()

start_time = time.time()


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_data(path="../data/", dataset="cve"):
    """Load citation network data (e.g., Cora)"""
    # print('Loading {} dataset...'.format(dataset))
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))  # Use numpy to read .txt file
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)  # Get feature matrix
    labels = encode_onehot(idx_features_labels[:, -1])  # Get labels

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    # 20
    # idx_train = range(130)
    # idx_val = range(130, 170)
    # idx_test = range(170, 215)

    # 40
    # idx_train = range(260)
    # idx_val = range(260, 345)
    # idx_test = range(345, 430)
    # #
    # 60
    # idx_train = range(390)
    # idx_val = range(390, 515)
    # idx_test = range(515, 645)
    #
    # 80
    # idx_train = range(505)
    # idx_val = range(505, 675)
    # idx_test = range(675, 843)


    # # 100
    idx_train = range(645)
    idx_val = range(645, 859)
    idx_test = range(859, 1073)

    features = torch.FloatTensor(np.array(features.todense()))
    # print(features)
    labels = torch.LongTensor(np.where(labels)[1])
    adj = torch.FloatTensor(np.array(adj.todense()))
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    # print(adj)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test




class GATLayer(nn.Module):
    """GAT Layer"""

    def __init__(self, input_feature, output_feature, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.alpha = alpha
        self.dropout = dropout
        self.concat = concat
        self.a = nn.Parameter(torch.empty(size=(2 * output_feature, 1)))
        self.w = nn.Parameter(torch.empty(size=(input_feature, output_feature)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.w)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # Replace values at adj > 0 positions with e values, others with -9e15
        attention = F.softmax(attention, dim=1)  # Apply softmax row-wise
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.mm(attention, Wh)  # Get input for the next layer

        if self.concat:
            return F.elu(h_prime)  # Apply activation
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):

        Wh1 = torch.matmul(Wh, self.a[:self.output_feature, :])  # N*out_size @ out_size*1 = N*1

        Wh2 = torch.matmul(Wh, self.a[self.output_feature:, :])  # N*1
        e = Wh1 + torch.transpose(Wh2, 0, 1)

        # e = Wh1 + Wh2.T  # Add each element of Wh1 with all elements of Wh2.T to form an N*N matrix
        return self.leakyrelu(e)


class GAT(nn.Module):
    """GAT Model"""

    def __init__(self, input_size, hidden_size, output_size, dropout, alpha, nheads, concat=True):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attention = [GATLayer(input_size, hidden_size, dropout=dropout, alpha=alpha, concat=True) for _ in
                          range(nheads)]
        for i, attention in enumerate(self.attention):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GATLayer(hidden_size * nheads, output_size, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attention], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))

        return F.log_softmax(x, dim=1)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    # loss_train = F.cross_entropy(output[idx_train], labels[idx_train])

    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    # if not args.fastmode:
    model.eval()
    output = model(features, adj)

    acc_val = accuracy(output[idx_val], labels[idx_val])
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return loss_val.data.item()




# Calculate confusion matrix during the testing phase
def calculate_confusion_matrix(output, labels):
    # Convert probability values to predicted results
    predictions = output.argmax(dim=1)

    # Get the number of classes
    num_classes = output.shape[1]

    # Initialize confusion matrix as a zero matrix
    confusion_matrix = torch.zeros(num_classes, num_classes)

    # Compute cross frequencies in the confusion matrix
    for i in range(len(labels)):
        confusion_matrix[labels[i], predictions[i]] += 1

    return confusion_matrix


def find_misclassified_samples(output, labels):
    # Convert probability values to predicted results
    predictions = output.argmax(dim=1)

    # Find indices of misclassified samples
    misclassified_indices = (predictions != labels).nonzero().squeeze()

    return misclassified_indices


def find_tp_fp_tn_fn_indices(output, labels, class_index):
    # Convert probability values to predicted results
    predictions = output.argmax(dim=1)

    # Find indices of true positive samples
    tp_indices = (predictions == class_index) & (labels == class_index)

    # Find indices of false positive samples
    fp_indices = (predictions == class_index) & (labels != class_index)

    # Find indices of true negative samples
    tn_indices = (predictions != class_index) & (labels != class_index)

    # Find indices of false negative samples
    fn_indices = (predictions != class_index) & (labels == class_index)

    return tp_indices.nonzero().squeeze(), fp_indices.nonzero().squeeze(), tn_indices.nonzero().squeeze(), fn_indices.nonzero().squeeze()


def compute_test():

    pynvml.nvmlShutdown()

    model.eval()
    output = model(features, adj)

    preds = output[idx_test].max(1)[1].type_as(labels)
    y_pred = np.array(preds)
    y_test = np.array(labels[idx_test])


    # Get confusion matrix
    confusion_matrix = calculate_confusion_matrix(output, labels)
    # Print confusion matrix
    print("Confusion Matrix All:")
    print(confusion_matrix)

    # Compute accuracy, precision, recall, and F1-Score for each class
    num_classes = output.shape[1]
    for class_index in range(num_classes):
        tp_indices, fp_indices, tn_indices, fn_indices = find_tp_fp_tn_fn_indices(output, labels, class_index)

        # Compute accuracy
        tp = torch.sum(tp_indices).item()
        fp = torch.sum(fp_indices).item()
        tn = torch.sum(tn_indices).item()
        fn = torch.sum(fn_indices).item()
        accuracy_class = (tp + tn) / (tp + fp + tn + fn)

        # Compute precision
        if tp + fp == 0:
            precision_class = 0
        else:
            precision_class = tp / (tp + fp)

        # Compute recall
        if tp + fn == 0:
            recall_class = 0
        else:
            recall_class = tp / (tp + fn)

        # Compute F1-Score
        if precision_class + recall_class == 0:
            f1_score_class = 0
        else:
            f1_score_class = 2 * (precision_class * recall_class) / (precision_class + recall_class)

        # Print performance metrics for each class
        print("Class {}: Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, F1-Score: {:.3f}".format(class_index,
                                                                                                       accuracy_class,
                                                                                                       precision_class,
                                                                                                       recall_class,
                                                                                                       f1_score_class))

    misclassified_indices = find_misclassified_samples(output, labels)

    # Print misclassified sample indices
    print("Misclassified Samples Index: ", misclassified_indices)


    loss_test = F.nll_loss(output[idx_test], labels[idx_test].cuda())
    acc_test = accuracy(output[idx_test], labels[idx_test].cuda())



    # Precision
    precision_result = precision_score(y_test, y_pred, labels=None, pos_label=1, average='macro', sample_weight=None)
    print("precision:%s" % precision_result)
    print("Test Precision: {:.3f}%".format(precision_result * 100))

    # Recall
    recall_result = recall_score(y_test, y_pred, labels=None, pos_label=1, average='macro', sample_weight=None)
    print("recall:%s" % recall_result)
    print("Test Recall: {:.3f}%".format(recall_result * 100))

    # F1-Score
    f1_score_result = f1_score(y_test, y_pred, labels=None, pos_label=1, average='macro', sample_weight=None)
    print("f1_score:%s" % f1_score_result)
    print("Test F1-Score: {:.3f}%".format(f1_score_result * 100))
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))
    end_time = time.time()
    t_total = end_time - start_time
    print("Total time elapsed: {:.4f}s".format(t_total))

    print("{:.4f}, {:.4f}".format(loss_test.data.item(), acc_test.data.item()))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--hidden', type=int, default=128, help='hidden size')
    parser.add_argument('--epochs', type=int, default=800, help='Number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--nheads', type=int, default=8, help='Number of head attentions')
    parser.add_argument('--dropout', type=float, default=0.01,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.9, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=150, help='Patience')
    parser.add_argument('--seed', type=int, default=42, help='Seed number')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    adj, features, labels, idx_train, idx_val, idx_test = load_data()
    model = GAT(input_size=features.shape[1],
                hidden_size=args.hidden,
                output_size=int(labels.max()) + 1,
                dropout=args.dropout,
                nheads=args.nheads,
                alpha=args.alpha)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    # print(labels)
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = 1000 + 1
    best_epoch = 0

    for epoch in range(args.epochs):
        train(epoch)
        loss_values.append(train(epoch))
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    compute_test()

print("Optimization Finished!")
