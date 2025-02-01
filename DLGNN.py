import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


with open('cve.cites', 'r') as f:
    cites = f.readlines()


with open('cve.content', 'r') as f:
    content = f.readlines()


papers = []
features = []
labels = []
for line in content:
    parts = line.strip().split()
    papers.append(parts[0])  # paper_id
    features.append([float(x) for x in parts[1:-1]])  # word_attributes
    labels.append(parts[-1])  # label

features = np.array(features)
labels = np.array(labels)


features = torch.tensor(features, dtype=torch.float)
labels, label_map = pd.factorize(labels)
labels = torch.tensor(labels, dtype=torch.long)


paper_idx_map = {paper: idx for idx, paper in enumerate(papers)}


edge_index = []
for line in cites:
    paper1, paper2 = line.strip().split()
    if paper1 in paper_idx_map and paper2 in paper_idx_map:
        edge_index.append([paper_idx_map[paper1], paper_idx_map[paper2]])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)



data = Data(x=features, edge_index=edge_index, y=labels)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(num_node_features=features.shape[1], num_classes=len(label_map)).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()


def test():
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        loss = F.nll_loss(out, data.y).item()
        correct = pred.eq(data.y).sum().item()
        acc = correct / data.num_nodes

        # Convert to numpy for sklearn metrics
        y_true = data.y.cpu().numpy()
        y_pred = pred.cpu().numpy()

        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_true, y_pred)

    return loss, acc, precision, recall, f1, conf_matrix



for epoch in range(10):
    train_loss = train()
    test_loss, acc, precision, recall, f1, conf_matrix = test()
    print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, '
          f'Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

print("Confusion Matrix:")
print(conf_matrix)
