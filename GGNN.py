import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GatedGraphConv
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# 加载本地 Cora 数据集文件
content_path = 'cve.content'
cites_path = 'cve.cites'

# 1. 读取 content 文件
content_df = pd.read_csv(content_path, sep='\t', header=None)

# 去除ID前后的空格
paper_ids = content_df[0].str.strip().values
features = torch.tensor(content_df.iloc[:, 1:-1].values, dtype=torch.float)
labels = pd.factorize(content_df.iloc[:, -1].str.strip())[0]  # 将文本标签转为整数编码
labels = torch.tensor(labels, dtype=torch.long)

# 创建 paper_id 到索引的映射，去除ID前后的空格
paper_id_map = {id: i for i, id in enumerate(paper_ids)}

# 2. 读取 cites 文件
cites_df = pd.read_csv(cites_path, sep='\t', header=None)

# 去除引用对中ID前后的空格
cites_df[0] = cites_df[0].str.strip()
cites_df[1] = cites_df[1].str.strip()

# 创建边索引
edge_index = []
for cited, citing in zip(cites_df[0], cites_df[1]):
    if cited in paper_id_map and citing in paper_id_map:
        edge_index.append([paper_id_map[cited], paper_id_map[citing]])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# 3. 创建 PyTorch Geometric 数据对象
data = Data(x=features, edge_index=edge_index, y=labels)

# 4. 设置训练、验证和测试掩码
num_nodes = data.num_nodes

# 计算分割点
train_split = int(num_nodes * 0.8)
val_split = int(num_nodes * 0.9)

# 初始化掩码
data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

# 设置掩码
data.train_mask[:train_split] = True
data.val_mask[train_split:val_split] = True
data.test_mask[val_split:] = True

# 5. 定义 GGNN 模型
class GGNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, hidden_channels):
        super(GGNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        # 输入层到隐藏层
        self.lin0 = nn.Linear(in_channels, hidden_channels)

        # GatedGraphConv 层
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GatedGraphConv(hidden_channels, num_layers))

        # 隐藏层到输出层
        self.lin1 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # 输入层到隐藏层
        x = self.lin0(x).relu()
        for conv in self.convs:
            x = conv(x, edge_index)
        # 隐藏层到输出层
        x = self.lin1(x)
        return x

# 6. 模型初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GGNN(in_channels=data.num_node_features, out_channels=len(labels.unique()), num_layers=3, hidden_channels=64).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.004, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

# 7. 训练模型
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# 8. 测试模型
def test(mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred[mask] == data.y[mask]).sum()
    acc = int(correct) / int(mask.sum())
    return acc

# 计算精确率、召回率和 F1 分数
def calculate_metrics(mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1).cpu().numpy()
    true = data.y.cpu().numpy()
    mask = mask.cpu().numpy()

    # 使用掩码过滤相应数据
    pred = pred[mask]
    true = true[mask]

    precision = precision_score(true, pred, average='weighted')
    recall = recall_score(true, pred, average='weighted')
    f1 = f1_score(true, pred, average='weighted')
    return precision, recall, f1

# 9. 运行训练和测试
for epoch in range(100):
    loss = train()
    train_acc = test(data.train_mask)
    val_acc = test(data.val_mask)
    test_acc = test(data.test_mask)
    precision, recall, f1 = calculate_metrics(data.test_mask)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

# 10. 对所有数据进行预测并输出混淆矩阵
model.eval()
out = model(data.x, data.edge_index)
pred_all = out.argmax(dim=1).cpu().numpy()
true_all = data.y.cpu().numpy()

# 计算混淆矩阵
conf_matrix = confusion_matrix(true_all, pred_all)
print("Confusion Matrix:")
print(conf_matrix)
