import scipy.io
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import torch.nn as nn

# Base directory containing the folders
base_directory = 'data'


# Function to load and process each .mat file
def load_mat_file(filepath):
    data = scipy.io.loadmat(filepath)
    # Assuming 'inputfeats_ph' is the key
    val = data['inputfeats_ph']
    # Extract real and imaginary parts
    R_real = np.real(val)
    R_imag = np.imag(val)
    # Concatenate real and imaginary parts
    features = np.concatenate((R_real, R_imag), axis=-1)
    # Normalize the features
    norm = np.linalg.norm(features, ord=2, axis=1, keepdims=True)
    norm[norm == 0] = 1e-10  # Replace zeros with a small constant to avoid division by zero
    features = features / norm
    return features


# Load all .mat files and extract features
features_list = []
labels = []

folders = ['30', '60', '90']  # Assuming folders correspond to different angles

for folder in folders:
    folder_path = os.path.join(base_directory, folder)
    filepaths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.mat')]

    for filepath in filepaths:
        features = load_mat_file(filepath)
        # Divide features into nodes
        num_nodes = features.shape[0] // 6
        for i in range(num_nodes):
            node_features = features[i * 6:(i + 1) * 6, :]
            features_list.append(node_features.flatten())
            labels.append(folders.index(folder))  # Use the folder name as the label (30, 60, 90)

# Stack features and normalize
features_stack = np.vstack(features_list)
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features_stack)

# Convert data type to float32
normalized_features = normalized_features.astype(np.float32)
print("Updated data type of normalized features:", normalized_features.dtype)

# Convert features to torch tensor
x = torch.tensor(normalized_features, dtype=torch.float)

# Create edges based on the distance between nodes
num_nodes = len(labels)
edges = []
for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        if np.linalg.norm(x[i] - x[j]) < 222.68:
            edges.append((i, j))
            edges.append((j, i))  # Add both directions for an undirected graph

# Ensure edge_index is a two-dimensional tensor
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
print("Edge index shape:", edge_index.shape)

perm = torch.randperm(len(features_stack))
y = torch.tensor(labels, dtype=torch.long)
print(features_stack.shape)
x_shuffled = x[perm]
y_shuffled = y[perm]

data = Data(x=x_shuffled, edge_index=edge_index, y=y_shuffled)

# data = Data(x=x, edge_index=edge_index, y=y)
num_nodes = data.num_nodes
num_train = int(0.8 * num_nodes)

perm = torch.randperm(num_nodes)

# Create train and test masks
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[perm[:num_train]] = True
test_mask[perm[num_train:]] = True

data.train_mask = train_mask
data.test_mask = test_mask

print("Input feature dimension:", data.x.shape[1])
print("Number of nodes:", data.x.shape[0])
print("Number of edges:", data.edge_index.shape)


# Define the GNN model
class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = SAGEConv(data.num_features, 16)
        self.conv2 = SAGEConv(16, len(folders))  # Adjust the output layer based on your specific task
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, len(folders))

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # x = torch.mean(x, dim=0)  # Global pooling
        # x = self.fc1(x)
        return F.log_softmax(x, dim=1)


# Instantiate the model and optimizer
model = GNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Create labels tensor



def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    l1_lambda = 0.001
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    loss += l1_lambda * l1_norm
    loss.backward()
    optimizer.step()
    return loss.item()


def tst():
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        # print(pred)

        # Compute loss and accuracy for test set
        test_loss = F.cross_entropy(out[data.test_mask], data.y[data.test_mask])
        test_acc = accuracy(pred[data.test_mask], data.y[data.test_mask])

        # Compute accuracy for train set (optional, for monitoring)
        train_acc = accuracy(pred[data.train_mask], data.y[data.train_mask])

    return test_loss.item(), test_acc.item(), train_acc.item()



def accuracy(pred, true):
    return (pred == true).float().mean()
test_loss_all = []
train_loss_all = []
test_acc_all = []
train_acc_all = []

for epoch in range(1, 90):
    train_loss = train()
    test_loss, test_acc, train_acc = tst()
    train_loss_all.append(train_loss)
    test_loss_all.append(test_loss)
    test_acc_all.append(test_acc)
    train_acc_all.append(train_acc)
    # if epoch % 2 == 0:
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
        f'Test Loss: {test_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

plt.figure(figsize=(12,8))
plt.plot(np.arange(1, len(test_acc_all) + 1), np.array(test_acc_all)*100, label='Test accuracy', c='orange')
plt.plot(np.arange(1, len(train_acc_all) + 1), np.array(train_acc_all)*100, label='Train accuracy', c='blue')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('GCNConv')
plt.legend(loc='lower right', fontsize='x-large')
plt.savefig('gcn_metrics_accuracy.png')
plt.show()


plt.figure(figsize=(12,8))
plt.plot(np.arange(1, len(train_loss_all) + 1), np.array(train_loss_all), label='Train Loss', c='orange')
plt.plot(np.arange(1, len(test_loss_all) + 1), np.array(test_loss_all), label='Test Loss', c='blue')
plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.title('GCNConv')
plt.legend(loc='lower right', fontsize='x-large')
plt.savefig('gcn_train_loss.png')
plt.show()


print(f'Final Test Loss: {test_loss_all[-1]:.4f}, Final Test Accuracy: {test_acc_all[-1]:.4f}')
print(f'Final Train Accuracy: {train_acc_all[-1]:.4f}')
