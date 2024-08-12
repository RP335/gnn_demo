import scipy.io
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import psutil
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import sys

print("Python is 64-bit:", sys.maxsize > 2 ** 32)

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda if torch.cuda.is_available() else "N/A")
print("PyTorch version:", torch.__version__)
print("Number of CUDA devices:", torch.cuda.device_count() if torch.cuda.is_available() else "N/A")
# Base directory containing the folders
base_directory = 'data'

# Folders corresponding to different angles
folders = [ '60', '90', '30']

# Function to load and process each .mat file
def load_mat_file(filepath):
    data = scipy.io.loadmat(filepath)
    # val = data['inputfeats_ph']  # Adjust the key as needed
    # features = val.reshape(val.shape[0], -1)
    val = data['inputfeats_ph']
    # Aggregate features by taking the mean across the time and frequency dimensions
    # features = np.mean(val, axis=(1, 2, 3))
    return val


features_list = []
labels = []

# for folder in folders:
#     folder_path = os.path.join(base_directory, folder)
#     filepaths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.mat')]
#
#     for filepath in filepaths:
#         features = load_mat_file(filepath)
#         features_list.append(features.mean(axis=0))
#         labels.append(folders.index(folder))

for folder_index, folder in enumerate(folders):
    folder_path = os.path.join(base_directory, folder)
    filepaths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.mat')]

    for filepath in filepaths:
        features = load_mat_file(filepath)
        features_list.append(features.reshape(1, -1))
        labels.append(folder_index)  # Use the index of the folder as t

print("list_shape", features_list[0].shape)
features_stack = np.vstack(features_list)
# print(features_stack)
# scaler = StandardScaler()
# normalized_features = scaler.fit_transform(features_stack)
# features_2d = features_stack.reshape(features_stack.shape[0], -1)

similarity_matrix = cosine_similarity(features_stack)
print("similarity_matrix_shape", similarity_matrix.shape)
# Define a threshold for similarity
threshold = 0.6 # Adjust this threshold as needed

# Create edges based on the similarity threshold
edges = []
for i in range(similarity_matrix.shape[0]):
    for j in range(i+1, similarity_matrix.shape[1]):
        if similarity_matrix[i, j] > threshold:
            edges.append((i, j))
            edges.append((j, i))
# k = 5
# nn = NearestNeighbors(n_neighbors=k, metric='cosine')
# nn.fit(normalized_features)
# distances, indices = nn.kneighbors(normalized_features)

# Create edges based on the nearest neighbors
# edges = []
# for i in range(len(indices)):
#     for j in indices[i][1:]:  # Skip the first neighbor (self)
#         edges.append((i, j))
#         edges.append((j, i))  # Add both directions for undirected graph




edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# print("Shape of features:", features_2d.shape)
print("Shape of stack:", features_stack.shape)

# print("Contains NaN:", np.isnan(features_2d).any())
# print("Contains Inf:", np.isinf(features_2d).any())
print(f"Available memory: {psutil.virtual_memory().available / (1024 * 1024 * 1024):.2f} GB")

x = torch.tensor(features_stack, dtype=torch.float)
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
print(labels)
y = torch.tensor(labels, dtype=torch.long)

perm = torch.randperm(len(features_stack))

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


class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_channels, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


input_dim = features_stack.shape[1]
model = GNN(input_dim=input_dim, hidden_channels=64, output_dim=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
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

for epoch in range(1, 1000):
    train_loss = train()
    test_loss, test_acc, train_acc = tst()
    train_loss_all.append(train_loss)
    test_loss_all.append(test_loss)
    test_acc_all.append(test_acc)
    train_acc_all.append(train_acc)
    if epoch % 2 == 0:
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