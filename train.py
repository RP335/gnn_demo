import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F

from datset import data
from model import model, optimizer, criterion


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

def tst(mask):
    model.eval()
    out = model(data.x, data.edge_index)
    # print("out=",out)
    pred = out.squeeze()
    true = data.y[mask].squeeze()
    # print("true=", true)
    # print("pred=", pred)
    mse = F.mse_loss(pred[mask], true)
    acc = accuracy(pred[mask], true)
    return mse, acc

def accuracy(pred, true, threshold=5):
    diff = torch.abs(pred - true)
    correct = (diff <= threshold).float()
    acc = correct.mean()
    return acc

test_mse_all = []
test_acc_all = []

for epoch in range(1, 5000):
    loss = train()
    test_mse, test_acc = tst(data.test_mask)
    test_mse_all.append(test_mse)
    test_acc_all.append(test_acc)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test MSE: {test_mse:.4f}, Test Acc: {test_acc:.4f}')

test_mse, test_acc = tst(data.test_mask)
print(f'Test MSE: {test_mse:.4f}, Test Accuracy: {test_acc:.4f}')

# Detach the tensors and convert to numpy arrays
test_mse_all_np = np.array([v.detach().numpy() for v in test_mse_all])
test_acc_all_np = np.array([v.detach().numpy() for v in test_acc_all])

# Plot the metrics
plt.figure(figsize=(12,8))
plt.plot(np.arange(1, len(test_mse_all_np) + 1), test_mse_all_np, label='Testing MSE', c='red')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('GATConv')
plt.legend(loc='upper right', fontsize='x-large')
plt.savefig('gat_metrics_mse_loss.png')
plt.show()


plt.figure(figsize=(12,8))
plt.plot(np.arange(1, len(test_acc_all_np) + 1), test_acc_all_np*100, label='Testing accuracy in %', c='orange')
plt.xlabel('Epochs')
plt.ylabel('Metric')
plt.title('GATConv')
plt.legend(loc='upper right', fontsize='x-large')
plt.savefig('gat_metrics_accuracy.png')
plt.show()


print(f'Test MSE: {test_mse_all_np[-1]:.4f}, Test Accuracy: {test_acc_all_np[-1]:.4f}')

