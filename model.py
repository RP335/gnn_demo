import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GAT(torch.nn.Module):
    def __init__(self,  hidden_channels, input_dim, output_dim):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(input_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# Create an instance of the model
model = GAT(input_dim=18, hidden_channels=16, output_dim=1)  # input_dim=3 for FIFA features, output_dim=1 for regression
print(model)

# Optimizer and Loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.MSELoss()
