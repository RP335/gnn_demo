import itertools
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import torch_geometric
from torch_geometric.data import Dataset, Data
import numpy as np
import os

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")


player_df = pd.read_csv("tbl_player.csv")
skill_df = pd.read_csv("tbl_player_skill.csv")
team_df = pd.read_csv("tbl_team.csv")

# Extract subsets
player_df = player_df[["int_player_id", "str_player_name", "str_positions", "int_overall_rating", "int_team_id"]]
skill_df = skill_df[["int_player_id", "int_long_passing", "int_ball_control", "int_dribbling"]]
team_df = team_df[["int_team_id", "str_team_name", "int_overall"]]

# Merge data
player_df = player_df.merge(skill_df, on='int_player_id')
fifa_df = player_df.merge(team_df, on='int_team_id')

# Sort dataframe
fifa_df = fifa_df.sort_values(by="int_overall_rating", ascending=False)
print("Players: ", fifa_df.shape[0])
fifa_df.head()


sorted_df = fifa_df.sort_values(by="int_player_id")
# Select node features
node_features = sorted_df[["str_positions", "int_long_passing", "int_ball_control", "int_dribbling"]]
# Convert non-numeric columns
pd.set_option('mode.chained_assignment', None)
positions = node_features["str_positions"].str.split(",", expand=True)
node_features["first_position"] = positions[0]
# One-hot encoding
node_features = pd.concat([node_features, pd.get_dummies(node_features["first_position"])], axis=1, join='inner')
node_features.drop(["str_positions", "first_position"], axis=1, inplace=True)
node_features = node_features.astype(int)

node_features.head()

sorted_df = fifa_df.sort_values(by="int_player_id")
labels = sorted_df[["int_overall"]]
print(labels)

y = labels.to_numpy()
y.shape # [num_nodes, 1] --> node regression

x = node_features.to_numpy()
x.shape # [num_nodes x num_features]

fifa_df["int_player_id"] = fifa_df.reset_index().index
fifa_df.head()

team_counts = fifa_df["str_team_name"].value_counts()
print(team_counts)

teams = fifa_df["str_team_name"].unique()
all_edges = np.array([], dtype=np.int32).reshape((0, 2))
for team in teams:
    team_df = fifa_df[fifa_df["str_team_name"] == team]
    players = team_df["int_player_id"].values
    # Build all combinations, as all players are connected
    permutations = list(itertools.combinations(players, 2))
    edges_source = [e[0] for e in permutations]
    edges_target = [e[1] for e in permutations]
    team_edges = np.column_stack([edges_source, edges_target])
    all_edges = np.vstack([all_edges, team_edges])

# Convert to Pytorch Geometric format
edge_index = all_edges.transpose()
edge_index # [2, num_edges]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
edge_index = torch.tensor(edge_index, dtype=torch.long).to(device).contiguous()
edge_index.shape # [2, num_edges]
x = torch.tensor(x, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float)

data = Data(x=x, edge_index=edge_index, y=y)


num_nodes = data.num_nodes


indices = list(range(num_nodes))

# Split indices into training, validation, and test sets
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

# Create masks
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

# Assign masks
train_mask[train_indices] = True
test_mask[test_indices] = True

# Add masks to data object
data.train_mask = train_mask
data.test_mask = test_mask
