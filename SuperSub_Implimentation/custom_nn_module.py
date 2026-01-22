"""
A modification of the basic template to use the new structure for super/sub implimentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.optim as optim

# Some global variables for model training analysis
accuracy_list = []
loss_list = []
f1_list = []

"""
This model should
1. Define the graphs
2. Define the paths to take
3. Make an edge nn.Module to track the edges
4. Make a graph.nn module to track the nodes
5. Create a super.nn module to combine multiple graphs together
"""
class EdgesModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.edge = nn.Linear(dim,dim)
        
    def forward(self, x):
        input = self.edge(x)
        output = nn.functional.relu(input)
        return output
    

class GraphModule(nn.Module):
    def __init__(self, graph, dim):
        super().__init__()
        self.graph = graph
        self.entry_node = list(graph.keys())[0]
        self.exit_node = list(graph.keys())[-1]
        self.dim = dim
        # Generating all paths from graph
        self.all_paths = self.generate_paths(graph, self.entry_node, self.exit_node)

        # Making all the edge objects
        self.edges = nn.ModuleDict()
        for src, dsts in self.graph.items():
            for dst in dsts:                
                key = f"{src}-{dst}"
                self.edges[key] = EdgesModule(self.dim)
        
    def forward(self, x):
        node_outputs = {}
        node_outputs[self.entry_node] = x

        for paths in self.all_paths:
            inital_node = self.entry_node
            for node in paths:
                key = f"{inital_node}-{node}"
                # Logic for if the edge hasn't been touched yet
                prev_out = node_outputs[inital_node]
                edge_out = self.edges[key].forward(prev_out)
                if node not in node_outputs:
                    node_outputs[node] = edge_out
                # Else if edge has been used before
                else:
                    node_outputs[node] = node_outputs[node] + edge_out
                inital_node = node
        
        return node_outputs[self.exit_node]
    
    def generate_paths(self, graph, start, end):
        all_paths = []

        def backtrack(current_node, visited, current_path):
            if current_node == end:
                all_paths.append(current_path.copy())
                return
            
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    current_path.append(neighbor)

                    backtrack(neighbor, visited, current_path)

                    current_path.pop()
                    visited.remove(neighbor)

        backtrack(start, {start}, [])
        return all_paths


graph = {
        0:[1,2,4],
        1:[0,3,5],
        2:[0,3,6],
        3:[1,2,7],
        4:[0,5,6],
        5:[1,4,7],
        6:[2,4,7],
        7:[3,5,6]
    }

dim = 4
model = GraphModule(graph, dim)

# ------------------------
# Training setup
# ------------------------

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ------------------------
# Synthetic data
# ------------------------

def make_batch(batch_size=32):
    x = torch.randn(batch_size, dim)
    y = 2 * x
    return x, y

# ------------------------
# Training loop
# ------------------------

epochs = 1000

for epoch in range(epochs):
    x, y = make_batch()

    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")
