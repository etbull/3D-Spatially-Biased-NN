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
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score



"""
EdgesModule defines edges as linear neurons
"""
class EdgesModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.edge = nn.Linear(dim,dim)
        
    def forward(self, x):
        input = self.edge(x)
        output = nn.functional.relu(input)
        return output
    
"""
Graph Module defines model where input goes into graph, is calculated along several paths, normalised, and outputed
"""
class GraphModule(nn.Module):
    def __init__(self, graph, dim):
        super().__init__()
        # Defining important stuff like the graph dict, the entry and exit node, and how many dimensions in the input data
        self.graph = graph
        self.entry_node = list(graph.keys())[0]
        self.exit_node = list(graph.keys())[-1]
        self.dim = dim
        # Generating all paths from graph
        self.all_paths = self.generate_paths(graph, self.entry_node, self.exit_node)
        self.norm = nn.LayerNorm(dim)

        # Making all the edge objects
        self.edges = nn.ModuleDict()
        # Iterating over the pairs of nodes and list of connected nodes, creating an edge object for each edge and adding to dict
        for src, dsts in self.graph.items():
            for dst in dsts:                
                key = f"{src}-{dst}"
                self.edges[key] = EdgesModule(self.dim)
        
    def forward(self, x):
        # Creatintg structure to hold the outputs of each node
        node_outputs = {}
        node_outputs[self.entry_node] = x

        # Iteratinf over all paths
        for paths in self.all_paths:
            inital_node = self.entry_node
            for node in paths:
                key = f"{inital_node}-{node}"
                # Logic for if the edge hasn't been touched yet, adding the result to the dictionary
                prev_out = node_outputs[inital_node]
                edge_out = self.edges[key].forward(prev_out)
                if node not in node_outputs:
                    node_outputs[node] = edge_out
                # Else if edge has been used before
                else:
                    node_outputs[node] = node_outputs[node] + edge_out
                inital_node = node
        
        # Normalising and returning the final output node
        return self.norm(node_outputs[self.exit_node])
    
    def generate_paths(self, graph, start, end):
        all_paths = []
        # Using simple backtrack DFS algo to get all possible paths
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
        # Could filter paths here in the future if wanted, e.g, only getting smallest path, or a random path
        return all_paths
    
"""
GraphLayer stacks Graphs vertically, they all get the same input
"""
class GraphLayer(nn.Module):
    def __init__(self, graph, dim, num_layers):
        super().__init__()
        # Adding multiple graphs to one layer
        self.vertical_layers = nn.ModuleList(
            [GraphModule(graph, dim) for i in range(num_layers)]
        )
        
    def forward(self, x):
        # This ensures that all graphs get the same intput. For densely connected network
        output = [g(x) for g in self.vertical_layers]
        return output
    
"""
SuperModule connects together multiple graphLayers.
"""
class SuperModule(nn.Module):
    def __init__(self, graph, dim, width, depth, input_dim, output_classes):
        super().__init__()
        self.width = width
        self.depth = depth

        # First Layer
        self.input_layer = nn.Linear(input_dim, dim)

        # Graph layers
        self.layers = nn.ModuleList(
            [GraphLayer(graph, dim, width) for _ in range(depth)]
        )

        # Connectors between layers (depth - 1)
        self.connectors = nn.ModuleList(
            [nn.Linear(width * dim, dim) for _ in range(depth - 1)]
        )

        # Final classifier
        self.classifier = nn.Linear(dim, output_classes)

    def forward(self, x):
        x = self.input_layer(x)

        for layer in range(self.depth):
            output = self.layers[layer](x)  # list of [B, dim]

            if layer < self.depth - 1:
                # Dense connection to next layer
                x = torch.cat(output, dim=-1)      # [B, width*dim]
                x = self.connectors[layer](x)      # [B, dim]
            else:
                # Final layer: aggregate graphs
                x = torch.stack(output).mean(0)    # [B, dim]

        # Binary classification head
        x = self.classifier(x)                     # [B, 1]
        return x




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

dim = 8
model = SuperModule(graph, dim, 6, 4, 1, 2)

# ------------------------
# Training setup
# ------------------------

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ------------------------
# Synthetic data
# ------------------------

def make_batch(batch_size=32):
    x = torch.randn(batch_size, 1)
    y = (x.sum(dim=1) > 0).long()
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

    if epoch % 10 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

# ------------------------
# Testing loop
# ------------------------
x, y = make_batch()
with torch.no_grad():
    yHat = model(x)
    loss = criterion(yHat, y)

    probs = torch.sigmoid(yHat)
    preds = (probs > 0.5).int()
    accuracy = accuracy_score(y, preds)

print(accuracy, loss)

