"""
A basic template I made that I will modify and use within this repo
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

# Some global variables for model training analysis
accuracy_list = []
loss_list = []
f1_list = []
first_grad_list = [] 
last_grad_list = []
ratio_list = []


"""
EdgesModule defines edges as linear neurons
"""
class EdgesModule(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.edge = nn.Linear(dim,dim)

        # Initiialisation to fix exploding/vanishing gradients
        nn.init.xavier_uniform_(self.edge.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.edge.bias)

        
    def forward(self, x):
        input = self.edge(x)
        output = F.leaky_relu(self.edge(x), negative_slope=0.01)
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
        self.exit_weight = EdgesModule(self.dim)

        # Generating all paths from graph
        self.all_paths = self.generate_paths(graph, self.entry_node, self.exit_node)
        self.norm = nn.BatchNorm1d(dim)

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
        node_counts = {}
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
                    node_counts[node] = 1
                # Else if edge has been used before
                else:
                    node_outputs[node] = node_outputs[node] + edge_out
                    node_counts[node] += 1
                inital_node = node
        
        # Normalising and returning the final output node
        exit_output = self.exit_weight(node_outputs[self.exit_node]) #node_outputs[self.exit_node] / node_counts[self.exit_node]  # NOTE: Could make this a learnable paramter?
        return self.norm(exit_output+x) #  residual connection in form of + x

    
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
    def __init__(self, graph, dim, num_layers, dropout=0.05):
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



"""
The Data class defines the dataset which returns the training and target values when called
"""
class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# This function does data loading + standardization
def standardise(trainingPathExo, testingPathExo):
    """
    Standardises and Cleans the data, creates dataframe for training and testing data
    
    :param trainingPathExo: path to training data csv
    :param testingPathExo: path to testing data csv
    """

    trainDF = pd.read_csv(trainingPathExo).drop(columns="Patient_ID")
    testDF = pd.read_csv(testingPathExo).drop(columns="Patient_ID")
    print(testDF.head())

    # Replacement maps for onehot encoding if required
    oneHotRequired = False
    if oneHotRequired:
        Gender_map = {'Male':1, 'Female':2}
        Subscription_Type_map = {'Basic':1, 'Standard':2, 'Premium':3}
        Contract_Length_map = {'Monthly':1, 'Annual':2, 'Quarterly':3}

        # Actually replacing
        trainDF['Gender'] = trainDF['Gender'].replace(Gender_map)
        trainDF['Subscription Type'] = trainDF['Subscription Type'].replace(Subscription_Type_map)
        trainDF['Contract Length'] = trainDF['Contract Length'].replace(Contract_Length_map)
        testDF['Gender'] = testDF['Gender'].replace(Gender_map)
        testDF['Subscription Type'] = testDF['Subscription Type'].replace(Subscription_Type_map)
        testDF['Contract Length'] = testDF['Contract Length'].replace(Contract_Length_map)

    # Dropping Nan rows
    testDF.dropna(inplace=True)
    trainDF.dropna(inplace=True)

    # Standardising
    scaler = StandardScaler()
    scaler.fit(trainDF.drop(columns=['Heart_Disease_Risk']))
    xTrain = scaler.transform(trainDF.drop(columns=['Heart_Disease_Risk']))
    xTest = scaler.transform(testDF.drop(columns=['Heart_Disease_Risk']))
    yTrain = trainDF['Heart_Disease_Risk'].values
    yTest = testDF['Heart_Disease_Risk'].values


    #print("Class distribution (train):", np.bincount(yTrain))
    #print("Class distribution (test):", np.bincount(yTest))
    return xTrain, xTest, yTrain, yTest

# This function saves the model weights
def summarySave(finalLoss, model):
    print(f'\nModel Finished Training!\nFinal Loss = {round(finalLoss, 2)}')
    torch.save(model.state_dict(), "modelSave.pth")
    print('Model Saved!')

def sigmoid(x):
    return 1/(1+np.exp(-x))

# This is the main training and evaluation loop
def train_model(model, trainLoader, testLoader, device, epochs=10, lr=0.0005):
    # Compute class weights based on training labels
    #class_weights = torch.tensor([1.0, 1.5]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_f1 = 0.0

    for epoch in range(epochs):
        count = 0
        # --- Training ---
        model.train()
        total_loss = 0
        for x, y in tqdm(trainLoader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            yHat = model(x)

            # Struggling to distinguish between classes, guessing all 1, so trying to seperate the classes more by rewarding wide distinctions.
            logits = yHat 
            separation = torch.abs(logits[:, 0] - logits[:, 1])
            margin = 10.0
            separation_penalty = torch.clamp(margin - separation, min=0)

            loss = criterion(logits, y) + 0.0 * separation_penalty.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Debugging stuff here, makes graphs of gradients and rcords means / std dev of predictions 
            if count%100 == 0: 
                print(f"Logits mean: {yHat.mean(dim=0)}, std: {yHat.std(dim=0)}")
                first_grad = model.layers[0].vertical_layers[0].edges['0-1'].edge.weight.grad.norm().item()
                last_grad = model.layers[-1].vertical_layers[0].edges['0-1'].edge.weight.grad.norm().item()
                try:
                    first_grad_list.append(first_grad)
                except ZeroDivisionError:
                    first_grad_list.append(0)
                try:
                    last_grad_list.append(last_grad)
                except ZeroDivisionError:
                    last_grad_list.append(0)
                try:
                    ratio_list.append(first_grad/last_grad)
                except:
                    ratio_list.append(0)

            total_loss += loss.item()
            count += 1
        avg_loss = total_loss / len(trainLoader)
        print(f"Epoch {epoch+1} training loss: {avg_loss:.4f}")

        # --- Evaluation ---
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for x, y in testLoader:
                x, y = x.to(device), y.to(device)
                yHat = model(x)
                probs = F.softmax(yHat, dim=1)
                threshold = 0.5
                preds = (probs[:,1] >= threshold).long()

                all_probs.extend(probs[:,1].cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        # Metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)
        tp = np.sum((all_preds == 1) & (all_labels == 1))

        # Assigning metrics to tracking lists
        accuracy_list.append(accuracy)
        loss_list.append(avg_loss)
        f1_list.append(f1)

        print(f"Top positive probs in this epoch: {np.sort(all_probs)[-5:]}")
        print(f"True positives: {tp} / {np.sum(all_labels==1)}")
        print(f"Test Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print("-"*50)
        save_current_gradient_deails()

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "bestModel.pth")
            print(f"New best F1: {best_f1:.4f} — model saved!\n")

    # Save final model
    print("Training complete!")
    torch.save(model.state_dict(), "modelSave.pth")
    print("Model saved!")

def save_current_gradient_deails():
    # Global variables are first_grad_list, last_grad_list, ratio_list
    length = range(1, len(first_grad_list) + 1)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(length, first_grad_list, marker='o', label="First Gradient")
    plt.plot(length, last_grad_list, marker='o', label="Last Gradient")
    plt.yscale("log")

    for x, y in zip(length, first_grad_list):
        plt.text(x, y, f"{y:.2e}", fontsize=8, ha='center', va='bottom')

    for x, y in zip(length, last_grad_list):
        plt.text(x, y, f"{y:.2e}", fontsize=8, ha='center', va='top')

    plt.xlabel("Epoch")
    plt.ylabel("Gradient Value (log scale)")
    plt.title("First and Last Gradient Over Time")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.subplot(2, 1, 2)
    plt.plot(length, ratio_list, marker='o', label="Ratio of first to last gradient")
    plt.yscale("log")

    for x, y in zip(length, ratio_list):
        plt.text(x, y, f"{y:.2e}", fontsize=8, ha='center', va='bottom')

    plt.xlabel("Epoch")
    plt.ylabel("Ratio (log scale)")
    plt.title("Gradient Ratio Over Time")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()

    save_path = os.path.join(os.getcwd(), "gradient_graphs.png")
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    print(f'First Grads: {first_grad_list}\nLast Grads: {last_grad_list}\nRatio List: {ratio_list}')


def plot_training_metrics():
    """
    Plots several training metrics
    
    :param accuracy_list: List of all accuracy score over epochs
    :param loss_list: KList of all losses ovee epochs
    :param f1_list: List of all f1 scores over epochs
    """
    epochs = range(1, len(accuracy_list) + 1)
    plt.figure(figsize=(10, 6))

    # Graphing Accurtacy and loss
    plt.subplot(2, 1, 1)
    plt.plot(epochs, accuracy_list, label="Accuracy")
    plt.plot(epochs, loss_list, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Accuracy & Loss Over Time")
    plt.legend()
    plt.grid(True)

    # Graphiong f1 score
    plt.subplot(2, 1, 2)
    plt.plot(epochs, f1_list, label="F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("F1 Score Over Time")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    cwd = os.getcwd()  # get current working directory
    save_path = os.path.join(cwd, "training_metrics.png")
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")

    plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
import numpy as np

# The main loop of the program 
def main():
    # Using GPU if avalible 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print("Using device:", device)
  
    base_dir = os.path.dirname(os.path.abspath(__file__))
    trainingDataPath = os.path.join(base_dir, "Data", "heart_data", "heart_train.csv")
    testingDataPath = os.path.join(base_dir, "Data", "heart_data", "heart_test.csv")

    # First, standardising data, function returns 4 lists, all standardised.
    xTrain, xTest, yTrain, yTest = standardise(trainingDataPath, testingDataPath)

    # DEBUGGIN - REMOVE LATER
    print(f'Unique Y Values : {np.unique(yTrain)}')
    # ABOVE REMOVE LATER

    # Next, creating dataset objects
    trainDataset = Data(xTrain, yTrain)
    testDataset = Data(xTest, yTest)
    print('Datasets created...')

    # Creating dataloader object 
    batchSize = 32
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=False)
    print('Dataloaders created...')

    # Creating Model
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
    dim = 64

    model = SuperModule(graph, dim, 4, 3, xTrain.shape[1], 2)
    print(f"Number of paths found: {len(model.layers[0].vertical_layers[0].all_paths)}")
    print(f"Paths: {model.layers[0].vertical_layers[0].all_paths}")
    model.to(device)
    print('Model created, starting training...')
    train_model(model, trainLoader, testLoader, device)

    # Plotting Training Metrics
    plot_training_metrics()

if __name__ == "__main__":
    main()