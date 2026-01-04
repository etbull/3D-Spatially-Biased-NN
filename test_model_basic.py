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

"""
Basic Fully connected Deep Learning NN
"""
class Model(nn.Module):
    def __init__(self, input_length, num_classes):
        super(Model, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_length, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )
        
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


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
    Standardises the data, creates dataframe for training and testing data
    
    :param trainingPathExo: path to training data csv
    :param testingPathExo: path to testing data csv
    """

    trainDF = pd.read_csv(trainingPathExo).drop(columns="id")
    testDF = pd.read_csv(testingPathExo).drop(columns="id")
    #print(testDF.head())

    scaler = StandardScaler()
    scaler.fit(trainDF.drop(columns=['diagnosis']))
    xTrain = scaler.transform(trainDF.drop(columns=['diagnosis']))
    xTest = scaler.transform(testDF.drop(columns=['diagnosis']))
    yTrain = trainDF['diagnosis'].replace('M',1).replace('B',0).values
    yTest = testDF['diagnosis'].replace('M',1).replace('B',0).values
    print("Class distribution (train):", np.bincount(yTrain))
    print("Class distribution (test):", np.bincount(yTest))
    return xTrain, xTest, yTrain, yTest

# This function augments the positive exoplanets so that there's more data
def augmentPositives(X, y, factor=3, shiftMax=1500, noiseStd=0.001):
    xAug, yAug = X.copy(), y.copy()
    pos_idxs = np.where(y==1)[0]
    for _ in range(factor-1):
        for i in pos_idxs:
            sample = np.roll(X[i], np.random.randint(-shiftMax, shiftMax))
            sample += np.random.normal(0, noiseStd, size=sample.shape)
            xAug = np.vstack([xAug, sample])
            yAug = np.append(yAug, 1)
    return xAug, yAug

# This function saves the model weights
def summarySave(finalLoss, model):
    print(f'\nModel Finished Training!\nFinal Loss = {round(finalLoss, 2)}')
    torch.save(model.state_dict(), "modelSave.pth")
    print('Model Saved!')

# This is the main training and evaluation loop
def test_model(model, testLoader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()  

    with torch.no_grad():
        count = 0
        for x, y in testLoader:
            x, y = x.to(device), y.to(device)
            yHat = model(x)
            loss = criterion(yHat, y)
            total_loss += loss.item()

            preds = torch.argmax(yHat, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            count+=1

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Testing Loss: {total_loss/len(testLoader):.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)


# The main loop of the program 
def main():

    # Using GPU if avalible 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print("Using device:", device)
  
    base_dir = os.path.dirname(os.path.abspath(__file__))
    trainingDataPath = os.path.join(base_dir, "Data", "breast_cancer", "training.csv")
    testingDataPath = os.path.join(base_dir, "Data", "breast_cancer", "testing.csv")

    # First, standardising data, function returns 4 lists, all standardised.
    xTrain, xTest, yTrain, yTest = standardise(trainingDataPath, testingDataPath)

    # Next, creating dataset objects
    trainDataset = Data(xTrain, yTrain)
    testDataset = Data(xTest, yTest)
    print('Datasets created...')

    # Creating dataloader object 
    batchSize = 16
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=False)
    print('Dataloaders created...')

    # Creating Model
    PATH = "bestModel.pth"
    model = Model(xTrain.shape[1], 2)
    model.to(device)
    model.load_state_dict(torch.load(PATH, weights_only=True))
    print('Model Loaded, starting training...')
    test_model(model, testLoader, device)

if __name__ == "__main__":
    main()