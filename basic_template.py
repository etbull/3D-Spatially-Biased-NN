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
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        
    def forward(self, x):
        x = self.flatten(x)
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
    trainDF = pd.read_csv(trainingPathExo)
    testDF = pd.read_csv(testingPathExo)

    scaler = StandardScaler()
    scaler.fit(trainDF.drop(columns=['LABEL']))
    xTrain = scaler.transform(trainDF.drop(columns=['LABEL']))
    xTest = scaler.transform(testDF.drop(columns=['LABEL']))
    yTrain = trainDF['LABEL'].values
    yTest = testDF['LABEL'].values
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
def train_model(model, trainLoader, testLoader, device, epochs=50, lr=0.001):
    # Compute class weights based on training labels
    y_train_labels = np.array(trainLoader.dataset.y)
    classes = np.unique(y_train_labels)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_f1 = 0.0

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        total_loss = 0
        for x, y in tqdm(trainLoader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            yHat = model(x)
            loss = criterion(yHat, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
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
                preds = torch.argmax(yHat, dim=1)

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

        print(f"Top positive probs in this epoch: {np.sort(all_probs)[-5:]}")
        print(f"True positives: {tp} / {np.sum(all_labels==1)}")
        print(f"Test Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print("-"*50)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "bestModel.pth")
            print(f"New best F1: {best_f1:.4f} — model saved!\n")

    # Save final model
    print("Training complete!")
    torch.save(model.state_dict(), "modelSave.pth")
    print("Model saved!")


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
    model = Model(xTrain.shape[1], 2)
    model.to(device)
    print('Model created, starting training...')
    train_model(model, trainLoader, testLoader, device)

if __name__ == "__main__":
    main()