# Notes
This outlines the processes I went through with this project. I am recording it here to eventually put formally into the readme. 

## The basic template
The basic template is a way to get a baseline measurement. It is a simple deep NN. 

The training data used is heart disease data. The output of the model is 1 or 0, 1 meaning the person is at risk. 

The model is purposly not optimised very much so that the perfomance gains or pitfalls of the new models will be particually obvious

Initial Result:
```
Epoch 10 training loss: 0.5111
Top positive probs in this epoch: [0.9880862  0.98890173 0.99113715 0.99393445 0.9950433 ]
True positives: 158 / 440
Test Accuracy: 0.6856, F1-score: 0.5008
Confusion Matrix:
[[529  33]
 [282 158]]
```

 After adding seperation penalty:
 ```
Epoch 10 training loss: 2.8024
Top positive probs in this epoch: [0.9999883  0.99999154 0.9999926  0.9999931  0.99999523]
True positives: 322 / 440
Test Accuracy: 0.7246, F1-score: 0.7000
Confusion Matrix:
[[404 158]
 [118 322]]
 ```

Finally, lowered the pentlty and increased learning rate:
```
Epoch 10 training loss: 2.6683
Top positive probs in this epoch: [0.99999905 0.99999905 0.9999994  0.9999995  0.99999964]
True positives: 312 / 440
Test Accuracy: 0.7176, F1-score: 0.6880
Confusion Matrix:
[[407 155]
 [128 312]]
 ```

 Key training values:
 1. 10 epochs
 2. Learning rate = 0.002
 3. 17 input features
 4. 2 output classes
 5. 1043 nodes
 6. 271872 connections

 Here is the model architecture:
 ```
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
```

## General Idea of 3D graphs
The below shows my intial ideas for the 3D graph as well as the algoithm used to traverse it.

![First 3D sketch](Pictures/draftmisc.png "Initial Sketch")

From this, there are two 3D implimentations I plan to try:
1. The implimentation where I have one input node in each 3d Shape making a super node (the 3d shape) with each node within the shape being a sub node. This will be the "Super/Sub Implimentation (SSI)"
2. The implimentation where I have treat the shape (or multiple chained together) as the entire layer (e.g. the same input goes to the red and green nodes in the above picture) will be called the "Traditional Implimentation (TI)"

## Super Sub Implimentation

# Proof of concept
The initial proof of concept algorithm is shown in [ssi_algo_demo.py](ssi_algo_demo.py).

This uses a backtrack DFS algorithm to collect all paths from the chosen graph. These can then be filtered. 

Below is a gif on the paths the algorithm generated for the graph I made, which is currently a cube. 

![POC GIF](Pictures/paths.gif "Proof of Concept SSI")
