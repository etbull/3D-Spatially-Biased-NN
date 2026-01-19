# Notes
This outlines the processes I went through with this project. I am recording it here to eventually put formally into the readme. 

## The basic template
The basic template is a way to get a baseline measurement. It is a simple deep NN. 

The training data used is customer churn data. The output of the model is 1 or 0, 1 meaning the customer would churn. 

After an initial 25 epochs of training, the output looked like this:
![alt text](image.png)


Initial Result:
Epoch 10 training loss: 0.5111
Top positive probs in this epoch: [0.9880862  0.98890173 0.99113715 0.99393445 0.9950433 ]
True positives: 158 / 440
Test Accuracy: 0.6856, F1-score: 0.5008
Confusion Matrix:
[[529  33]
 [282 158]]


 After adding seperation penalty:
Epoch 10 training loss: 2.8024
Top positive probs in this epoch: [0.9999883  0.99999154 0.9999926  0.9999931  0.99999523]
True positives: 322 / 440
Test Accuracy: 0.7246, F1-score: 0.7000
Confusion Matrix:
[[404 158]
 [118 322]]

 After lowering seperation penalty and increasing learning rate:
 