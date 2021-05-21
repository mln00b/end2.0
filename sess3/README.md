# Session-3

A neural network that can:
take 2 inputs:
    an image from MNIST dataset, and
    a random number between 0 and 9
and gives two outputs:
    the "number" that was represented by the MNIST image, and
    the "sum" of this number with the random number that was generated and sent as the input to the network

## Network details

2 networks - 1 for classification, 1 for adding the sum

Classification network has 2 convolution, followed by max pool, followed by 2 linear layers
Sum network: takes the output of the 1st image-linear layer, adds a FC layer on the input, then another FC layer to get the final sum value

```
Net(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
  (dropout1): Dropout(p=0.25, inplace=False)
  (dropout2): Dropout(p=0.5, inplace=False)
  (fc1): Linear(in_features=9216, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
  (fc3): Linear(in_features=1, out_features=16, bias=True)
  (fc4): Linear(in_features=144, out_features=16, bias=True)
  (fc5): Linear(in_features=16, out_features=1, bias=True)
)
```

## Loss Functions

Cross Entropy Loss for classification - usual classification loss
L1 loss used for the sum part - treated as regression problem

## Train Results

Trained for 20 epochs, Adam optimizer, LR: 3e-4


MNIST Classification accuracy achieves 99.3, Sum achieves an accuracy of 78.3

## Improvements

MNIST results seem comparable, more hyper-param tuning can be done.
For the sum network: more weightage can be given to the loss, currently both are given equal weightage.
Further hyper-param tuning can be done
