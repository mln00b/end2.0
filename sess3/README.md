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

## Train Results

Trained for 20 epochs, Adam optimizer, LR: 3e-4

train logs:

```
Epoch: 0, Train Cls loss: 0.0447203554213047, Train Sum loss: 0.8054041862487793, Train Total loss: 0.8501245379447937
Epoch: 0, Test Cls accuracy: 0.9736, Test Sum accuracy: 0.2551
Epoch: 1, Train Cls loss: 0.2555199861526489, Train Sum loss: 0.840211808681488, Train Total loss: 1.0957317352294922
Epoch: 1, Test Cls accuracy: 0.9828, Test Sum accuracy: 0.3222
Epoch: 2, Train Cls loss: 0.13116218149662018, Train Sum loss: 0.5348316431045532, Train Total loss: 0.6659938097000122
Epoch: 2, Test Cls accuracy: 0.9863, Test Sum accuracy: 0.3745
Epoch: 3, Train Cls loss: 0.015532579272985458, Train Sum loss: 0.3780766725540161, Train Total loss: 0.39360925555229187
Epoch: 3, Test Cls accuracy: 0.9873, Test Sum accuracy: 0.3576
Epoch: 4, Train Cls loss: 0.3651069104671478, Train Sum loss: 0.40289586782455444, Train Total loss: 0.7680027484893799
Epoch: 4, Test Cls accuracy: 0.9896, Test Sum accuracy: 0.3922
Epoch: 5, Train Cls loss: 0.004591717850416899, Train Sum loss: 0.3161105513572693, Train Total loss: 0.3207022547721863
Epoch: 5, Test Cls accuracy: 0.9893, Test Sum accuracy: 0.589
Epoch: 6, Train Cls loss: 0.08979840576648712, Train Sum loss: 0.3045210540294647, Train Total loss: 0.39431947469711304
Epoch: 6, Test Cls accuracy: 0.9904, Test Sum accuracy: 0.5652
Epoch: 7, Train Cls loss: 0.07604522258043289, Train Sum loss: 0.6878137588500977, Train Total loss: 0.76385897397995
Epoch: 7, Test Cls accuracy: 0.9907, Test Sum accuracy: 0.6313
Epoch: 8, Train Cls loss: 0.010247101075947285, Train Sum loss: 0.2910170555114746, Train Total loss: 0.3012641668319702
Epoch: 8, Test Cls accuracy: 0.9915, Test Sum accuracy: 0.4382
Epoch: 9, Train Cls loss: 0.0009132736013270915, Train Sum loss: 0.2674673795700073, Train Total loss: 0.26838064193725586
Epoch: 9, Test Cls accuracy: 0.9911, Test Sum accuracy: 0.4117
Epoch: 10, Train Cls loss: 0.008119918406009674, Train Sum loss: 0.597327470779419, Train Total loss: 0.6054474115371704
Epoch: 10, Test Cls accuracy: 0.991, Test Sum accuracy: 0.5215
Epoch: 11, Train Cls loss: 0.003412364050745964, Train Sum loss: 0.2016725391149521, Train Total loss: 0.2050849050283432
Epoch: 11, Test Cls accuracy: 0.9917, Test Sum accuracy: 0.472
Epoch: 12, Train Cls loss: 0.007631594315171242, Train Sum loss: 0.2219007909297943, Train Total loss: 0.229532390832901
Epoch: 12, Test Cls accuracy: 0.9923, Test Sum accuracy: 0.3678
Epoch: 13, Train Cls loss: 0.0013715188251808286, Train Sum loss: 0.198391854763031, Train Total loss: 0.19976337254047394
Epoch: 13, Test Cls accuracy: 0.9907, Test Sum accuracy: 0.5873
Epoch: 14, Train Cls loss: 0.0240501556545496, Train Sum loss: 0.2669416666030884, Train Total loss: 0.29099181294441223
Epoch: 14, Test Cls accuracy: 0.9918, Test Sum accuracy: 0.3732
Epoch: 15, Train Cls loss: 0.0011886509601026773, Train Sum loss: 0.22864079475402832, Train Total loss: 0.22982944548130035
Epoch: 15, Test Cls accuracy: 0.9915, Test Sum accuracy: 0.3523
Epoch: 16, Train Cls loss: 0.005533467046916485, Train Sum loss: 0.25233685970306396, Train Total loss: 0.25787031650543213
Epoch: 16, Test Cls accuracy: 0.9912, Test Sum accuracy: 0.392
Epoch: 17, Train Cls loss: 0.0017441347008571029, Train Sum loss: 0.23421594500541687, Train Total loss: 0.23596008121967316
Epoch: 17, Test Cls accuracy: 0.9926, Test Sum accuracy: 0.6977
Epoch: 18, Train Cls loss: 0.0033635697327554226, Train Sum loss: 0.24075782299041748, Train Total loss: 0.24412138760089874
Epoch: 18, Test Cls accuracy: 0.9915, Test Sum accuracy: 0.3831
Epoch: 19, Train Cls loss: 0.001968518365174532, Train Sum loss: 0.32065799832344055, Train Total loss: 0.322626531124115
Epoch: 19, Test Cls accuracy: 0.9914, Test Sum accuracy: 0.2831
```

MNIST Classification accuracy achieves 99.2, Sum achieves an accuracy of 58.7

## Improvements

MNIST results seem comparable, more hyper-param tuning can be done.
For the sum network: more weightage can be given to the loss, currently both are given equal weightage.
Further hyper-param tuning can be done