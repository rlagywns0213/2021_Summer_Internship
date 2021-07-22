# Implementation of Factorization Machine (FM)

## In Paper
![image](https://user-images.githubusercontent.com/28617444/126597610-79e67f31-9e49-4181-a2bf-34628f53a756.png)

1. Model Equation

![image](https://user-images.githubusercontent.com/28617444/126599320-e389afd5-c21b-40da-a481-caa219a3d50c.png)

2. for only linear complexity : O(kn)

![image](https://user-images.githubusercontent.com/28617444/126599461-4e2da78a-aa33-4aa9-871a-97ca00dd14ba.png)


## About

- SVM : fails to cover spase matrix
- FM : Interactions between values `V_ij = nn.Parameter(nn.init.normal_(torch.zeros((field_dims, latent_dims))`
    - cover huge sparsity matrix

- I used `sklearn.feature_extraction` to DictVectorizer on MovieLens Dataset
    - Like a One-Hot encoding in Dictionary

- I used `torch` to use autograd, DataLoader, Dataset
- The RMSE is the result calculated by batch units.

## Results

- Epoch 100

| k |  total loss | best_RMSE | 1 epoch Time |
| :---: | :---: |  :---: | :---: |
| 5  |  987.1157 |  0.3506 |  9.2686 |
| 15 |  865.5704 |  0.4488 |  10.3403 |
| 30 |  849.7148 |  0.3744 |  10.5632 |

## Usage

```bash
python train.py --batch_size 128 --k 15
```

> initial configuration  
epochs: 100<br>
k: 5<br>
learning_rate: 0.05<br>
batch_size: 64

## Terminal_Results

1. k : 5
```bash
$ python train.py  
Epoch 10 of 100, training Loss: 1238.1840, Test RMSE: 0.9533
===> 1 epoch mean Time : 8.631
Epoch 20 of 100, training Loss: 1184.6510, Test RMSE: 0.9118
===> 1 epoch mean Time : 9.434
Epoch 30 of 100, training Loss: 1157.0186, Test RMSE: 0.8571
===> 1 epoch mean Time : 9.653
Epoch 40 of 100, training Loss: 1133.6735, Test RMSE: 0.7650
===> 1 epoch mean Time : 8.822
Epoch 50 of 100, training Loss: 1108.6221, Test RMSE: 0.5581
===> 1 epoch mean Time : 10.418
Epoch 60 of 100, training Loss: 1081.6615, Test RMSE: 0.9874
===> 1 epoch mean Time : 9.031
Epoch 70 of 100, training Loss: 1054.3945, Test RMSE: 0.7202
===> 1 epoch mean Time : 9.148
Epoch 80 of 100, training Loss: 1027.6660, Test RMSE: 0.7603
===> 1 epoch mean Time : 9.363
Epoch 90 of 100, training Loss: 1005.9219, Test RMSE: 0.7966
===> 1 epoch mean Time : 9.085
Epoch 100 of 100, training Loss: 987.1157, Test RMSE: 0.9098
===> 1 epoch mean Time : 9.100
Best RMSE :0.35063427686691284,
1 EPOCH MEAN TIME :9.26866399526596
```
2. k : 15
```bash
$ python train.py --k 15
Epoch 10 of 100, training Loss: 1230.0500, Test RMSE: 0.8768
===> 1 epoch mean Time : 10.443
Epoch 20 of 100, training Loss: 1185.0828, Test RMSE: 0.9542
===> 1 epoch mean Time : 11.684
Epoch 30 of 100, training Loss: 1155.7051, Test RMSE: 0.9288
===> 1 epoch mean Time : 10.114
Epoch 40 of 100, training Loss: 1122.6117, Test RMSE: 0.5842
===> 1 epoch mean Time : 11.311
Epoch 50 of 100, training Loss: 1087.8828, Test RMSE: 1.0657
===> 1 epoch mean Time : 10.480
Epoch 60 of 100, training Loss: 1047.1539, Test RMSE: 0.8841
===> 1 epoch mean Time : 10.526
Epoch 70 of 100, training Loss: 1000.8752, Test RMSE: 0.7743
===> 1 epoch mean Time : 10.319
Epoch 80 of 100, training Loss: 953.8219, Test RMSE: 0.9249
===> 1 epoch mean Time : 10.388
Epoch 90 of 100, training Loss: 908.3422, Test RMSE: 0.5947
===> 1 epoch mean Time : 9.920
Epoch 100 of 100, training Loss: 865.5704, Test RMSE: 0.5540
===> 1 epoch mean Time : 8.218
Best RMSE :0.4488947093486786,
1 EPOCH MEAN TIME :10.340331273078919
```

3. k : 30
```bash
$ python train.py --k 30
Epoch 10 of 100, training Loss: 1246.1898, Test RMSE: 0.8885
===> 1 epoch mean Time : 11.556
Epoch 20 of 100, training Loss: 1201.1475, Test RMSE: 1.0217
===> 1 epoch mean Time : 12.227
Epoch 30 of 100, training Loss: 1163.8662, Test RMSE: 1.0164
===> 1 epoch mean Time : 10.173
Epoch 40 of 100, training Loss: 1133.2944, Test RMSE: 0.6523
===> 1 epoch mean Time : 12.571
Epoch 50 of 100, training Loss: 1100.7368, Test RMSE: 1.0388
===> 1 epoch mean Time : 10.353
Epoch 60 of 100, training Loss: 1063.2628, Test RMSE: 1.0007
===> 1 epoch mean Time : 11.295
Epoch 70 of 100, training Loss: 1020.1530, Test RMSE: 0.9870
===> 1 epoch mean Time : 11.356
Epoch 80 of 100, training Loss: 968.8021, Test RMSE: 0.6425
===> 1 epoch mean Time : 10.635
Epoch 90 of 100, training Loss: 910.9095, Test RMSE: 1.0923
===> 1 epoch mean Time : 8.720
Epoch 100 of 100, training Loss: 849.7148, Test RMSE: 0.6360
===> 1 epoch mean Time : 6.744
Best RMSE :0.3744339942932129,
1 EPOCH MEAN TIME :10.56323165178299,
```
## References

[1] Rendle, Steffen. "Factorization machines." 2010 IEEE International conference on data mining. IEEE, 2010.

[2] https://github.com/Namkyeong/RecSys_paper/tree/main/FactorizationMachine
