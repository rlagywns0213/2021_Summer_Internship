# Implementation of Deep Graph InfoMax in PyTorch
====

## How to extract local feature and global feature in Graph

![image](https://user-images.githubusercontent.com/28617444/127652667-f87b5442-2cb9-420f-b803-4a043002a643.png)

![image](https://user-images.githubusercontent.com/28617444/127652697-d2fff30a-d4fe-4273-be7f-48d1bb036dba.png)


## Usage

I compute F = 512 features in each layer, with the PReLU activation according to Paper.
```bash
python train.py --dataset cora
```

> initial configuration  
epochs: 100<br>
factor: 50<br>
lambda_param: 0.01<br>
learning_rate: 0.05


## Example  
```bash
python train.py --bias
```

## Terminal_Results

```bash
$ python train.py   
Loading cora dataset...
Epoch: 0001 loss_train: 0.6921 time: 2.2291s
Epoch: 0002 loss_train: 1.0717 time: 1.8863s
Epoch: 0003 loss_train: 0.8086 time: 1.8973s
Epoch: 0004 loss_train: 0.8917 time: 2.3387s
Epoch: 0005 loss_train: 0.7135 time: 2.0027s
.
.
.
Epoch: 0095 loss_train: 0.0408 time: 1.8570s
Epoch: 0096 loss_train: 0.0429 time: 1.8267s
Epoch: 0097 loss_train: 0.0425 time: 1.9485s
Epoch: 0098 loss_train: 0.0441 time: 1.7973s
Epoch: 0099 loss_train: 0.0392 time: 2.6459s
Epoch: 0100 loss_train: 0.0402 time: 2.6929s
Optimization Finished!
Total time elapsed: 207.5775s
classification Start
Accuracy: 0.8231801390647888
```
2. Adding bias
```bash
$ python train.py --bias
adding bias
기존 loss: 0.781466335477013
Iteration: 10, train_loss = 0.5777, test_loss = 1.7902, average time for 1 epoch : 1.4632
Iteration: 20, train_loss = 0.5214, test_loss = 1.6160, average time for 1 epoch : 1.4680
Iteration: 30, train_loss = 0.5083, test_loss = 1.5752, average time for 1 epoch : 1.5398
Iteration: 40, train_loss = 0.5023, test_loss = 1.5566, average time for 1 epoch : 1.5191
Iteration: 50, train_loss = 0.4987, test_loss = 1.5456, average time for 1 epoch : 1.5643
Iteration: 60, train_loss = 0.4963, test_loss = 1.5382, average time for 1 epoch : 1.4794
Iteration: 70, train_loss = 0.4946, test_loss = 1.5328, average time for 1 epoch : 1.5510
Iteration: 80, train_loss = 0.4933, test_loss = 1.5288, average time for 1 epoch : 1.7602
Iteration: 90, train_loss = 0.4923, test_loss = 1.5257, average time for 1 epoch : 1.6292
Iteration: 100, train_loss = 0.4915, test_loss = 1.5231, average time for 1 epoch : 1.4900
```


## References

[1] Veličković, Petar, et al. "Deep graph infomax." arXiv preprint arXiv:1809.10341 (2018).