# Implementation of Probabilistic Matrix Factorization (PMF)

### the conditional distribution over the observed ratings
![image](https://user-images.githubusercontent.com/28617444/126933848-0eddc74e-ef08-4d10-8481-1482c789efae.png)


### objective function

![image](https://user-images.githubusercontent.com/28617444/126935195-bdec0413-7177-47b4-9642-1df3cc4d0813.png)

- Gaussian : `np.random.normal`
- new Parameter : `_lambda_U` , `_lambda_V`
- `non_zero()` for I (indicator)

## Usage

```bash
python train.py
```

> initial configuration  
epochs: 100<br>
factor: 2<br>
learning_rate: 0.05<br>
lambda_u : 0.01<br>
lambda_v : 0.01



## Results

1. default

```bash
$ python train.py --factor 2
Iteration: 10, train_loss = 1.0057, test_loss = 1.2013, average time for 1 epoch : 2.6037
Iteration: 20, train_loss = 0.9858, test_loss = 1.1893, average time for 1 epoch : 2.5492
Iteration: 30, train_loss = 0.9817, test_loss = 1.1857, average time for 1 epoch : 2.7946
Iteration: 40, train_loss = 0.9804, test_loss = 1.1841, average time for 1 epoch : 2.6821
Iteration: 50, train_loss = 0.9798, test_loss = 1.1839, average time for 1 epoch : 3.0660
Iteration: 60, train_loss = 0.9794, test_loss = 1.1843, average time for 1 epoch : 3.3074
Iteration: 70, train_loss = 0.9791, test_loss = 1.1845, average time for 1 epoch : 3.1432
Iteration: 80, train_loss = 0.9789, test_loss = 1.1847, average time for 1 epoch : 3.4739
Iteration: 90, train_loss = 0.9786, test_loss = 1.1847, average time for 1 epoch : 3.6275
Iteration: 100, train_loss = 0.9784, test_loss = 1.1848, average time for 1 epoch : 3.4093
```

