# Implementation of SVD (comparison of adding bias)

## Basic VS Adding Bias

1) Basic (No adding bias)
![image](https://user-images.githubusercontent.com/28617444/126273649-2096f725-fa8e-4060-818d-e4742c7c9adc.png)

2) Adding Bias
![image](https://user-images.githubusercontent.com/28617444/126274547-a31b3d30-77da-4751-afc0-a47ae6c15625.png)

```python
self._b_U = np.zeros(self._num_users) # user bias
self._b_V = np.zeros(self._num_items) # item bias
self._b = np.mean(self._R[np.where(self._R != 0)]) # rating means
```
## Issue

If not scaling in initalizing Latent facotr, overflow 
```bash
RuntimeWarning: overflow encountered in multiply  dp = (error * self._V[i, :]) - (self._lambda_param * self._U[u, :])
c:\Users\rlagy\2021_DSAIL\Recommend System\Netflix\models.py:60: RuntimeWarning: overflow encountered in multiply  dq = (error * self._U[u, :]) - (self._lambda_param * self._V[i, :])
```

- So scaling was used to implement it.
```python
self._U = np.random.normal(scale = 1.0/self._factor,size=(self._num_users, self._factor))
self._V = np.random.normal(scale = 1.0/self._factor, size=(self._num_items, self._factor))
```

## Results

| Model | 50 factors | 100 factors | 200 factors|
| :---: | :---: |  :---: |  :---: |
| SVD | 1.7441 |  1.3774 |  1.3466 |
| SVD adding bias | 1.5231 |  1.3019 |  1.2893 |

- **When the bias was added, it was possible to confirm the improvement of the test loss.**
- **Also, there was no difference in "average time for 1 epoch".**

## Usage

```bash
python train.py
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

## Results

1. No adding bias

```bash
$ python train.py   
No adding bias
기존 loss: 0.781466335477013
Iteration: 10, train_loss = 0.6696, test_loss = 2.0753, average time for 1 epoch : 1.2698
Iteration: 20, train_loss = 0.6088, test_loss = 1.8867, average time for 1 epoch : 1.2345
Iteration: 30, train_loss = 0.5908, test_loss = 1.8309, average time for 1 epoch : 1.2659
Iteration: 40, train_loss = 0.5812, test_loss = 1.8013, average time for 1 epoch : 1.2799
Iteration: 50, train_loss = 0.5752, test_loss = 1.7827, average time for 1 epoch : 1.3412
Iteration: 60, train_loss = 0.5709, test_loss = 1.7694, average time for 1 epoch : 1.2852
Iteration: 70, train_loss = 0.5678, test_loss = 1.7598, average time for 1 epoch : 1.4558
Iteration: 80, train_loss = 0.5656, test_loss = 1.7529, average time for 1 epoch : 1.3488
Iteration: 90, train_loss = 0.5640, test_loss = 1.7480, average time for 1 epoch : 1.3307
Iteration: 100, train_loss = 0.5628, test_loss = 1.7441, average time for 1 epoch : 1.3007
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