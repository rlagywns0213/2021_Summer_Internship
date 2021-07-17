# Stochastic Gradient Descent on SVD Model

![image](https://user-images.githubusercontent.com/28617444/126027877-992f50af-14be-4f55-a488-05124423c600.png)

## Compare With ALS

- In experiment, I use ```np.random.seed(0)``` for understanding Learning Algorithms
    - [Go To ALS](https://github.com/rlagywns0213/2021_Summer_Internship/tree/main/RecSys/ALS)

## Usage

```bash
python train.py
```

> initial configuration  
epochs: 100<br>
factor: 2<br>
lambda_param: 0.01<br>
learning_rate: 0.05


## Example  
```bash
python train.py --epochs 100 --factor 5 --lambda_param 0.01 --learning_rate 0.05
```

## Results

1. default

```bash
$ python train.py   

기존 loss: 2.6594469696143266
Iteration: 1, loss = 2.195954, time : 1.4623
Iteration: 11, loss = 1.000025, time : 15.2979
Iteration: 21, loss = 0.987084, time : 29.5726
Iteration: 31, loss = 0.983117, time : 44.7645
Iteration: 41, loss = 0.981293, time : 68.7507
Iteration: 51, loss = 0.980330, time : 98.5991
Iteration: 61, loss = 0.979769, time : 120.1321
Iteration: 71, loss = 0.979385, time : 134.4758
Iteration: 81, loss = 0.979080, time : 150.1975
Iteration: 91, loss = 0.978817, time : 170.9049
```
2. lr : 0.01
```bash
$ python train.py --epochs 100 --factor 5 --lambda_param 0.01 --learning_rate 0.01

기존 loss: 2.6594469696143266
Iteration: 1, loss = 2.441292, time : 2.3830
Iteration: 11, loss = 0.935807, time : 34.0210
Iteration: 21, loss = 0.910442, time : 64.4674
Iteration: 31, loss = 0.899001, time : 92.5779
Iteration: 41, loss = 0.892985, time : 120.4310
Iteration: 51, loss = 0.889490, time : 151.7462
Iteration: 61, loss = 0.887310, time : 183.7442
Iteration: 71, loss = 0.885873, time : 214.2618
Iteration: 81, loss = 0.884879, time : 245.6110
Iteration: 91, loss = 0.884159, time : 273.9916
```

3. factor : 15 lr : 0.01
```bash
$ python train.py --epochs 100 --factor 15 --lambda_param 0.01 --learning_rate 0.01

기존 loss: 15.497373776988232
Iteration: 1, loss = 2.142102, time : 1.3983
Iteration: 11, loss = 0.861622, time : 15.7199
Iteration: 21, loss = 0.807998, time : 29.8937
Iteration: 31, loss = 0.772904, time : 43.5717
Iteration: 41, loss = 0.747511, time : 58.1406
Iteration: 51, loss = 0.728002, time : 73.1187
Iteration: 61, loss = 0.712402, time : 87.8181
Iteration: 71, loss = 0.699645, time : 103.2539
Iteration: 81, loss = 0.689060, time : 118.8111
Iteration: 91, loss = 0.680182, time : 133.5687
```
4. factor : 100 lr : 0.01
```bash
$ python train.py --epochs 100 --factor 100 --lambda_param 0.01 --learning_rate 0.01

기존 loss: 99.7249859708129
Iteration: 1, loss = 3.468377, time : 1.4681
Iteration: 11, loss = 0.544069, time : 16.2426
Iteration: 21, loss = 0.436718, time : 30.2972
Iteration: 31, loss = 0.370918, time : 44.7987
Iteration: 41, loss = 0.324535, time : 59.8980
Iteration: 51, loss = 0.290478, time : 75.6638
Iteration: 61, loss = 0.264583, time : 91.5584
Iteration: 71, loss = 0.244322, time : 107.2103
Iteration: 81, loss = 0.228086, time : 122.1926
Iteration: 91, loss = 0.214762, time : 137.7542
```
- **ALS와는 달리, factor 수가 늘어나도 시간적 비용이 비슷하였음(확인 필요)**
