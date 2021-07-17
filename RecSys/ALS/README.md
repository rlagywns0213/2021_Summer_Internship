# Alternating Least Squares (ALS) on SVD Model

![image](https://user-images.githubusercontent.com/28617444/126021210-ef650205-24fc-4491-911b-978ac6ee34fb.png)

## Compare With SGD
    
- In experiment, I use ```np.random.seed(0)``` for understanding Learning Algorithms 
    - [Go To SGD](https://github.com/rlagywns0213/2021_Summer_Internship/tree/main/RecSys/ALS)
    
## Usage

```bash
python train.py
```

> initial configuration  
epochs=100
factor=2
lambda_param = 0.01


## Example  
```bash
python train.py --epochs 100 --factor 5 --lambda_param 0.01
```

## Results

1. default

```bash
$ python train.py   

기존 loss: 2.6594469696143266
Iteration: 1, loss = 0.605044, time : 0.9794
Iteration: 11, loss = 0.520694, time : 3.9203
Iteration: 21, loss = 0.520619, time : 6.7991
Iteration: 31, loss = 0.520616, time : 8.6362
Iteration: 41, loss = 0.520614, time : 10.5217
Iteration: 51, loss = 0.520613, time : 12.4505
Iteration: 61, loss = 0.520612, time : 14.3744
Iteration: 71, loss = 0.520612, time : 16.1706
Iteration: 81, loss = 0.520611, time : 17.9828
Iteration: 91, loss = 0.520611, time : 21.8057
```

2. factor : 15
```bash
$ python train.py --factor 15

기존 loss: 15.497373776988232
Iteration: 1, loss = 0.533825, time : 0.3760
Iteration: 11, loss = 0.409911, time : 3.8134
Iteration: 21, loss = 0.409838, time : 7.0643
Iteration: 31, loss = 0.409810, time : 10.2219
Iteration: 41, loss = 0.409786, time : 13.5610
Iteration: 51, loss = 0.409768, time : 16.8124
Iteration: 61, loss = 0.409756, time : 20.0622
Iteration: 71, loss = 0.409749, time : 22.9266
Iteration: 81, loss = 0.409745, time : 25.9660
Iteration: 91, loss = 0.409743, time : 28.9546
```

3. factor : 100
```bash
$ python train.py --factor 100

기존 loss: 99.7249859708129
Iteration: 1, loss = 0.335995, time : 2.5956
Iteration: 11, loss = 0.229574, time : 31.0992
Iteration: 21, loss = 0.228984, time : 65.2534
Iteration: 31, loss = 0.228855, time : 99.6673
Iteration: 41, loss = 0.228801, time : 135.1778
Iteration: 51, loss = 0.228773, time : 164.2999
Iteration: 61, loss = 0.228756, time : 196.1487
Iteration: 71, loss = 0.228746, time : 231.6902
Iteration: 81, loss = 0.228740, time : 268.3997
Iteration: 91, loss = 0.228736, time : 303.7614
```

- **factor 수가 늘어나면서, time과 loss의 trade off 관계를 확인할 수 있음**