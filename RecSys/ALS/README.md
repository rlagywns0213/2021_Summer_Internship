# Alternating Least Squares (ALS)

![image](https://user-images.githubusercontent.com/28617444/126021210-ef650205-24fc-4491-911b-978ac6ee34fb.png)


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
기존 loss: 2.848284424351714
Iteration: 1, loss = 0.673627, time : 0.1795
Iteration: 11, loss = 0.520789, time : 1.3850
Iteration: 21, loss = 0.520620, time : 2.5522
Iteration: 31, loss = 0.520617, time : 3.7277
Iteration: 41, loss = 0.520615, time : 5.0121
Iteration: 51, loss = 0.520614, time : 6.1835
Iteration: 61, loss = 0.520613, time : 7.3545
Iteration: 71, loss = 0.520612, time : 8.5221
Iteration: 81, loss = 0.520611, time : 9.6823
Iteration: 91, loss = 0.520611, time : 10.8405
```

2. factor : 15
```bash
$ python train.py
기존 loss: 15.61308535071082
Iteration: 1, loss = 0.522849, time : 0.3308
Iteration: 11, loss = 0.410287, time : 2.9945
Iteration: 21, loss = 0.409881, time : 5.6949
Iteration: 31, loss = 0.409789, time : 8.3139
Iteration: 41, loss = 0.409761, time : 10.9561
Iteration: 51, loss = 0.409750, time : 13.6452
Iteration: 61, loss = 0.409745, time : 16.2007
Iteration: 71, loss = 0.409743, time : 18.7975
Iteration: 81, loss = 0.409741, time : 21.4234
Iteration: 91, loss = 0.409741, time : 24.0526
```

- **factor 수가 늘어나면서, time과 loss의 trade off 관계를 확인할 수 있음**