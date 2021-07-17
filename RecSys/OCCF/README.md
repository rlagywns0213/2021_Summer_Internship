# Implementation of Collaborative Filtering for Implicit Feedback Datasets

For a summary of the paper, click [here](https://github.com/rlagywns0213/2021_Summer_Internship/tree/main/RecSys/OCCF/Paper_Summary)

![image](https://user-images.githubusercontent.com/28617444/126036848-319deb23-c545-4a17-a2a2-d4ff8996ebf9.png)

- The explicit datasets, movielens were used and implemented as implicit datasets.
- Test_rank : sort by prediction of test set

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
Iteration: 1, loss = 0.935235, test_rank = 0.4112, time : 14.6921
Iteration: 11, loss = 0.782257, test_rank = 0.3198, time : 165.4987
Iteration: 21, loss = 0.782209, test_rank = 0.3225, time : 316.8405
Iteration: 31, loss = 0.782206, test_rank = 0.3205, time : 461.1011
Iteration: 41, loss = 0.782206, test_rank = 0.3202, time : 603.4584
Iteration: 51, loss = 0.782206, test_rank = 0.3200, time : 746.7500
Iteration: 61, loss = 0.782206, test_rank = 0.3200, time : 890.8059
Iteration: 71, loss = 0.782206, test_rank = 0.3202, time : 1031.6739
Iteration: 81, loss = 0.782206, test_rank = 0.3201, time : 1174.6270
Iteration: 91, loss = 0.782206, test_rank = 0.3201, time : 1319.9767
```

2. factor : 100
```bash
$ python train.py --factor 100

기존 loss: 99.7249859708129
Iteration: 1, loss = 0.590910, test_rank = 0.3857, time : 34.1154
Iteration: 11, loss = 0.504560, test_rank = 0.4017, time : 456.6179
Iteration: 21, loss = 0.500221, test_rank = 0.4043, time : 886.4854
Iteration: 31, loss = 0.498456, test_rank = 0.4069, time : 1310.5543
Iteration: 41, loss = 0.497475, test_rank = 0.4071, time : 1727.1409
Iteration: 51, loss = 0.496836, test_rank = 0.4060, time : 2139.9794
Iteration: 61, loss = 0.496377, test_rank = 0.4092, time : 2547.7033
Iteration: 71, loss = 0.496039, test_rank = 0.4064, time : 2963.9670
Iteration: 81, loss = 0.495783, test_rank = 0.4057, time : 3377.2253
Iteration: 91, loss = 0.495580, test_rank = 0.4110, time : 3779.6455
```