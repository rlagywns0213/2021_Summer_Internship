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


```bash
$ python train.py --epochs 50 --factor 40
bottleneck problem
기존 loss: 40.24770828391712
time to compute rank : 0.1506
Iteration: 1, loss = 0.742698, test_rank = 0.3872, time : 27.4694
time to compute rank : 0.1655
Iteration: 2, loss = 0.614921, test_rank = 0.3640, time : 30.5546
time to compute rank : 0.1556
Iteration: 3, loss = 0.591714, test_rank = 0.3725, time : 30.3122
time to compute rank : 0.1596
Iteration: 4, loss = 0.582945, test_rank = 0.3714, time : 30.3857
time to compute rank : 0.1536
Iteration: 5, loss = 0.578381, test_rank = 0.3761, time : 29.8251
time to compute rank : 0.1606
Iteration: 6, loss = 0.575621, test_rank = 0.3827, time : 30.3143
time to compute rank : 0.1566
Iteration: 7, loss = 0.573788, test_rank = 0.3770, time : 29.6165
time to compute rank : 0.1606
Iteration: 8, loss = 0.572478, test_rank = 0.3836, time : 33.0068
time to compute rank : 0.1536
Iteration: 9, loss = 0.571488, test_rank = 0.3806, time : 34.3357
time to compute rank : 0.1745
Iteration: 10, loss = 0.570710, test_rank = 0.3811, time : 34.3407
time to compute rank : 0.1616
Iteration: 11, loss = 0.570084, test_rank = 0.3835, time : 33.5701
time to compute rank : 0.1626
Iteration: 12, loss = 0.569571, test_rank = 0.3880, time : 37.0905
time to compute rank : 0.1685
Iteration: 13, loss = 0.569146, test_rank = 0.3873, time : 38.2352
time to compute rank : 0.1705
Iteration: 14, loss = 0.568788, test_rank = 0.3842, time : 34.6016
time to compute rank : 0.1616
Iteration: 15, loss = 0.568484, test_rank = 0.3864, time : 34.0437
time to compute rank : 0.1636
Iteration: 16, loss = 0.568221, test_rank = 0.3846, time : 32.8388
time to compute rank : 0.1556
Iteration: 17, loss = 0.567992, test_rank = 0.3849, time : 36.7249
time to compute rank : 0.1606
Iteration: 18, loss = 0.567790, test_rank = 0.3905, time : 32.9783
time to compute rank : 0.1646
Iteration: 19, loss = 0.567610, test_rank = 0.3847, time : 35.3301
time to compute rank : 0.1506
Iteration: 20, loss = 0.567449, test_rank = 0.3892, time : 34.8355
time to compute rank : 0.1546
Iteration: 21, loss = 0.567304, test_rank = 0.3861, time : 37.6830
time to compute rank : 0.1686
Iteration: 22, loss = 0.567172, test_rank = 0.3845, time : 45.9498
time to compute rank : 0.2074
Iteration: 23, loss = 0.567052, test_rank = 0.3836, time : 42.7077
time to compute rank : 0.1586
Iteration: 24, loss = 0.566942, test_rank = 0.3834, time : 44.5225
time to compute rank : 0.1666
Iteration: 25, loss = 0.566841, test_rank = 0.3874, time : 42.3046
time to compute rank : 0.1526
Iteration: 26, loss = 0.566748, test_rank = 0.3861, time : 36.7452
time to compute rank : 0.1735
Iteration: 27, loss = 0.566663, test_rank = 0.3833, time : 36.8729
time to compute rank : 0.1676
Iteration: 28, loss = 0.566583, test_rank = 0.3902, time : 33.6852
time to compute rank : 0.1596
Iteration: 29, loss = 0.566510, test_rank = 0.3837, time : 36.5404
time to compute rank : 0.1765
Iteration: 30, loss = 0.566442, test_rank = 0.3884, time : 35.0993
time to compute rank : 0.1616
Iteration: 31, loss = 0.566378, test_rank = 0.3842, time : 54.0581
time to compute rank : 0.1716
Iteration: 32, loss = 0.566320, test_rank = 0.3895, time : 34.7156
time to compute rank : 0.1556
Iteration: 33, loss = 0.566266, test_rank = 0.3874, time : 42.4118
time to compute rank : 0.1636
Iteration: 34, loss = 0.566216, test_rank = 0.3881, time : 40.0040
time to compute rank : 0.1805
Iteration: 35, loss = 0.566169, test_rank = 0.3853, time : 51.4110
time to compute rank : 0.1546
Iteration: 36, loss = 0.566126, test_rank = 0.3877, time : 56.5436
time to compute rank : 0.1676
Iteration: 37, loss = 0.566087, test_rank = 0.3817, time : 41.9102
time to compute rank : 0.1616
Iteration: 38, loss = 0.566049, test_rank = 0.3861, time : 34.7290
time to compute rank : 0.1676
Iteration: 39, loss = 0.566015, test_rank = 0.3881, time : 31.4193
time to compute rank : 0.1666
Iteration: 40, loss = 0.565983, test_rank = 0.3865, time : 29.5089
time to compute rank : 0.1486
Iteration: 41, loss = 0.565953, test_rank = 0.3853, time : 31.4307
time to compute rank : 0.1646
Iteration: 42, loss = 0.565924, test_rank = 0.3865, time : 31.9078
time to compute rank : 0.1715
Iteration: 43, loss = 0.565898, test_rank = 0.3878, time : 33.3811
time to compute rank : 0.1606
Iteration: 44, loss = 0.565873, test_rank = 0.3830, time : 30.8008
time to compute rank : 0.1676
Iteration: 45, loss = 0.565849, test_rank = 0.3810, time : 34.5724
time to compute rank : 0.1666
Iteration: 46, loss = 0.565827, test_rank = 0.3849, time : 34.4319
time to compute rank : 0.1546
Iteration: 47, loss = 0.565805, test_rank = 0.3832, time : 29.4574
time to compute rank : 0.1566
Iteration: 48, loss = 0.565785, test_rank = 0.3855, time : 30.4178
time to compute rank : 0.1606
Iteration: 49, loss = 0.565766, test_rank = 0.3910, time : 29.5163
time to compute rank : 0.1456
Iteration: 50, loss = 0.565747, test_rank = 0.3911, time : 30.1177
```

### Additional Implementation

- Computational bottleneck is computing Y_T, C_U, Y
- So i did a significant speedup (Y_T*Y + Y_T(C_U-I)Y) 

```bash
$ python train.py --epochs 50 --factor 40 --bottleneck
Reduce bottleneck problem
기존 loss: 40.24770828391712
time to compute rank : 0.1695
Iteration: 1, loss = 0.742698, test_rank = 0.3872, time : 79.3030
time to compute rank : 0.1656
Iteration: 2, loss = 0.614921, test_rank = 0.3640, time : 78.9446
time to compute rank : 0.1676
Iteration: 3, loss = 0.591714, test_rank = 0.3725, time : 82.0225
time to compute rank : 0.1965
Iteration: 4, loss = 0.582945, test_rank = 0.3714, time : 85.8256
time to compute rank : 0.1685
Iteration: 5, loss = 0.578381, test_rank = 0.3761, time : 92.4284
time to compute rank : 0.1695
Iteration: 6, loss = 0.575621, test_rank = 0.3827, time : 87.0184
time to compute rank : 0.1765
Iteration: 7, loss = 0.573788, test_rank = 0.3770, time : 89.3223
time to compute rank : 0.1685
Iteration: 8, loss = 0.572478, test_rank = 0.3836, time : 89.2158
time to compute rank : 0.1626
Iteration: 9, loss = 0.571488, test_rank = 0.3806, time : 100.4290
time to compute rank : 0.1685
Iteration: 10, loss = 0.570710, test_rank = 0.3811, time : 93.9412
time to compute rank : 0.1676
Iteration: 11, loss = 0.570084, test_rank = 0.3835, time : 87.9799
time to compute rank : 0.2204
Iteration: 12, loss = 0.569571, test_rank = 0.3880, time : 90.2553
time to compute rank : 0.1875
Iteration: 13, loss = 0.569146, test_rank = 0.3873, time : 93.0134
time to compute rank : 0.2274
Iteration: 14, loss = 0.568788, test_rank = 0.3842, time : 97.2952
time to compute rank : 0.2015
Iteration: 15, loss = 0.568484, test_rank = 0.3864, time : 103.2011
time to compute rank : 0.1695
Iteration: 16, loss = 0.568221, test_rank = 0.3846, time : 81.6100
time to compute rank : 0.1715
Iteration: 17, loss = 0.567992, test_rank = 0.3849, time : 80.6860
time to compute rank : 0.1666
Iteration: 18, loss = 0.567790, test_rank = 0.3905, time : 78.6135
time to compute rank : 0.1705
Iteration: 19, loss = 0.567610, test_rank = 0.3847, time : 82.6830
time to compute rank : 0.1685
Iteration: 20, loss = 0.567449, test_rank = 0.3892, time : 78.5741
time to compute rank : 0.1576
Iteration: 21, loss = 0.567304, test_rank = 0.3861, time : 64.4374
time to compute rank : 0.1446
Iteration: 22, loss = 0.567172, test_rank = 0.3845, time : 62.1357
time to compute rank : 0.2822
Iteration: 23, loss = 0.567052, test_rank = 0.3836, time : 63.2589
...
```

### ToDO

- I have to change (C_U-I) only n_u non_zero elements.