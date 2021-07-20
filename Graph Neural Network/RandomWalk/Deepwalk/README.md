# Implementation of DeepWalk: Online Learning of Social Representations

## RandomWalk + SkipGram

- DeepWalk Algorithm

![image](https://user-images.githubusercontent.com/28617444/126334651-5b0a0a32-a5cb-4d55-aa58-ce6d621982e3.png)

![image](https://user-images.githubusercontent.com/28617444/126335324-56194444-37cd-4fd1-87c2-ee7d3b1f158b.png)

## About

- I used `gensim` library to use SkipGram model
- I used `sklearn.decomposition` to visualize
    - Number of latent dimensions to learn for each node > 2 : Principal component analysis 2D Projection

## Usage

```bash
python train.py 
```

> initial configuration  
num_walks: 10<br>
represent_size: 2<br>
walk_length: 40<br>
window_size: 5

## Example  
```bash
python train.py --represent_size 4
```

## Results

```bash
python train.py --represent_size 4
Target Graph: karate
Walking...
modeling time: 0.0170 
Traning...
modeling time: 0.0409 
Visualize Results...
Graph image saved!
```

