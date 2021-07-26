import time
import numpy as np
import argparse
from models import Probabilistic_MF
import dataset
import yaml

with open('configuration.yaml') as f:
  configuration = yaml.load(f)

epochs = configuration['epochs']
factor = configuration['factor']
learning_rate = configuration['learning_rate']
lambda_u = configuration['lambda_u']
lambda_v = configuration['lambda_v']


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', required = False, type=int, default=epochs,
                    help='# of epochs (default = 10)')
parser.add_argument('--factor', required = False, type=int, default=factor,
                    help='factor (default = 2)')
parser.add_argument('--learning_rate', required = False, type=float, default=learning_rate,
                    help='learning_rate (default = 0.01)')
parser.add_argument('--lambda_u', required = False, type=float, default=lambda_u,
                    help='lambda_u (default = 0.01)')
parser.add_argument('--lambda_v', required = False, type=float, default=lambda_v,
                    help='lambda_v(default = 0.01)')
args = parser.parse_args()


# load data
data = dataset.Dataset()
train = data.train
test = data.test


pmf = Probabilistic_MF(train,test, factor = args.factor, lambda_u=args.lambda_u, lambda_v=args.lambda_v, epochs=args.epochs, learning_rate=args.learning_rate)

pmf.fit()
