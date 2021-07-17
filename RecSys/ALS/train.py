import time
import numpy as np
import argparse
from sklearn.metrics import mean_squared_error
from models import als_basic
import dataset
import yaml

with open('configuration.yaml') as f:
  configuration = yaml.load(f)

epochs = configuration['epochs']
factor = configuration['factor']
lambda_param = configuration['lambda_param']

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', required = False, type=int, default=epochs,
                    help='# of epochs (default = 10)')
parser.add_argument('--factor', required = False, type=int, default=factor,
                    help='factor (default = 2)')
parser.add_argument('--lambda_param', required = False, type=float, default=lambda_param,
                    help='lambda_param (default = 0.01)')
args = parser.parse_args()


# load data
data = dataset.Dataset()
train = data.train
test = data.test


als = als_basic(train, factor = args.factor, lambda_param=args.lambda_param, epochs=args.epochs)

als.fit()
