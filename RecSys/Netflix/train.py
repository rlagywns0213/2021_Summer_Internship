import time
import numpy as np
import argparse
from sklearn.metrics import mean_squared_error
from models import Matrix_Factorization
from models_bias import Matrix_Factorization_bias
import dataset
import yaml

with open('configuration.yaml') as f:
  configuration = yaml.load(f)

epochs = configuration['epochs']
factor = configuration['factor']
lambda_param = configuration['lambda_param']
learning_rate = configuration['learning_rate']

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', required = False, type=int, default=epochs,
                    help='# of epochs (default = 10)')
parser.add_argument('--factor', required = False, type=int, default=factor,
                    help='factor (default = 2)')
parser.add_argument('--lambda_param', required = False, type=float, default=lambda_param,
                    help='lambda_param (default = 0.01)')
parser.add_argument('--learning_rate', required = False, type=float, default=learning_rate,
help='learning_rate (default = 0.01)')
parser.add_argument('--bias',action='store_true', default=False,
                    help='for adding bias')

args = parser.parse_args()


# load data
data = dataset.Dataset()
train = data.train
test = data.test

if args.bias == True:
  als = Matrix_Factorization_bias(train, test, factor = args.factor, lambda_param=args.lambda_param, epochs=args.epochs, learning_rate=args.learning_rate)
  print("adding bias")

else:
  als = Matrix_Factorization(train, test, factor = args.factor, lambda_param=args.lambda_param, epochs=args.epochs, learning_rate=args.learning_rate)

  print("No adding bias")

als.fit()
