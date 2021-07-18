import time
import numpy as np
import argparse
from sklearn.metrics import mean_squared_error
from models import occf
from models2 import occf_bottleneck
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
parser.add_argument('--bottleneck',action='store_true', default=False,
                    help='for computational bottleneck')
args = parser.parse_args()


# load data
data = dataset.Dataset()
train = data.train
test = data.test

if args.bottleneck == True:
  occf = occf_bottleneck(train,test, factor = args.factor, lambda_param=args.lambda_param, epochs=args.epochs)
  print("Reduce bottleneck problem")

else:
  occf = occf(train,test, factor = args.factor, lambda_param=args.lambda_param, epochs=args.epochs)
  print("bottleneck problem")

occf.fit()
