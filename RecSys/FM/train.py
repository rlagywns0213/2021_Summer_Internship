import time
import torch.nn as nn
from torch import optim
import argparse
from models import FactorizationMachine
from utils import *
import dataset
import yaml

with open('configuration.yaml') as f:
  configuration = yaml.load(f)

epochs = configuration['epochs']
k = configuration['k']
learning_rate = configuration['learning_rate']
batch_size = configuration['batch_size']
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', required = False, type=int, default=epochs,
                    help='# of epochs (default = 10)')
parser.add_argument('--batch_size', required = False, type=int, default=batch_size,
                    help='batch_size (default = 64)')
parser.add_argument('--k', required = False, type=int, default=k,
                    help='v의 hyperparameter (default = 5)')
parser.add_argument('--learning_rate', required = False, type=float, default=learning_rate,
help='learning_rate (default = 0.01)')


args = parser.parse_args()


# load data
X_train, X_test, y_train, y_test, train_users, train_items, test_users, test_items = dataset.vectorize()
train_loader, test_loader = get_data_loader(batch_size = args.batch_size, X_train = X_train,  y_train= y_train, X_test = X_test, y_test = y_test)

#model
_, num_feature= X_train.shape
num_v_feature = args.k
model = FactorizationMachine(num_feature, num_v_feature)
optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
criterion = nn.MSELoss()

#train, test
time_list = 0 
best_rmse = 1000
base_start = time.time()
for epoch in range(args.epochs):
   start = time.time()
   total_loss = train(model, train_loader, optimizer, criterion)
   RMSE = validation(model ,train_loader, criterion)
   time_list += time.time()-start
   
   if ((epoch + 1) % 10 == 0 ):
        end = time.time()
        print ('Epoch {} of {}, training Loss: {:.4f}, Test RMSE: {:.4f}'.format(epoch + 1, args.epochs, total_loss, RMSE))
        print("===> 1 epoch mean Time : {:.3f}".format(time_list/10))
        time_list=0

   if best_rmse > RMSE:
      best_rmse = RMSE
      #torch.save(model.state_dict(), 'model_best.pt')
print("Best RMSE :{}, ".format(best_rmse))
print("1 EPOCH MEAN TIME :{}, ".format((time.time()-base_start)/args.epochs))