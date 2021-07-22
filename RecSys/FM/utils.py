import torch
import time
from torch.utils.data import Dataset, DataLoader

### Dataloader

class MovielensDataset(Dataset):
    """
    torch.utils.data.Dataset 상속
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return self.X.size(0)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

def get_data_loader(batch_size, X_train, y_train, X_test, y_test):
   torch.manual_seed(0)
   train_dataset = MovielensDataset(X = torch.FloatTensor(X_train.toarray()),
                                y = y_train)
   test_dataset = MovielensDataset(X = torch.FloatTensor(X_test.toarray()),
                                y = y_test)
   train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
   test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
   return train_loader, test_loader


###
def train(model, train_loader, optimizer, criterion):

   model.train()
   total_loss = 0
   
   for i, (input, label) in enumerate(train_loader):
      y_pred = model(input)
      optimizer.zero_grad()
      loss = criterion(y_pred.flatten().float(), label.float())
      # compute gradient and do SGD step
      loss.backward()
      optimizer.step()
      total_loss += loss
   end = time.time() #epoch time
   return total_loss
#   print('==>  Train Loss: {loss:.3f}, 1 epoch time : {time:.3f}'.format(loss=total_loss,time=end-start))

def validation(model, test_loader, criterion):
      model.eval()
      criterion = criterion
      for i, (input, label) in enumerate(test_loader):
         output = model(input)
         criterion_ = criterion(output.flatten().float(),label.float())
         RMSE = torch.sqrt(criterion_)
      
 #     print('==> Test RMSE {:.3f}'.format(RMSE))
      return RMSE