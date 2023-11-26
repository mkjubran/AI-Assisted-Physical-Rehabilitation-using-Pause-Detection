
import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import pdb
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

start_time = datetime.now()

class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 6400)
        self.fc2 = nn.Linear(6400, 3200)
        self.fc3 = nn.Linear(3200, 1600)
        self.fc4 = nn.Linear(1600, 800)
        self.fc5 = nn.Linear(800, 400)
        self.fc6 = nn.Linear(400, 200)
        self.fc7 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        return F.log_softmax(self.fc7(x), dim=1)


def LoadData(l,s,a):
   for label in np.arange(10):
       print(f"Loading E{label} Data")
       tmp = np.load(f"../PDSaveData_Downloaded/SavedData_E{label}_l{l}_s{s}_a{a}.npy",allow_pickle=True)
       if label == 0:
          Zload=tmp.copy()
       else:
          Zload=np.concatenate((Zload,tmp),axis=0)

   X=Zload[:,:-1]
   y=Zload[:,-1]
   return X,y


if __name__=='__main__':
   torch.manual_seed(42)

   # Check if CUDA (GPU) is available
   if torch.cuda.is_available():
       device = torch.device("cuda")
   else:
       device = torch.device("cpu")

   # You can print the selected device for verification
   print("Using device:", device)

   #Parameters
   l = 11
   s = 4
   a = 50

   batch_size = 100
   MaxEpoch = 3000

   X,y = LoadData(l,s,a)
   
   input_size=X.shape[1]
   print(f"feature vector length (input_size) = {input_size}")

   # Split the data into training and testing sets
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   tensor_x_train = torch.Tensor(X_train) # transform to torch tensor
   tensor_y_train = torch.Tensor(y_train)

   tensor_x_test = torch.Tensor(X_test) # transform to torch tensor
   tensor_y_test = torch.Tensor(y_test)

   dataset_train = TensorDataset(tensor_x_train,tensor_y_train) # create your datset
   dataset_test = TensorDataset(tensor_x_test,tensor_y_test) # create your datset

   trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=1)
   testloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=1)

   model = MLP(input_size).to(device)
   criterion = nn.NLLLoss()
   optimizer = optim.SGD(model.parameters(), lr=1e-4)

   for epoch in range(0,MaxEpoch):

      #Training
      current_loss = 0.0
      current_correct = 0.0

      # Set the model to train mode
      model.train()

      for i, data in enumerate(trainloader, 0):
          inputs, targets = data
          inputs, targets = inputs.float().to(device), targets.long().to(device)

          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs, targets)
          loss.backward()
          optimizer.step()

          current_loss += loss.item()
          output = outputs.argmax(dim=1).float()
          current_correct += (output == targets).float().sum() 

          #if i%10 == 0:
          #   print(f"Epoch {epoch+1}/{MaxEpoch} - minibatch {i+1}, Loss: {current_loss/((i+1)*batch_size):.4f}, Accuracy: {100*current_correct/((i+1)*batch_size):.2f}%")

      print(f"Epoch {epoch+1}/{MaxEpoch} - Training, Loss: {current_loss/(len(trainloader)*batch_size):.4f}, Accuracy: {100*current_correct/(len(trainloader)*batch_size):.2f}%")

      writer.add_scalar("Loss/train", current_loss/(len(trainloader)*batch_size), epoch)
      writer.add_scalar("Accuracy/train", 100*current_correct/(len(trainloader)*batch_size), epoch)

      #Testing
      current_loss = 0.0
      current_correct = 0.0

      # Set the model to evaluation mode
      model.eval()

      # Disable gradient computation for evaluation
      with torch.no_grad():
         for i, data in enumerate(testloader, 0):
             inputs, targets = data
             inputs, targets = inputs.float().to(device), targets.long().to(device)

             outputs = model(inputs)
             loss = criterion(outputs, targets)

             current_loss += loss.item()
             output = outputs.argmax(dim=1).float()
             current_correct += (output == targets).float().sum() 
     
      print(f"Epoch {epoch+1}/{MaxEpoch} - Testing, Loss: {current_loss/(len(testloader)*batch_size):.4f}, Accuracy: {100*current_correct/(len(testloader)*batch_size):.2f}%\n")

      writer.add_scalar("Loss/test", current_loss/(len(testloader)*batch_size), epoch)
      writer.add_scalar("Accuracy/test", 100*current_correct/(len(testloader)*batch_size), epoch)

   print("Training has completed")

print("--- Time: %s  ---" % (datetime.now() - start_time))

writer.flush()
writer.close()
