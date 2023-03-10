import numpy as np
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def _lr_step_decay(epoch, lr):

    drop_rate = 0.1
    epochs_drop = 100.0

    return lr * math.pow(drop_rate, math.floor(epoch/epochs_drop))

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size, device):

        super(LSTM, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size = input_size, 
                            hidden_size = hidden_size, 
                            num_layers = num_layers, 
                            batch_first = True)
                
        self.fc = nn.Sequential(nn.Linear(hidden_size, int(hidden_size/4)), 
                                nn.Dropout(0.5), 
                                nn.Linear(int(hidden_size/4), output_size)
                                )
        
    def reset_hidden_state(self):
        self.hidden = (Variable(torch.zeros(self.num_layers, self.hidden_size)).to(device), 
                       Variable(torch.zeros(self.num_layers, self.hidden_size)).to(device))
        
    def forward(self, x):

        h0 = torch.zeros(self.num_layers, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, self.hidden_size).to(device)
        
        output, _ = self.lstm(x, (h0, c0))
        out = self.fc(output)

        return out
    
class LSTM_01(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size, device):

        super(LSTM_01, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size = input_size, 
                            hidden_size = hidden_size, 
                            num_layers = num_layers, 
                            batch_first = True)
                
        self.fc = nn.Sequential(nn.Linear(hidden_size, int(hidden_size/4)), 
                                nn.Dropout(0.5), 
                                nn.Linear(int(hidden_size/4), output_size)
                                )
        
    def reset_hidden_state(self):
        self.hidden = (Variable(torch.zeros(self.num_layers, self.hidden_size)).to(device), 
                       Variable(torch.zeros(self.num_layers, self.hidden_size)).to(device))
        
    def forward(self, x):

        h0 = torch.zeros(self.num_layers, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, self.hidden_size).to(device)
        
        output, _ = self.lstm(x, (h0, c0))
        out = self.fc(output)

        return out

class Optimization:
    
    def __init__(self, model, loss_fn, optimizer, learning_rate, learning_rate_decay):
        
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr = learning_rate
        self.train_losses= []
        self.val_losses = []
        self.learning_rate_decay = learning_rate_decay
        
    def train_step(self, x, y):
        
        self.model.train()
                
        yhat = self.model(x)
        
        loss = self.loss_fn(y, yhat)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
            
        return loss.item()
    
    def train(self, train_loader, vali_loader, epochs=300, patience=20):
       
        count = 0

        for epoch in range(1, epochs+1):
            batch_losses = []
            for i, (train_x, train_y) in enumerate(train_loader):
                self.model.reset_hidden_state()
                loss = self.train_step(train_x, train_y)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            if self.learning_rate_decay:
                lr = _lr_step_decay(epoch, self.optimizer.param_groups[0]['lr'])
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            
            with torch.no_grad():
                batch_val_losses = []
                for i, (vali_x, vali_y) in enumerate(vali_loader):
                    self.model.eval()
                    yhat = self.model(vali_x)
                    val_loss = self.loss_fn(vali_y, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

                if min(self.val_losses) < self.val_losses[epoch-1]:
                    count += 1
                    print(f'Early Stopping in: {count}/{patience}')
                else: count = 0

            if count == patience:
                print('\n Early Stopping')
                break
            
            if (epoch <= 10) | (epoch % 30 == 0) | (epoch == epochs):
                print(f"[{epoch}/{epochs}] Training loss: {training_loss:.4f}/t Validation loss: {validation_loss:.4f}")
                                        
    def evaluate(self, test_loader):
        
        with torch.no_grad():
            preds = []
            targets = []
            for i, (test_x, test_y) in enumerate(test_loader):
                self.model.eval()
                pred = self.model(test_x)
                pred = pred.detach().cpu().numpy().flatten()
                target = test_y.detach().cpu().numpy().flatten()
                for item in pred:
                    preds.append(item)
                for item in target:
                    targets.append(item)
        preds = np.array(preds)
        targets = np.array(targets)
        preds = preds.reshape(-1,1)
        targets = targets.reshape(-1,1)
        return preds, targets
    
    def plot_losses(self):

        plt.plot(self.train_losses, label="training Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.legend()
        plt.title("Losses")
        plt.show()

