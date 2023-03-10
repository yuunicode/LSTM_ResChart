import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

class Data:

    def __init__(self):

        pass

    def ar_1(self, size, phi, shift = 0):
        
        ''' Generate AR(1) data of size = size '''
        
        e = np.random.normal(0, 1, size)
        x = np.array(np.repeat(0, size), dtype=np.float64)
        x[0] = e[0] + shift
        for i in range(1, size):
            x[i] = x[i-1]*phi + e[i] + shift
        
        return x

    def ar_2(sself, size, phi, shift = 0):
    
        ''' Generate AR(2) data of size = size '''
        
        e = np.random.normal(0, 1, size)
        x = np.array(np.repeat(0, size), dtype=np.float64)
        
        x[0] = e[0]  + shift
        x[1] = x[0] * phi + e[1]
        for i in range(2, size):
            x[i] = x[i-1]*phi + x[i-2]* 0.2 + e[i]
        
        return x
            
    def arma_11(self, size, phi, theta, shift = 0):

        ''' Generate ARMA(1,1) data of size = size'''

        e = np.random.normal(0, 1, size)
        x = np.array(np.repeat(0, size), dtype=np.float64)

        x[0] = e[0] + shift
        for i in range(1, size):
            x[i] = x[i-1] * phi + e[i-1] * theta + e[i] + shift
    

        return x

    def polish_up(self, data, window_size):

        ''' 
        moving window of window_size
        '''
        
        size = data.size
        x = np.empty(shape=(size-window_size, window_size))
        y = np.empty(shape=(size-window_size, 1))

        for i in range(size-window_size):
            x[i] = data[i:i+window_size]
            y[i] = data[i+window_size]
     
        return x, y    
    
    def load_data_train(self, phi_list, theta_list, shift, window_size, train_size, vali_size, batch_size, seed=True): 
        
        train_data = []
        vali_data = []

        for phi in phi_list:
            for theta in theta_list:
                if seed: 
                    fixed_seed = 2023 + int(phi*100)
                    random.seed(fixed_seed)
                    np.random.seed(fixed_seed)
                    torch.manual_seed(fixed_seed)

                train = self.arma_11(int(train_size/(len(phi_list)*len(theta_list))), phi, theta, shift)
                vali = self.arma_11(int(vali_size/(len(phi_list)*len(theta_list))), phi, theta, shift)
                
                train_data.append(train)
                vali_data.append(vali)

        train_data = np.array(train_data).flatten()
        vali_data = np.array(vali_data).flatten()

        train_x, train_y = self.polish_up(train_data, window_size)
        vali_x, vali_y = self.polish_up(vali_data, window_size)
            
        scaler_x = StandardScaler()
        scaler_x.fit(train_x)
        train_x = scaler_x.transform(train_x)
        vali_x = scaler_x.transform(vali_x)

        scaler_y = StandardScaler()
        scaler_y.fit(train_y)
        train_y = scaler_y.transform(train_y)
        vali_y = scaler_y.transform(vali_y)

        train_x = Variable(torch.Tensor(train_x)).to(device)
        train_y = Variable(torch.Tensor(train_y)).to(device)
        vali_x = Variable(torch.Tensor(vali_x)).to(device)
        vali_y = Variable(torch.Tensor(vali_y)).to(device)

        train = TensorDataset(train_x, train_y)
        val = TensorDataset(vali_x, vali_y)

        train_loader = DataLoader(train, batch_size, shuffle=True)
        vali_loader = DataLoader(val, vali_size, shuffle=False) # False 로 돌려

        return train_loader, vali_loader
    
    def load_data_preds(self, size, phi, theta, shift, window_size, batch_size):

        data = self.arma_11(size, phi, theta, shift)
        X, Y = self.polish_up(data, window_size)
        target = Y
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        x = scaler_x.fit_transform(X)
        y = scaler_y.fit_transform(Y)
        x = torch.Tensor(x).to(device)
        y = torch.Tensor(y).to(device)
        mean, std = scaler_y.mean_, scaler_y.scale_
        
        dataset = TensorDataset(x, y)
        data_loader = DataLoader(dataset, batch_size, shuffle=False)

        return data_loader, mean, std, target
