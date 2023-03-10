import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

from module.data import Data

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

class ResChart:

    def __init__():
        
        pass

    def ebar(pred, true):

        res = np.subtract(pred, true)
        mean = np.mean(res)

        return res, mean
    
    def moving_range(res):

        list = []
        for i in range(0, len(res)-1):
            list.append(np.absolute(res[i+1] - res[i]))
        mrbar = np.mean(list)
        
        return mrbar
    
    def mae(res):
        list = []
        for i in range(0, len(res)-1):
            list.append(np.absolute(res[i]))
        mae = np.mean(list)

        return mae
    
    def control_limit(ebar, mae, l):

        ucl = np.float(ebar + l*mae)
        lcl = np.float(ebar - l*mae)

        return ucl, lcl
    
    def record_result(opt, size, phi, theta, shift, window_size, bs):

        data = Data()
        rc = ResChart

        test_loader, mean, std, targets = data.load_data_preds(size, phi, theta, shift, window_size, bs)
        
        preds, targets = opt.evaluate(test_loader)
        preds = preds*std + mean

        res, _ = rc.ebar(preds, targets)

        return res

    def arl0(model, size, phi, theta, shift, window_size, ucl, lcl):
        
        ''' 
        arl_flag = True : ARL0
        arl_flag = False : ARL1
        '''
        # Data Generation
        data = Data()
        data_loader, mean, std, targets = data.load_data_preds(size, phi, theta, shift, window_size, size)


        # Evaluate
        preds, targets = model.evaluate(data_loader)
        preds = preds*std + mean
        res = np.subtract(preds, targets)

        # rl
        rl = 0
        for item in res:
            if (item > ucl).any():
                break
            elif (item < lcl).any():
                break
            else:
                rl += 1
        
        return rl
    
    def arl1(model, size, phi, theta, shift, window_size, batch_size, ucl, lcl):

        data = Data()
                
        data_loader, mean, std, targets = data.load_data_preds(size, phi, theta, 0, window_size, batch_size)
        data_loader_shifted, mean_shifted, std_shifted, _ = data.load_data_preds(size, phi, theta, shift, window_size, batch_size)

        preds, targets = model.evaluate(data_loader)
        preds_shifted, _ = model.evaluate(data_loader_shifted)

        preds = preds*std + mean
        preds_shifted = preds_shifted*std_shifted + mean_shifted
        
        res = np.subtract(preds, targets)
        res_shifted = np.subtract(preds_shifted, targets)

        res_final = np.concatenate((res[:70], res_shifted[70:]))

        rl = 0
        sub_count = 0
        
        for item in res_final:
            if (item > ucl).any():
                break
            elif (item < lcl).any():
                break
            else:
                rl += 1
        
        if rl <=70:
            rl = 0
            sub_count = 1
        else:
            rl = rl - 70
    
        return rl, sub_count
            
        

