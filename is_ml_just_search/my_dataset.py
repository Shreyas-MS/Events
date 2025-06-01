import torch
import numpy as np
import pandas as pd
# y = 2*a +0b - c/2
def create_dataset(num_samples, get_tensor = False):
    X = np.random.randn(num_samples, 3)
    y = 2*X[:,0]-X[:,2]/2

    if get_tensor:
        return torch.tensor(X,dtype=torch.float32),torch.tensor(y,dtype=torch.float32)
        
    X = pd.DataFrame(X,columns=['a','b','c'])
    y = pd.DataFrame(y,columns=['target'])

    return X,y