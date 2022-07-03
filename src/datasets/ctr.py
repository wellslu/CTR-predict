import mlconfig
import random
import pandas as pd
import numpy as np
from torch.utils import data
from torchvision import datasets, transforms

class CTRDataset(data.Dataset):
    def __init__(self, transform, file, train):
        super(CTRDataset, self).__init__()
        self.train = train
        self.transform = transform
        #data_chunk = pd.read_csv(file, chunksize=5000000)
        #self.data = pd.DataFrame()
        #for chunk in data_chunk:
            #self.data = pd.concat([self.data, chunk])
        self.data = pd.read_csv(file)
        #self.data.drop(['site_id', 'device_id', 'device_ip', 'C14', 'C17', 'C21'], axis=1, inplace=True)
        self.targets = self.data['click'].to_numpy()
        self.data = self.data.drop('click', axis=1).to_numpy()
        self.length = len(self.data)
#         self.columns = list(self.data.columns)
#         self.columns.remove('click')
#         if train:
#             self.length = 10000000
#         else:
#             self.length = 1000000
    
#     def random_sample(self):
#         length = int(self.length / 2)
#         self.r_data = pd.concat([self.data[self.data['click']==0].sample(n=length), self.data[self.data['click']==1].sample(n=length)], ignore_index=True)
#         self.r_data = self.r_data.sample(frac=1).reset_index(drop=True)
#         for i, column in enumerate(self.columns):
#             self.r_data = pd.concat([self.r_data.drop([column], axis=1), pd.get_dummies(self.r_data[column])], axis=1)
#         self.targets = self.r_data['click'].to_numpy()
#         self.r_data = self.r_data.drop('click', axis=1).to_numpy()
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inputs = self.data[index]
        target = self.targets[index]
        
        return self.transform(inputs.reshape(1, -1)), target

@mlconfig.register
class CTRDataloader(data.DataLoader):

    def __init__(self, file: str, train: bool, batch_size: int, **kwargs):
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        dataset = CTRDataset(transform, file, train)

        super(CTRDataloader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=train, **kwargs)
    
#     def random_sample(self):
#         self.dataset.random_sample()
