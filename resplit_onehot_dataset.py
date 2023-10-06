import pandas as pd
import numpy as np

transfer_list = ['site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model', 'C14', 'C17', 'C19', 'C20', 'C21']
all_data = pd.concat([pd.read_csv("train.csv"), pd.read_csv("validation.csv"), pd.read_csv("test.csv")], ignore_index=True)
columns = all_data.columns[1:-1]
for col in columns:
    if col in transfer_list:
        continue
    all_data = pd.concat([all_data.drop(col, axis=1), pd.get_dummies(all_data[col])], axis=1)
all_data[all_data['id'].notnull()].drop('click', axis=1).to_csv('./data/test_hot.csv', index=False)
all_data = all_data[all_data['click'].notnull()].drop('id', axis=1)
all_data = all_data.sample(frac=1).reset_index(drop=True)
all_data[:int(len(all_data)*0.2)].to_csv('./data/validation_hot.csv', index=False)
all_data[int(len(all_data)*0.2):].to_csv('./data/train_hot.csv', index=False)
train = pd.read_csv("./data/train_hot.csv", chunksize=10000000)
for i, df in enumerate(train):
    df.to_csv(f'./data/train_hot_{i}.csv', index=False)
