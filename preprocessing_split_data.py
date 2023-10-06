import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

test = pd.read_csv("./test.gz")

train = pd.read_csv("train.gz", chunksize=5000000)
sample = pd.DataFrame()
for s in train:
    sample = pd.concat([sample, s])

print(sample.isna().sum()) # no null
print(sample.click.value_counts()) # 0    33563901, 1     6865066

def transfer(df, col, mean, df_t):
    d = df.groupby([col])['click'].agg('mean').to_dict()
    for key in d.keys():
        if d[key] < mean-0.09:
            d[key] = 1
        elif mean-0.09 <= d[key] < mean-0.07:
            d[key] = 2
        elif mean-0.07 <= d[key] < mean-0.05:
            d[key] = 3
        elif mean-0.05 <= d[key] < mean-0.03:
            d[key] = 4
        elif mean-0.03 <= d[key] < mean-0.01:
            d[key] = 5
        elif mean-0.01 <= d[key] < mean+0.01:
            d[key] = 6
        elif mean+0.01 <= d[key] < mean+0.03:
            d[key] = 7
        elif mean+0.03 <= d[key] < mean+0.05:
            d[key] = 8
        elif mean+0.05 <= d[key] < mean+0.07:
            d[key] = 9
        elif mean+0.07 <= d[key] < mean+0.09:
            d[key] = 10
        else:
            d[key] = 11
    df[col] = df[col].apply(lambda x: d[x])
    df_t[col] = df_t[col].apply(lambda x: d[x] if x in d.keys() else 6)
    return df, df_t

def standardization(df, col, df_t):
    a_dictionary = df.groupby([col])['click'].agg('count').to_dict()
    max_key = max(a_dictionary, key=a_dictionary.get)
    d = {}
    num = 1
    for v in set(df[col]):
        d[v] = num
        num+=1
    df[col] = df[col].apply(lambda x: d[x])
    df_t[col] = df_t[col].apply(lambda x: d[x] if x in d.keys() else 0)
    return df, df_t

if __name__ == "__main__":
	# remove below 10%
	diff_value_list = []
	for column in sample.columns[3:]:
	    diff = list(set(test[column].to_list()).difference(set(sample[column].to_list())))
	    if len(test[test[column].isin(diff)]) > len(test)*0.01:
	        diff_value_list.append(column)
	
	sample.drop('id', axis=1, inplace=True)
	sample.drop(diff_value_list, axis=1, inplace=True)
	test.drop(diff_value_list, axis=1, inplace=True)
	
	sample['hour'] = sample['hour'].apply(lambda x: int(int(str(x)[-2:])/2))
	test['hour'] = test['hour'].apply(lambda x: int(int(str(x)[-2:])/2))
	
	transfer_list = []
	for column in sample.columns[2:]:
	    length = len(set(sample[column].to_list()))
	    if length >= 10:
	        transfer_list.append(column)
	    print(column, ' : ', length)
	
	mean = sample['click'].describe()['mean']
	for col in transfer_list:
	    sample, test = transfer(sample, col, mean, test)
	
	standard_list = set(sample.columns[1:].to_list()).difference(set(transfer_list))
	for col in standard_list:
	    sample, test = standardization(sample, col, test)
	
	test.to_csv('./data/test.csv', index=False)
	sample = sample.sample(frac=1).reset_index(drop=True)
	sample[:int(len(sample)*0.2)].to_csv('./data/validation.csv', index=False)
	sample[int(len(sample)*0.2):].to_csv('./data/train.csv', index=False)
