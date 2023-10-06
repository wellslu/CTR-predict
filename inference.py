import torch
from src.models import MLP
import pandas as pd
import numpy as np

test_chunk = pd.read_csv('./data/test.csv', chunksize=1000000)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP()
model.load_state_dict(torch.load('./best.pth')['model'])

if __name__ == "__main__":
	drop_lits = ['site_id', 'device_id', 'device_ip', 'C14', 'C17', 'C21']
	ids = []
	clicks = []
	for test in test_chunk:
	    test = test.reset_index(drop=True)
	    ids = ids + test['id'].to_list()
	    test.drop('id', axis=1, inplace=True)
	#     test.drop(drop_lits, axis=1, inplace=True)
	#     columns = test.columns
	#     for i, column in enumerate(columns):
	#         test = pd.concat([test.drop([column], axis=1), pd.DataFrame(np.eye(ohe_list[i]+1)[test[column].astype('int').to_list()])], axis=1)
	    test = torch.from_numpy(test.to_numpy())
	    test = test.float()
	    with torch.no_grad():
	        for inputs in test:
	            y = model(inputs.unsqueeze(0))
	            clicks.append(round(y[0].item(), 5))
	dtype={'id': np.dtype(int),
	    'click': np.dtype(float),
	      }
	submit = pd.read_csv("../sampleSubmission.gz", dtype=dtype)
	submit['click'] = clicks
	submit.to_csv('answer.csv', index=False)
