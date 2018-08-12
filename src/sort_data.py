import pandas as pd
import pickle as pkl
from math import*
import numpy as np



pkl_path = 'data.pkl'
def read_data(file_path, range_in, out_len, use_saved):
    closed =[]
    input = []
    output =[]
    
    if not use_saved:
        with open(file_path, 'r') as f:
            
            data = pd.read_csv(file_path, error_bad_lines=False)
            for row in data.itertuples():
                
                closed.append(row[6])
            closed_np = np. array(closed)
            
            close_min = np.min(closed_np)
            mean= np.mean(closed_np)
            std = np.std(closed_np)
            close_max = np.max(closed_np)
            # closed_np = (closed_np -mean)/std
            closed_np = (closed_np  - mean) / std

            data_len =len(closed_np)
            for i in range(data_len-range_in-out_len):
                input.append(closed_np[i:(i+range_in)])
                output.append(closed_np[(i+range_in+1):(i+range_in+out_len+1)])

        pkl_data = {'in':input,
                   'out':output}
        
  
        f = open(pkl_path,'wb')

        pkl.dump(pkl_data,f)

    else:
        f = open(pkl_path,'rb')
        pk_data = pkl.load(f)
        input =pk_data['in']
        output =pk_data['out']
    return input, output,input[0]


def create_sine(length, freq,range_in, out_len,):
    sine =[]
    input = []
    output =[]
    for i in range(length*freq):
        sine.append(sin(i*3.14/freq)*cos(i*6.28/freq))
        
    data_len =len(sine)
    for i in range(data_len-range_in-out_len):
        input.append(sine[i:(i+range_in)])
        output.append(sine[(i+range_in):(i+range_in+out_len)])
    
    return input, output, np.array(sine)

    