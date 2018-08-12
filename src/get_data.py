import urllib.request
import json
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
file_base_path = '/home/rj/workspace/trader/price_data/'
coins = ['USDT_BTC','USDT_NXT','USDT_LTC']
def download():
    start_time ="1514764800"
    period = '1800'

    for coin in coins:
        connct_to = "https://poloniex.com/public?command=returnChartData&currencyPair="+coin+"&end=9999999999&period="+period+"&start="+start_time
        btc = urllib.request.urlopen(connct_to).read()
        btc_file = open(file_base_path+coin+'.json','w')
        btc_str = btc.decode('utf8').replace("'", '"')
        j_btc = json.loads(btc_str)
        print('len set ', len(j_btc))
        json.dump(j_btc,btc_file)


def create_data(channels =3):
    data_len =20000
    val_change =[0.03,0.009]
    total_set = []
    last_ass_val = np.sin(-1*3.14)+np.cos(-1*6.28)
    for val in val_change:
        arr = []
        asset = 1
        for i in range(data_len):
            rand=1+np.random.randint(10)/1000
            new_val = np.sin(i*3.14*val)+np.cos(i*6.28*val) +3
            asset =(last_ass_val -new_val)/new_val
            entry = np.array([0.0] * 3, dtype=np.float)
            entry[0] = asset
            entry[1] = asset
            entry[2] = asset
            last_ass_val = new_val
            # print(asset)
            if channels is 3 and len(arr) == 0:
                arr = np.array([entry])
            elif  channels is 1:
                arr.append(asset)
            else:
                arr = np.append(arr, [entry], axis=0)
        arr *= 10
        if len(total_set) == 0:
            total_set = np.array([arr])
        else:
            total_set = np.append(total_set, [arr], axis=0)
    # total_set[1] = total_set[1]
    total_set[1] =-total_set[0]
        # plt.plot(total_set[0])
        # plt.show()
    return total_set


def load_data(get_file = True):
    pkl_file ='crpt_data.pkl'

    if get_file:
        pickle_fp = open(file_base_path+pkl_file ,'rb')
        data = pkl.load(pickle_fp)
        return data

    total_set = []
    pickle_fp = open(file_base_path+pkl_file,'wb')
    coins = coins[0]
    for coin in coins:

        fp = open(file_base_path+coin+'.json','r')
        data_set = json.load(fp)
        print('len data_set',len(data_set))
        arr = []
        last_ass_val = data_set[0]
        for point in data_set[1:-1]:
            entry = np.array([0.0]*3,dtype=np.float)
            entry[0] = (point['high']- last_ass_val['high'])/last_ass_val['high']
            entry[1] = (point['low']- last_ass_val['low'])/last_ass_val['low']
            entry[2] = (point['close'] - last_ass_val['close'])/last_ass_val['close']
            last_ass_val = point
            if len(arr) == 0:
                arr =np.array([entry])
            else:
                arr = np.append(arr,[entry], axis =0)
        arr*=100
        if len(total_set) == 0:
            total_set=np.array([arr])
        else:
            total_set = np.append(total_set,[arr],axis= 0)

    # total_set = np.array(total_set, dtype = np.float)
    pkl.dump(total_set,pickle_fp)
    return  total_set



# data=load(False)
# download()
