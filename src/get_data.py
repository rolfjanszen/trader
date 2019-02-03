import urllib.request
import json
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
file_base_path = '/home/rj/workspace/trader/price_data/'
# file_base_path = '/data/'
coins = ['USDT_BTC','USDT_NXT','USDT_LTC']

def download():
    start_time ="1483228800"
    period = '52560'

    for coin in coins:
        connct_to = "https://poloniex.com/public?command=returnChartData&currencyPair="+coin+"&end=9999999999&period="+period+"&start="+start_time
        btc = urllib.request.urlopen(connct_to).read()
        btc_file = open(file_base_path+coin+'.json','w')
        btc_str = btc.decode('utf8').replace("'", '"')
        j_btc = json.loads(btc_str)
        print('len set ', len(j_btc))
        json.dump(j_btc,btc_file)

def create_data(channels =3):
    print('Creating test data')
    data_len =20000
    val_change =[0.02,0.01]
    signs = [0.06 ,-0.05]
    total_set = []
    j=0
    last_ass_val = np.sin(-1*3.14)+np.cos(-1*6.28)
    for val in val_change:
        arr = []
        # asset = 1
        new_val=50
        for i in range(data_len):
            rand=0.6*(np.random.randint(10)-3)/(10+np.random.randint(100))
            new_val =new_val+signs[j]* (np.sin(i*3.14*val)*np.cos(i*0.28*val)) -signs[j] *np.cos(i*0.03)*np.sin(i*0.07)+0.015*np.cos(0.55*i)*np.sin(0.51*i)-rand
            if new_val < 0.01:
                new_val = 0.01
            # new_val = np.sin(i*3.14*val)+np.cos(i*6.28*val) +3
            # asset =(last_ass_val -new_val)/new_val
            entry = np.array([0.0] * 3, dtype=np.float)

            entry[0] = new_val+3
            entry[1] = new_val+3
            entry[2] = new_val+3

            # print(asset)
            if channels is 3 and len(arr) == 0:
                arr = np.array([entry])
            elif  channels is 1:
                arr.append(new_val)
            else:
                arr = np.append(arr, [entry], axis=0)
        arr *= 10
        j+=1
        if len(total_set) == 0:
            total_set = np.array([arr])
        else:
            total_set = np.append(total_set, [arr], axis=0)

        # plt.plot(total_set[0])
        # plt.show()
    print('Done creating test data')
    return total_set



def load_data(get_file = True):
    pkl_file ='crpt_data.pkl'

    if get_file:
        pickle_fp = open(file_base_path+pkl_file ,'rb')
        data = pkl.load(pickle_fp)

        return data

    total_set = []
    pickle_fp = open(file_base_path+pkl_file,'wb')

    for coin in coins:

        fp = open(file_base_path+coin+'.json','r')
        data_set = json.load(fp)
        print('len data_set',len(data_set))
        arr = []
        last_ass_val = data_set[0]
        for point in data_set[1:-1]:
            entry = np.array([1.0]*3,dtype=np.float)
            # entry[0] = (point['high']- last_ass_val['high'])/last_ass_val['high']
            # entry[1] = (point['low']- last_ass_val['low'])/last_ass_val['low']
            # entry[2] = (point['close'] - last_ass_val['close'])/last_ass_val['close']

            entry[0] = point['high']
            entry[1] = point['low']
            entry[2] = point['close']

            last_ass_val = point
            if len(arr) == 0:
                arr =np.array([entry])
            else:
                arr = np.append(arr,[entry], axis =0)

        # plt.plot(arr)
        # plt.show()
        if len(total_set) == 0:
            total_set=np.array([arr])
        else:
            total_set = np.append(total_set,[arr],axis= 0)

    # total_set = np.array(total_set, dtype = np.float)
    pkl.dump(total_set,pickle_fp)
    return  total_set


def ensure_same_length(coin_data):
    shortest = 100000000
    for coin in coin_data:
        if len(coin) < shortest:
            shortest = len(coin)


    for i in range(len(coin_data)):
        coin_data[i] = coin_data[i][:shortest]

    return coin_data

# data=load(False)
# download()
