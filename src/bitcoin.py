from rnn_model import RnnModel
from sort_data import read_data, create_sine
import matplotlib.pyplot as plt

RANGE =50
OUT_LEN = 1
model_file_path ='../output/model.ckpt'
input_file = '/home/rj/Documents/kaggle/bitcoin/BTC.csv'
data_in, data_out,  all_data = read_data(input_file,RANGE, OUT_LEN, use_saved = True)
# plt.plot(data_in)
# plt.show()
# data_in, data_out , all_data= create_sine(100, 60, RANGE, OUT_LEN)
# new_arra = data_in[10] + data_out[10]
plt.plot(all_data)
# plt.plot(data_out[10])
# plt.show(block = False)

rnn = RnnModel(chunk_size_ = 1, max_len_sent = RANGE, rnn_size_ = 40, classes= OUT_LEN)
rnn.train_neural_network(data_in, data_out )
pred  = rnn.make_stock_prediction(data_in)

correct = [0]
for i in range(len(data_in)):
    correct.append(data_in[i][0])
plt.plot(pred)
plt.plot(correct)
plt.show()