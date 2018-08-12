from stockmarket import StockMarketEnv
from RNN_PG import LSTMPolicyGrad
from sort_data import read_data, create_sine
import numpy as np
import matplotlib.pyplot as plt
from get_data import load_data, download, create_data

# download()
channels = 3
RANGE =50
OUT_LEN = 4
model_file_path ='../output/trader.ckpt'
input_file = '/home/rj/Documents/kaggle/bitcoin/BTC.csv'
# data_in, data_out, close_data = read_data(input_file,RANGE, OUT_LEN, use_saved = False)
# # data_in, data_out , close_data_sin= create_sine(100, 60, RANGE, OUT_LEN)
#
# quotes_set = [close_data,close_data*3,close_data*10]
quotes_set = load_data()
# quotes_set = create_data(channels)
env = StockMarketEnv(quotes_set, RANGE, RANGE, channels)

n_x = RANGE
n_y = env.action_space
model_file_path = None
bot = LSTMPolicyGrad(chunk_size_ = 10, positions =n_y,max_len_sent = RANGE, rnn_size_ = 10, input_sz_ = n_x, output_sz_ = n_y,channels_ =  channels)#, save_path = model_file_path)
sum_rewards =[]

for i_episode in range(20000):

    observation = env.reset()

    print('len observation ',len(observation['quote']))

    actions = []
    states = []
    rewards = []
    print('new try')
    done = False
    sum_reward = 0
    start_val = observation['quote'][-1]
    T = 0
    test_outputs = []
    eps_capital = []
    end_truth = []
    trade_period = 1000
    while T < trade_period or (not done and T  < trade_period):

        # env.render(0)

        action = bot.get_action(observation)
        # action = env.action_space.sample()
        # action=np.array([0.25])
        observation, reward, done , test_out= env.make_step(action)
        bot.gather_data(observation, reward)
        test_outputs.append(test_out)
        # print(' reward', reward, ' observation ', observation['position'])
        # observation = state['quote']
        # print('goal: ', observation['quote'][:,-1])
        sum_reward = sum_reward + reward
        end_truth.append(test_out)
        T += 1
        if T > trade_period :
            done = True
        eps_capital.append(action)
    print('sum_reward ', sum_reward, 'action ',action, 'done ', done,' remaining capital ', env.capital)
    sum_rewards.append(sum_reward)

    if channels > 1:
        plt.plot(np.array(end_truth)[:,:,0])
    else:
        plt.plot(np.array(end_truth))
    plt.show(block=False)
    plt.plot(eps_capital)
    plt.draw()
    # plt.plot(sum_rewards)
    # plt.show(block=False)
    plt.pause(0.1)
    end_val = observation['quote'][-1]
    # print('training sum_reward (profit)', sum_reward,  'final reward ', reward)
    # print(' capital ',env.capital, 'rewards ',bot.rewards)

    # if channels > 1:
    #     bot.actions = np.array(test_outputs)[:,:,0]
    # else:
    #     bot.actions = np.array(test_outputs)
    bot.train()