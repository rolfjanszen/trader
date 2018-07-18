from stockmarket import StockMarketEnv
from RNN_PG import LSTMPolicyGrad
from sort_data import read_data, create_sine
import numpy as np
import matplotlib.pyplot as plt

RANGE =200
OUT_LEN = 4
model_file_path ='../output/model.ckpt'
input_file = '/home/rj/Documents/kaggle/bitcoin/BTC.csv'
data_in, data_out, close_data = read_data(input_file,RANGE, OUT_LEN, use_saved = False)
# data_in, data_out , close_data= create_sine(100, 60, RANGE, OUT_LEN)
env = StockMarketEnv(close_data, RANGE, RANGE)


n_x = RANGE
n_y = env.action_space

bot = LSTMPolicyGrad(chunk_size_ = 40, positions = 2,max_len_sent = RANGE, rnn_size_ = 40, input_sz_ = n_x, output_sz_ = n_y, save_path = 'models/crypto_ai')
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
    while T < 120 or (not done and T  < 100):

        env.render()

        action = bot.get_action(observation)
        #         action = env.action_space.sample()

        observation, reward, done = env.make_step(action)
        bot.gather_data(observation, reward)


        # observation = state['quote']

        sum_reward = sum_reward + reward
        new_action = np.zeros(n_y)
        #         print('new_action ',action)
        new_action[action] = 1

        T += 1
        if T > 200 :
            done = True
        print('final reward ', reward, 'action ',action, 'done ', done)
    sum_rewards.append(sum_reward)
    plt.plot(sum_rewards)
    plt.show(block=False)
    plt.pause(0.1)
    end_val = observation['quote'][-1]
    print('training sum_reward (profit)', sum_reward, ' buy n hold ', end_val - start_val, 'final reward ', reward)
    print('rewards ',bot.rewards)
    bot.train()