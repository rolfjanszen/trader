from stockmarket import StockMarketEnv
from PG_CNN import CNNPolicyGrad
import matplotlib.pyplot as plt
from get_data import load_data, download, create_data
import numpy as np


def show_results(earned, fig , actions, capital, sub_title):
    # fig = plt.figure()
    fig.clf()
    fig.suptitle(sub_title, fontsize=16)
    # plt.clf()

    ax = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.cla()
    colors = ['r', 'g', 'y', 'b', '--']
    # env.plot_results(ax2, colors)
    ax.cla()
    actions = np.array(actions)
    capital = np.array(capital)
    # ax.plot(actions)
    # ax.plot(capital)


    for i in range(actions.shape[1]):
        print('i',i)
        ax.plot(actions[:,i],label =colors[i])
    # ax.legend()
    print('actionss',actions.shape,'cap ', capital.shape)
    plt.show(block=False)
    for i in range(capital.shape[0]):
        ax2.plot(capital[i,:],label =colors[i])

    ax2.plot(earned,label ='--')
        # ax.legend()
    plt.show(block=False)

def loop_env(env, bot, trade_period):

    observation = env.reset()

    done = False
    sum_reward = 0
    # sum_rewards =[]
    T = 0.
    # test_outputs = []
    eps_capital = []

    while T < trade_period and not done:

        action = bot.get_action(observation)

        observation, reward, done = env.make_step(action)
        bot.gather_data(observation, reward)

        sum_reward = sum_reward + reward

        T += 1
        if T > trade_period :
            done = True
        if done:
            break

        eps_capital.append(action)

    print('action ',action, 'done ', done,' remaining capital ', env.capital, 'sum_reward ', sum_reward)
    # sum_rewards.append(sum_reward)
    # market = env.get_result()

    plt.pause(0.1)
    end_val = observation['quote'][-1]
    # env.reset()
    return eps_capital


def main():
    # download()

    channels = 3
    trade_period = 600
    test_period = 16000
    time_range = 100 #length historic data

    use_test_data = False
    if use_test_data:
        quotes_set = create_data(channels)
    else:
        quotes_set = load_data(get_file=True)

    test_size = int(0.15 * quotes_set.shape[1])
    test_quotes = np.array(quotes_set[:, -test_size:, :])
    train_quotes = np.array(quotes_set[:, :-test_size, :])
    env = StockMarketEnv(train_quotes, 10, time_range, channels, trade_period, testing_=use_test_data)
    test_env = StockMarketEnv(test_quotes, 10, time_range, channels, test_period, testing_=use_test_data)

    n_x = time_range
    n_y = env.action_space
    n_pos = env.portfolio_len
    model_file_path = 'output/crypto_trader9.ckpt'

    bot = CNNPolicyGrad(chunk_size_=10, positions=n_pos, max_len_sent=n_x, rnn_size_=10, input_sz_=n_x, action_sz=n_y,
                        channels_=channels, save_path=model_file_path, name_='train')
    # test_bot = CNNPolicyGrad(chunk_size_ = 10, positions =n_pos,max_len_sent = n_x, rnn_size_ = 10, input_sz_ = n_x, action_sz = n_y,channels_ =  channels, save_path = None, name_='test')

    fig = plt.figure()
    test_fig = plt.figure()
    score_fig = plt.figure().subplots()
    testScore = []
    for i_episode in range(20000):

        eps_capital = loop_env(env, bot, trade_period)
        eps_capital = env.get_quotes()
        show_results(env.capital_record, fig, bot.actions, eps_capital, 'training results')
        bot.train()
        bot.reset()
        if i_episode % 1 == 0:

            test_capital = loop_env(test_env,bot, test_period)
            test_capital = test_env.get_quotes()
            show_results(test_env.capital_record, test_fig, bot.actions, test_capital, 'testing results')
            profit = test_env.capital - 1
            testScore.append(profit)

            score_fig.plot(testScore)
            bot.reset()

if __name__ == '__main__':
    main()