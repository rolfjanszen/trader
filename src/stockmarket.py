import numpy as np
import matplotlib.pyplot as plt
import random

class StockMarketEnv:

    time = 2
    action_space = 3
    bougth_at = 0
    bought = False
    portfolio = [0]*2
    init_price = 0
    def __init__(self, stock_history_, start_time, range):
        self.stock_history = stock_history_
        self.time = start_time
        self.max_time = len(stock_history_)
        self.begin_time = start_time
        self.state_sz = range
        # plt.show(block=False)

    def render(self):
        graph = self.stock_history[self.time: self.time + self.state_sz]
        # plt.clf()
        # plt.plot(graph)
        # plt.draw()
        # plt.pause(0.001)

    def normalize_quote(self):
        quote = self.stock_history[self.time: self.time + self.state_sz]
        # m_q = np.mean(quote)
        # st_q = np.std(quote)
        # new_q =( quote - m_q)/st_q
        return dict({'quote' :quote , 'portfolio':self.portfolio})

    def profit(self):
        change_buy = self.stock_history[self.time] - self.bougth_at
        buyNhold = self.stock_history[self.time] - self.init_price
        reward = (change_buy - buyNhold)
        return reward


    def reset(self):
        self.bought = False
        self.time = random.randint( self.begin_time, self.max_time -  self.begin_time - 200)
        self.init_price = self.stock_history[self.time]
        self.portfolio = [0,1]
        print('self.time ',self.time)
        return self.normalize_quote()

    def make_step(self, action):
        new_price = self.stock_history[self.time]
        old_price = self.stock_history[self.time - 1]
        change = old_price - new_price


        if (self.time + 1)  >= self.max_time:
            self.time = self.begin_time
            return self.normalize_quote(), self.profit(), True

        self.time += 1
        if action == 1 and not self.bought : #buy
            self.bought = True
            self.bougth_at = self.stock_history[self.time]
            self.portfolio[0] = 1
            self.portfolio[1] = 0
            return self.normalize_quote(), 0, False
        elif action == 2 and self.bought: #sell
            self.portfolio[0] = 0
            self.portfolio[1] = 1
            self.bought = False
            return self.normalize_quote(), self.profit(), True
        else: #hold
            return self.normalize_quote() , 0, False