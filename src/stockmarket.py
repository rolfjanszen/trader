import numpy as np
import matplotlib.pyplot as plt
import random

class StockMarketEnv:

    portfolio_len = 12
    portfolio =np.array([])
    time = 2
    action_space = 1
    bougth_at = 0
    bought = False
    # portfolio = [0]*2
    # init_price = 0
    stock_history = np.array([])
    trans_cost = 0.0007
    devalue = 0.9
    time = 200

    def __init__(self, set_quotes_, start_time, time_range, channels):
        self.portfolio_len= len(set_quotes_)
        # collect_quotes = []
        # for quotes in set_quotes_:
        #     collect_quotes.append(quotes)

        self.stock_history = np.array(set_quotes_)
        self.channels =channels
        self.max_time = self.stock_history.shape[1]
        self.begin_time = start_time
        self.state_sz = time_range

        fiat_0 = 1

        self.fiat = np.array([[0.0]*channels]*(1+time_range))
        for i in range(time_range,0,-1):
            self.fiat[i] = fiat_0
            fiat_0 = fiat_0 * self.devalue

        self.action_space = self.portfolio_len #+1
        self.portfolio=np.array([0]*self.action_space)
        # self.portfolio[-1] = 1 #last element is fiat currency
        self.capital = 1
        self.portfolio = np.array([0.01] * self.action_space)
        self.portfolio[-1] = 0.98  # last element is fiat currency
        self.start_time = self.time
        self.quote_t0 =self.stock_history
        # self.time = random.randint(self.begin_time, self.max_time - self.begin_time - 200)
        # plt.show(block=False)

    def render(self, stock):
        graph = self.stock_history[stock, self.time: self.time + self.state_sz]
        plt.clf()
        plt.plot(graph)
        plt.draw()
        plt.pause(0.1)

    def normalize_quote(self):
        quote = np.copy(self.stock_history[:, self.time: self.time + self.state_sz+1])
        plt.clf()

        # for i in range(len(self.quote_t0 )):
        #     # quote_t0 = self.stock_history[i, self.time + self.state_sz] #TODO change to 2?
        #     quote[i]= quote[i]/self.quote_t0 [i]
            # plt.plot(quote[i])
            # plt.show(block= False)
            # plt.pause(0.1)

        # quote = self.stock_history[:, self.time: self.time + self.state_sz]/ self.stock_history[:, self.time + self.state_sz,2]
        # if self.action_space > 3:
        #     add_fiat = quote, (self.fiat[np.newaxis, :] - 0.97) * 100
        #     if self.channels < 2:
        #
        #         add_fiat =((self.fiat[np.newaxis,:][0]-0.97)*100).T
        #     quote = np.append(quote, add_fiat, axis=0)

        return dict({'quote' :quote , 'position':self.portfolio})

    def profit(self,old_capital):
        # if self.channels < 2:
        #     price_change = (self.stock_history[:, self.time] - self.stock_history[:, self.time - 1
        #                                                           ]) / self.stock_history[:, self.time - 1]
        # else:
        #     price_change = (self.stock_history[:, self.time,2]-self.stock_history[:, self.time-1,2])/self.stock_history[:, self.time-1,2]

        if self.channels < 2:
            price_change = self.stock_history[:, self.time]
        else:
            price_change = self.stock_history[:, self.time, 2]

        gains = np.sum(self.portfolio *price_change)#-self.portfolio[-1]*(1-self.devalue)

        self.capital += gains * self.capital
        if self.capital < 0.01 or self.capital != self.capital:
            self.capital = 1
        reward = (self.capital-old_capital)/self.capital

        return reward*1000


    def reset(self):
        self.bought = False
        if self.time > (self.max_time -  self.begin_time - 200):
            self.time = 100
            self.start_time=100
            self.portfolio = np.array([0.01] * self.action_space)
            self.portfolio[-1] = 0.98  # last element is fiat currency

        # self.time = random.randint( self.begin_time, self.max_time -  self.begin_time - 200)
        # self.init_price = self.stock_history[:, self.time]

        if self.capital < 0.01 or self.capital != self.capital:

            self.start_time = 100
        self.capital = 1  # everybody gets a second chance
        norm_quote = self.normalize_quote()
        norm_quote['quote' ]= norm_quote['quote' ][:][:,0:-1]
        self.quote_t0 = self.stock_history[:,self.time]
        return norm_quote

    def make_step(self, action):
        old_capital = self.capital
        self.capital -= self.capital*self.trans_cost*np.sum((np.abs(self.portfolio-action)))
        self.portfolio = action

        new_price = self.stock_history[:, self.time]
        old_price = self.stock_history[:, self.time - 1]
        change = old_price - new_price

        done = False
        self.time += 1
        if (self.time + 1)  >= self.max_time:
            self.time = self.begin_time
            done =True
        self.profit(old_capital)
        reward = 100*(self.capital -1)/(self.time-self.start_time)
        # reward = (np.random.randint(20, size=1)-10)[0]
        norm_quote = self.normalize_quote()
        norm_quote['quote'] = norm_quote['quote'][:][:, 0:-1]
        test_out = norm_quote['quote'][:,-1]
        return norm_quote , reward ,done,  test_out
