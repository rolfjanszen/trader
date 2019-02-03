import numpy as np
import matplotlib.pyplot as plt
import random

class StockMarketEnv:

    portfolio_len = 12
    capital_record = []
    action_space = 1

    trans_cost = 0.002
    devalue = 0.9

    time = 0
    def __init__(self, set_quotes_, start_time, time_range, channels, period_length_,testing_):
        self.portfolio_len= len(set_quotes_)
        # collect_quotes = []
        # for quotes in set_quotes_:
        #     collect_quotes.append(quotes)
        self.period_length = period_length_
        self.stock_history = np.array(set_quotes_)
        if testing_:
            self.stock_history[0] = 60-self.stock_history[1]


        self.price_changes = []
        self.channels =channels
        self.max_time = self.stock_history.shape[1]
        self.begin_time = start_time
        self.state_sz = time_range

        fiat_0 = 1

        self.fiat = np.array([[0.0]*channels]*(1+time_range))
        for i in range(time_range,0,-1):
            self.fiat[i] = fiat_0
            fiat_0 = fiat_0 * self.devalue

        self.action_space = self.portfolio_len + 1
        self.portfolio=np.array([0]*self.action_space)
        # self.portfolio[-1] = 1 #last element is fiat currency
        self.capital = 1
        self.portfolio = np.array([0.01] * self.action_space)
        self.portfolio[-1] = 0.98  # last element is fiat currency

        self.pre_process()
        # self.time = random.randint(self.begin_time, self.max_time - self.begin_time - 200)
        # plt.show(block=False)

    def render(self, stock):
        graph = self.stock_history[stock, self.time: self.time + self.state_sz]
        plt.clf()
        plt.plot(graph)
        plt.draw()
        plt.pause(0.1)


    def pre_process(self):
        print('preprocess')
        for quote in self.stock_history:
            print('stock_history ',self.stock_history.shape)
            if self.channels < 2:
                price_change = (quote[1:] - quote[:-1]) / quote[:-1]
            else:
                price_change = (quote[1:, :] - quote[:-1, :]) / quote[:-1, :]
            self.price_changes.append(price_change)
            print('price_change ', np.array(self.price_changes).shape)
        self.price_changes = np.array(self.price_changes)


    def normalize_quote(self):
        quote = np.copy(self.stock_history[:, self.time: self.time + self.state_sz])

        for i in range(len(quote)):
            quote[i] = (quote[i]- quote[i,0])/quote[i,0]

        return dict({'quote' :quote , 'position':self.portfolio})


    def profit(self,old_capital):
        if self.channels < 2:
            price_change = self.price_changes[:, self.time]
        else:
            price_change = self.price_changes[:, self.time, 2]


        gains = np.sum(self.portfolio[:-1] *price_change)#-self.portfolio[-1]*(1-self.devalue)

        self.capital += gains * self.capital
        # if self.capital < 0.01 or self.capital != self.capital:
        #     self.capital = 1
        reward = (self.capital-old_capital)/old_capital
        reward = self.capital -1
        return reward*100


    def reset(self):
        if self.period_length >= self.max_time:
            self.period_length = self.max_time
            self.start_time = 0
        else:
            self.start_time = self.time#random.randint( self.state_sz+1, self.max_time -  self.period_length - 200)

        self.time = self.start_time

        self.capital = 1  # everybody gets a second chance
        self.capital_record =[]
        norm_quote = self.normalize_quote()

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

        if (self.time + self.state_sz+1)  >= self.max_time:
            self.time = self.begin_time
            done = True
        self.profit(old_capital)

        self.capital_record.append(self.capital)
        reward = 100*(self.capital -1)/(self.time-self.start_time)
        # reward = (np.random.randint(20, size=1)-10)[0]
        norm_quote = self.normalize_quote()

        # reward = self.capital -1
        return norm_quote , reward ,done

    def get_quotes(self):
        close_quote = self.stock_history[:, self.start_time: self.start_time + self.period_length,2]
        # quote = quote
        for i in range(close_quote.shape[0]):
            close_quote[i] = close_quote[i]/close_quote[i,0]

        print('get quote ', close_quote)
        return close_quote


    def plot_results(self, ax, colors):

        performances = []
        i=0
        for quote in self.stock_history[:,self.start_time : self.start_time + self.period_length]:
            new_quote = quote / quote[0,2]
            performances.append(new_quote[-1])
            ax.plot(new_quote, label=colors[i])
            i+=1
        ax.plot(self.capital_record,'--r')

        print('capital left', self.capital,'performances ', performances)