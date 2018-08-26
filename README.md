# trader

uses a policy gradient method with a CNN to learn how to manage a portfolio of crypto currencies.

Works well on reasonably complex syntetic data that has some randomness in its price. This includes a transaction fee of 0.2%.
Tried batch normalization on several occasions but this prevents the model from converging. So its not used.
Have not yet trained it for a long time on real data, with which it still strugles.

The model is a 5 layered CNN with tanh activation (relu doesn't work)
Based on a paper "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem" by Zhengyao Jiang,
Dixing Xu, Jinjun Liang https://arxiv.org/pdf/1706.10059.pdf

To download data from poloniex use the download function in get_data.py

ro run: python3 crypto_ai.py

Require:
tensorflow
numpy
matplotlib
pickle
