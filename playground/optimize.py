import os
import sys

sys.path.insert(0, os.path.abspath("../../tensortrade"))
import ta
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np

from tensortrade.feed.core import Stream, DataFeed, NameSpace
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import USD, BTC
from tensortrade.oms.wallets import Wallet, Portfolio
import tensortrade.env.default as default


from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C


def load_csv(filename):
    df = pd.read_csv("data/" + filename, skiprows=1)
    df.drop(columns=["symbol", "volume_btc"], inplace=True)
    # Fix timestamp form "2019-10-17 09-AM" to "2019-10-17 09-00-00 AM"
    df["date"] = df["date"].str[:14] + "00-00 " + df["date"].str[-2:]
    # Convert the date column type from string to datetime for proper sorting.
    df["date"] = pd.to_datetime(df["date"])
    # Make sure historical prices are sorted chronologically, oldest first.
    df.sort_values(by="date", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    # Format timestamps as you want them to appear on the chart buy/sell marks.
    df["date"] = df["date"].dt.strftime("%Y-%m-%d %I:%M %p")

    return df


def log_and_diff(df, cols):
    for col in cols:
        df[col] = np.log(df[col]) - np.log(df[col]).shift(1)


# result = adfuller(df["close"].values[1:], autolag="AIC")
# print(result)
# print("p-value: %f" % result[1])

# plt.subplot(211)
# plt.plot(df["close"])
# plt.subplot(212)
# plt.plot(df["open"])
# plt.show()

prices = load_csv("Coinbase_BTCUSD_1h.csv")
prices = prices.head(1000)
df = prices
print(df.head())
df.drop(columns=["date"], inplace=True)
df = ta.add_all_ta_features(df, "open", "high", "low", "close", "volume", fillna=True)
df = df + 1 - 2 * df.min()  # Make positive
log_and_diff(df, df.columns)
df.drop(0, inplace=True)
prices.drop(0, inplace=True)
# df.plot()
# plt.show()
# Process
# print(df[["open", "high", "low", "volume"]].head())

# # Setup trading env
coinbase = Exchange("coinbase", service=execute_order)(
    Stream.source(prices["close"].tolist(), dtype="float").rename("USD-BTC")
)
portfolio = Portfolio(
    USD,
    [Wallet(coinbase, 10000 * USD)],
)


with NameSpace("coinbase"):
    streams = [Stream.source(df[c].tolist(), dtype="float").rename(c) for c in df.columns]
feed = DataFeed(streams)
print(feed.next())

# # Screen log

env = default.create(
    portfolio=portfolio,
    action_scheme="managed-risk",
    reward_scheme="risk-adjusted",
    feed=feed,
    renderer="screen-log",  # ScreenLogger used with default settings
    window_size=20,
)
print(env.observation_space)
print(env.action_space)

# model = A2C(MlpLstmPolicy, env, verbose=1)
# # model.learn(total_timesteps=25000)

# obs = env.reset()
# done = False
# while not done:
#     action, _states = model.predict(obs)
#     obs, rewards, done, info = env.step(action)
#     env.render()


# from tensortrade.agents import DQNAgent

# # agent = DQNAgent(env)
# # agent.train(n_episodes=2, n_steps=200, render_interval=10)


# from tensortrade.env.default.renderers import PlotlyTradingChart, FileLogger

# chart_renderer = PlotlyTradingChart(
#     display=True,  # show the chart on screen (default)
#     height=800,  # affects both displayed and saved file height. None for 100% height.
#     save_format="html",  # save the chart to an HTML file
#     auto_open_html=False,  # open the saved HTML chart in a new browser tab
# )

# file_logger = FileLogger(
#     filename="example.log",  # omit or None for automatic file name
#     path="training_logs",  # create a new directory if doesn't exist, None for no directory
# )

# renderer_feed = DataFeed(
#     [Stream.source(price_history[c].tolist(), dtype="float").rename(c) for c in price_history]
# )

# env = default.create(
#     portfolio=portfolio,
#     action_scheme="managed-risk",
#     reward_scheme="risk-adjusted",
#     feed=feed,
#     window_size=20,
#     renderer_feed=renderer_feed,
#     renderer=[chart_renderer, file_logger, "screen-log"],
# )

# agent = DQNAgent(env)

# # Set render_interval to None to render at episode ends only
# agent.train(n_episodes=2, n_steps=200, render_interval=10)