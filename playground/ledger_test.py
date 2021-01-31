import os
import sys
import pandas

sys.path.insert(0, os.path.abspath("../../tensortrade"))

import tensortrade.env.default as default

from tensortrade.feed.core import Stream, DataFeed
from tensortrade.data.cdd import CryptoDataDownload
from tensortrade.oms.wallets import Portfolio, Wallet
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import USD, BTC, ETH, LTC

DOWNLOAD = False

if DOWNLOAD:
    cdd = CryptoDataDownload()
    bitstamp_btc = cdd.fetch("Bitstamp", "USD", "BTC", "1h")
    bitstamp_eth = cdd.fetch("Bitstamp", "USD", "ETH", "1h")
    bitstamp_ltc = cdd.fetch("Bitstamp", "USD", "LTC", "1h")

    bitstamp_btc.to_csv("data/bitstamp_btc_hourly.csv")
    bitstamp_eth.to_csv("data/bitstamp_eth_hourly.csv")
    bitstamp_ltc.to_csv("data/bitstamp_ltc_hourly.csv")

bitstamp_btc = pandas.read_csv("data/bitstamp_btc_hourly.csv")
bitstamp_eth = pandas.read_csv("data/bitstamp_eth_hourly.csv")
bitstamp_ltc = pandas.read_csv("data/bitstamp_ltc_hourly.csv")



# Inspec transactions of Simple orders

bitstamp = Exchange("bitstamp", service=execute_order)(
    Stream.source(list(bitstamp_btc["close"][-100:]), dtype="float").rename("USD-BTC"),
    Stream.source(list(bitstamp_eth["close"][-100:]), dtype="float").rename("USD-ETH"),
    Stream.source(list(bitstamp_ltc["close"][-100:]), dtype="float").rename("USD-LTC"),
)

portfolio = Portfolio(
    USD,
    [
        Wallet(bitstamp, 1000 * USD),
        Wallet(bitstamp, 5 * BTC),
        Wallet(bitstamp, 20 * ETH),
        Wallet(bitstamp, 3 * LTC),
    ],
)

feed = DataFeed(
    [
        Stream.source(list(bitstamp_eth["volume"][-100:]), dtype="float").rename("volume:/USD-ETH"),
        Stream.source(list(bitstamp_ltc["volume"][-100:]), dtype="float").rename("volume:/USD-LTC"),
    ]
)

env = default.create(
    portfolio=portfolio,
    action_scheme=default.actions.SimpleOrders(),
    reward_scheme=default.rewards.SimpleProfit(),
    feed=feed,
)

done = False
obs = env.reset()
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print("action: {}".format(action))

head = portfolio.ledger.as_frame().head(7)
print(head)

# ManagedRiskOrders

portfolio = Portfolio(USD, [
    Wallet(bitstamp, 10000 * USD),
    Wallet(bitstamp, 0 * BTC),
    Wallet(bitstamp, 0 * ETH),
])

env = default.create(
    portfolio=portfolio,
    action_scheme=default.actions.ManagedRiskOrders(),
    reward_scheme=default.rewards.SimpleProfit(),
    feed=feed
)

done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

head = portfolio.ledger.as_frame().head(20)
print(head)