import datetime
from datetime import timedelta

import numpy as np
import pandas as pd
from binance import Client
from stockstats import wrap, unwrap

from hidden_module.constants import client
from hidden_module.individual_z_norm_node import IndividualZNormNode
from hidden_module.leaf_data_sets import LeafDataSet


def read_financial_leaf_sliding_window(data_config, time_config=None):
    high_train, low_train, high_test, low_test = None, None, None, None
    train, target_train, test, target_test = None, None, None, None
    if time_config is None:
        test, _, high_test, low_test = extract_sequence_set_and_targets(data_config["dataset_name"],
                                                                        data_config["interval"],
                                                                        data_config["future_range"],
                                                                        data_config["sequence_length"],
                                                                        data_config["quantile"])
    else:
        train, target_train, high_train, low_train = extract_sequence_set_and_targets(data_config["dataset_name"],
                                                                                      data_config["interval"],
                                                                                      data_config["future_range"],
                                                                                      data_config["sequence_length"],
                                                                                      data_config["quantile"],
                                                                                      date_from=time_config[
                                                                                          "train_date_from"],
                                                                                      date_to=time_config[
                                                                                          "train_date_to"],
                                                                                      )
        if "test_date_from" in time_config.keys():
            test, target_test, high_test, low_test = extract_sequence_set_and_targets(data_config["dataset_name"],
                                                                                      data_config["interval"],
                                                                                      data_config["future_range"],
                                                                                      data_config["sequence_length"],
                                                                                      data_config["quantile"],
                                                                                      date_from=time_config[
                                                                                          "test_date_from"],
                                                                                      date_to=time_config[
                                                                                          "test_date_to"],
                                                                                      )
    leaf = LeafDataSet(train, test, target_train, target_test, data_config["dataset_name"])
    leaf = IndividualZNormNode(leaf)
    return leaf, leaf.available_classes_count, high_train, low_train, high_test, low_test


def extract_sequence_set_and_targets(instrument_name, interval, future_range, sequence_length, quantile,
                                     date_from=None, date_to=None):
    if "VOL" in instrument_name:
        api_data = get_volume_based_bars(instrument_name, interval, date_from=date_from, date_to=date_to,
                                         sequence_length=sequence_length)
    else:
        api_data = get_ohlc(instrument_name, interval, date_from=date_from, date_to=date_to,
                            sequence_length=sequence_length)
    ohlcv = pd.DataFrame(api_data, dtype=float).iloc[:, 1:6]
    ohlcv.columns = ["open", "high", "low", "close", "volume"]
    ohlcv = wrap(ohlcv)
    features = ['close_7_smma', 'wt2', 'trix', 'wr']
    for f in features:
        ohlcv[f] = ohlcv.get(f)
    ohlcv.dropna(inplace=True)
    ohlcv = unwrap(ohlcv).to_numpy()
    feature_count = np.size(ohlcv, axis=1)
    x, y, high, low = None, None, None, None
    if date_from is not None:
        sliding_windows_count = ohlcv.shape[0] - sequence_length - future_range + 1
        x = np.zeros(shape=(sliding_windows_count, sequence_length, feature_count))
        y = np.zeros(shape=sliding_windows_count)
        high = np.zeros(shape=sliding_windows_count)
        low = np.zeros(shape=sliding_windows_count)
        for i in range(sliding_windows_count):
            x[i, :, :] = ohlcv[i:i + sequence_length, :]
            y[i], high[i], low[i] = target(ohlcv[i + sequence_length - 1, 3],
                                           ohlcv[i + sequence_length:i + sequence_length + future_range, :3],
                                           ohlcv[i:i + sequence_length, :3],
                                           quantile)
    else:
        x = np.zeros(shape=(1, sequence_length, feature_count))
        x[0, :, :] = ohlcv[:sequence_length, :]
        _, high, low = target(ohlcv[0, 3], np.zeros((future_range, 1)), ohlcv[:sequence_length, :3], quantile)
    return x, y, high, low


def get_ohlc(dataset_name, interval, date_from=None, date_to=None, sequence_length=None):
    return client.futures_historical_klines(dataset_name, interval, date_from, date_to) \
        if date_from is not None else client.futures_historical_klines(dataset_name,
                                                                       interval,
                                                                       "3 day ago UTC")[-1 - sequence_length:]


def target(last_close, future_ohlc, past_ohlc, quantile):
    ems = []
    for j in range(len(past_ohlc) - len(future_ohlc)):
        base_data = past_ohlc[j:j + len(future_ohlc)]
        d = base_data / base_data[0][0] - 1
        ems.append(np.max(np.abs(d)))

    tau = np.quantile(ems, quantile)

    changes = future_ohlc / last_close - 1
    up = (changes > tau).any(axis=1)
    down = (changes < -tau).any(axis=1)
    cha = np.stack([up, down]).any(axis=0)
    idx = np.where(cha == True)[0][0] if cha.any() else -1

    if idx == -1:
        return 1, tau, -tau
    elif up[idx] and down[idx]:
        if changes[idx][0] > tau:
            return 2, tau, -tau
        elif changes[idx][0] < tau:
            return 0, tau, -tau
        else:
            return 1, tau, -tau
    elif up[idx]:
        return 2, tau, -tau
    elif down[idx]:
        return 0, tau, -tau
    else:
        return 1, tau, -tau


def get_volume_based_bars(instrument_name, interval, date_from=None, date_to=None, sequence_length=None):
    take_only_last = False
    delta = volume_delta_dict[interval]
    if date_from is None:
        take_only_last = True
        date_to = datetime.datetime.now()
        date_from = date_to - delta
    df = client.futures_historical_klines(instrument_name, volume_interval_dict[interval], date_from,
                                          date_from + delta)
    date_from += delta
    ohlcv = pd.DataFrame(data=[df.iloc[0]], columns=["open", "high", "low", "close", "volume"])
    df = df.drop(df.index[[0]], axis=0)
    AVG_VOLUME = volume_avg_dict[interval]
    while date_from < date_to:
        while len(df) > 0:
            if ohlcv.iloc[-1, 4] > AVG_VOLUME:
                ohlcv = ohlcv.append(pd.DataFrame(data=[df.iloc[0]],
                                                  columns=["open", "high", "low", "close", "volume"]))
                df = df.drop(df.index[[0]], axis=0)
            else:
                ohlcv.iloc[-1, 3] = df['close'][0]
                ohlcv.iloc[-1, 2] = min(df['low'][0], ohlcv.iloc[-1, 2])
                ohlcv.iloc[-1, 1] = max(df['high'][0], ohlcv.iloc[-1, 1])
                ohlcv.iloc[-1, 4] += df['volume'][0]
                df = df.drop(df.index[[0]], axis=0)
        df = df.append(client.futures_historical_klines(instrument_name, volume_interval_dict[interval], date_from,
                                                        date_from + delta))
        date_from += delta
    return ohlcv if not take_only_last else ohlcv[-1 - sequence_length:]


volume_interval_dict = {
    Client.KLINE_INTERVAL_5MINUTE: Client.KLINE_INTERVAL_1MINUTE,
    Client.KLINE_INTERVAL_30MINUTE: Client.KLINE_INTERVAL_1MINUTE,
    Client.KLINE_INTERVAL_1HOUR: Client.KLINE_INTERVAL_5MINUTE,
    Client.KLINE_INTERVAL_4HOUR: Client.KLINE_INTERVAL_30MINUTE,
    Client.KLINE_INTERVAL_1DAY: Client.KLINE_INTERVAL_1HOUR,
}

volume_avg_dict = {
    Client.KLINE_INTERVAL_5MINUTE: 100,
    Client.KLINE_INTERVAL_30MINUTE: 200,
    Client.KLINE_INTERVAL_1HOUR: 500,
    Client.KLINE_INTERVAL_4HOUR: 1000,
    Client.KLINE_INTERVAL_1DAY: 80000,
}

volume_delta_dict = {
    Client.KLINE_INTERVAL_5MINUTE: timedelta(hours=20),
    Client.KLINE_INTERVAL_30MINUTE: timedelta(days=5),
    Client.KLINE_INTERVAL_1HOUR: timedelta(days=10),
    Client.KLINE_INTERVAL_4HOUR: timedelta(days=30),
    Client.KLINE_INTERVAL_1DAY: timedelta(weeks=6),
}
