import numpy as np
import pandas as pd
from stockstats import wrap, unwrap
from dotenv import load_dotenv
import os
from binance.client import Client

from individual_z_norm_node import IndividualZNormNode
from leaf_data_sets import LeafDataSet

load_dotenv()

binance_api_key = os.environ.get("BINANCE_API_KEY")
binance_api_secret = os.environ.get("BINANCE_API_SECRET")

client = Client(binance_api_key, binance_api_secret)


def target(last_close, future_ohlc, past_ohlc, quantile):
    maxes = []
    minis = []
    for j in range(len(past_ohlc) - len(future_ohlc)):
        base_data = past_ohlc[j:j + len(future_ohlc)]
        d = base_data / base_data[0][0] - 1
        minis.append(np.min(d))
        maxes.append(np.max(d))

    neg_tau = np.quantile(minis, 1 - quantile)
    pos_tau = np.quantile(maxes, quantile)

    changes = future_ohlc / last_close - 1
    up = (changes > pos_tau).any(axis=1)
    down = (changes < neg_tau).any(axis=1)
    cha = np.stack([up, down]).any(axis=0)
    idx = np.where(cha is True)[0] if cha.any() else -1

    if idx == -1:
        return 1
    elif up[idx] and down[idx]:
        if changes[idx][0] > pos_tau:
            return 2
        elif changes[idx][0] < neg_tau:
            return 0
        else:
            return 1
    elif up[idx]:
        return 2
    elif down[idx]:
        return 0
    else:
        return 1


def read_financial_leaf_sliding_window(dataset_name, interval, train_date_from, train_date_to, test_date_from,
                                       test_date_to, sequence_length, quantile, future_range):
    train, target_train = extract_sequence_set_and_targets(dataset_name, interval, train_date_from, train_date_to,
                                                           future_range,
                                                           sequence_length, quantile)
    test, target_test = extract_sequence_set_and_targets(dataset_name, interval, test_date_from, test_date_to,
                                                         future_range,
                                                         sequence_length, quantile)
    leaf = LeafDataSet(train, test, target_train, target_test, dataset_name)
    leaf.interval = interval
    leaf.future_range = future_range
    leaf.sequence_length = sequence_length
    leaf.quantile = quantile
    return leaf, leaf.available_classes_count


def get_ohlc(dataset_name, interval, date_from, date_to):
    return client.get_historical_klines(dataset_name, interval, date_from, date_to)


def extract_sequence_set_and_targets(dataset_name, interval, date_from, date_to, future_range, sequence_length,
                                     quantile):
    ohlcv = pd.DataFrame(get_ohlc(dataset_name, interval, date_from, date_to), dtype=float).iloc[:, 1:6]
    ohlcv.columns = ["open", "high", "low", "close", "volume"]
    ohlcv = wrap(ohlcv)
    features = ['rsi', 'close_7_smma', 'wt2', 'trix', 'wr']
    for f in features:
        ohlcv[f] = ohlcv.get(f)
    ohlcv.dropna(inplace=True)
    ohlcv = unwrap(ohlcv).to_numpy()
    sliding_windows_count = ohlcv.shape[0] - sequence_length - future_range + 1
    feature_count = np.size(ohlcv, axis=1)
    x = np.zeros(shape=(sliding_windows_count, sequence_length, feature_count))
    y = np.zeros(shape=(sliding_windows_count))
    for i in range(sliding_windows_count):
        x[i, :, :] = ohlcv[i:i + sequence_length, :]
        y[i] = target(ohlcv[i + sequence_length - 1, 3],
                      ohlcv[i + sequence_length:i + sequence_length + future_range, :3],
                      ohlcv[i:i + sequence_length, :3],
                      quantile)
    return x, y
