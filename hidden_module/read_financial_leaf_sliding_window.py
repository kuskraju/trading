import numpy as np
import pandas as pd
from stockstats import wrap, unwrap

from hidden_module.constants import client
from hidden_module.individual_z_norm_node import IndividualZNormNode
from hidden_module.leaf_data_sets import LeafDataSet


def read_financial_leaf_sliding_window(data_config, time_config=None):
    high, low = None, None
    train, target_train, test, target_test = None, None, None, None
    if time_config is None:
        test, _, high, low = extract_sequence_set_and_targets(data_config["dataset_name"],
                                                              data_config["interval"],
                                                              data_config["future_range"],
                                                              data_config["sequence_length"],
                                                              data_config["quantile"])
    else:
        train, target_train, _, _ = extract_sequence_set_and_targets(data_config["dataset_name"],
                                                                     data_config["interval"],
                                                                     data_config["future_range"],
                                                                     data_config["sequence_length"],
                                                                     data_config["quantile"],
                                                                     date_from=time_config["train_date_from"],
                                                                     date_to=time_config["train_date_to"],
                                                                     )
        if "test_date_from" in time_config.keys():
            test, target_test, _, _ = extract_sequence_set_and_targets(data_config["dataset_name"],
                                                                       data_config["interval"],
                                                                       data_config["future_range"],
                                                                       data_config["sequence_length"],
                                                                       data_config["quantile"],
                                                                       date_from=time_config["test_date_from"],
                                                                       date_to=time_config["test_date_to"],
                                                                       )
    leaf = LeafDataSet(train, test, target_train, target_test, data_config["dataset_name"])
    leaf = IndividualZNormNode(leaf)
    return leaf, leaf.available_classes_count, high, low


def extract_sequence_set_and_targets(dataset_name, interval, future_range, sequence_length, quantile,
                                     date_from=None, date_to=None):
    api_data = get_ohlc(dataset_name, interval, date_from=date_from, date_to=date_to, sequence_length=sequence_length)
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
        print(sliding_windows_count, sequence_length, feature_count)
        x = np.zeros(shape=(sliding_windows_count, sequence_length, feature_count))
        y = np.zeros(shape=sliding_windows_count)
        for i in range(sliding_windows_count):
            x[i, :, :] = ohlcv[i:i + sequence_length, :]
            y[i], _, _ = target(ohlcv[i + sequence_length - 1, 3],
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
        if date_from is not None else client.futures_historical_klines(dataset_name, interval, "1 hour ago UTC",
                                                                       limit=sequence_length)


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
    idx = np.where(cha == True)[0][0] if cha.any() else -1

    if idx == -1:
        return 1, last_close * (1 + pos_tau), last_close * (1 + neg_tau)
    elif up[idx] and down[idx]:
        if changes[idx][0] > pos_tau:
            return 2, last_close * (1 + pos_tau), last_close * (1 + neg_tau)
        elif changes[idx][0] < neg_tau:
            return 0, last_close * (1 + pos_tau), last_close * (1 + neg_tau)
        else:
            return 1, last_close * (1 + pos_tau), last_close * (1 + neg_tau)
    elif up[idx]:
        return 2, last_close * (1 + pos_tau), last_close * (1 + neg_tau)
    elif down[idx]:
        return 0, last_close * (1 + pos_tau), last_close * (1 + neg_tau)
    else:
        return 1, last_close * (1 + pos_tau), last_close * (1 + neg_tau)
