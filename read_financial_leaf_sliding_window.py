from datetime import datetime
import numpy as np
import pandas as pd
from stockstats import wrap, unwrap
from data_downloader.data_downloader import DataDownloader
from execution.preprocessing.tree_structure.leaves.leaf_data_sets import LeafDataSets
from dotenv import load_dotenv
import os
from binance.client import Client
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

    neg_tau = np.quantile(minis, 1-quantile)
    pos_tau = np.quantile(maxes, quantile)

    changes = future_ohlc / last_close - 1
    up = (changes > pos_tau).any(axis=1)
    down = (changes < neg_tau).any(axis=1)
    cha = np.stack([up, down]).any(axis=0)
    idx = np.where(cha is True)[0][0] if cha.any() else -1

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
                                       test_date_to, sequence_length, quantile):

    train = extract_sequence_set_and_targets(dataset_name)
    test = extract_sequence_set_and_targets(dataset_name)

    leaf.node_name = dataset_name
    leaf.interval = interval
    leaf.train_ohlc = train_ohlc
    leaf.test_ohlc = test_ohl
    leaf.future_range = future_range
    leaf.sequence_length = sequence_length
    leaf.quantile = quantile
    IndividualZNormNode(leaf)
    return leaf, leaf.available_classes_count


def get_ohlc(dataset_name, interval, date_from, date_to):
    return client.get_historical_klines(dataset_name, interval, date_from, date_to)


def extract_sequence_set_and_targets(data_downloader, date_from, date_to, future_range, sequence_length, instruments_ohlc):
    ohlc = wrap(get_ohlc(dataset_name, interval, date_from, date_to))()
    features = ['rsi', 'close_7_smma', 'wt2', 'trix', 'wr']
    for f in features:
        ohlc[f] = ohlc.get(f)
    ohlc.dropna(inplace=True)
    ohlc = unwrap(ohlc).to_numpy()
    sliding_windows_count = np.size(ohlc, axis=0) - sequence_length - future_range + 1
    feature_count = np.size(ohlc, axis=1)
    sequence_set = np.zeros(shape=((sliding_windows_count, sequence_length, feature_count)))
    targets = np.zeros(shape=((sliding_windows_count)))
    for i in range(sliding_windows_count):
        sequence_set[i, :, :] = ohlc[i:i + sequence_length, :]
        targets[i] = target(ohlc[i + sequence_length - 1, 3],
                                ohlc[i + sequence_length:i + sequence_length + future_range, :3],
                                ohlc[i:i + sequence_length, :3],
                                quantile)
    return sequence_set, targets, ohlc
