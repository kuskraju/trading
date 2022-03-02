import torch
from dotenv import load_dotenv
from binance.client import Client
import os

load_dotenv()

web_server = 'https://app.community.clear.ml'
api_server = 'https://api.community.clear.ml'
files_server = 'https://files.community.clear.ml'
access_key = os.environ.get("CLEARML_ACCESS")
secret_key = os.environ.get("CLEARML_SECRET")

project_name = "X_project"

trainer_config = {
    "optimizer_step_at_each_batch": False,
    "initial_learning_rate": 0.0001,
    "batch_size": 64,
}

model_config = {
    "name": "ConformerOrdinal",
    "d_model": 64,
    "n": 6,
    "heads": 8,
    "dropout": 0.1,
    "kernel_size": 16
}

data_config = {
    "dataset_name": "BTCUSDT",
    "sequence_length": 13,
    "future_range": 5,
    "quantile": 0.6,
    "interval": Client.KLINE_INTERVAL_1MINUTE
}

device = "cuda" if torch.cuda.is_available() else "cpu"

load_dotenv()
web_server = 'https://app.community.clear.ml'
api_server = 'https://api.community.clear.ml'
files_server = 'https://files.community.clear.ml'
access_key = os.environ.get("CLEARML_ACCESS")
secret_key = os.environ.get("CLEARML_SECRET")

binance_api_key = os.environ.get("BINANCE_API_KEY")
binance_api_secret = os.environ.get("BINANCE_API_SECRET")
test_binance_api_key = os.environ.get("TEST_BINANCE_API_KEY")
test_binance_api_secret = os.environ.get("TEST_BINANCE_API_SECRET")

client = Client(binance_api_key, binance_api_secret)
test_client = Client(test_binance_api_key, test_binance_api_secret, testnet=True)

money_to_play = 100

epochs = 130

info = client.futures_exchange_info()

pricePrecision = 1
for x in info['symbols']:
    if x['symbol'] == data_config["dataset_name"]:
        pricePrecision = x['pricePrecision']
        break

quantityPrecision = 1
for x in info['symbols']:
    if x['symbol'] == data_config["dataset_name"]:
        quantityPrecision = x['quantityPrecision']
        break

'''
exchange_info = client.get_exchange_info()
for s in exchange_info['symbols']:
    print(s['symbol'])
'''
