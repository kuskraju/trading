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
    "quantile": 0.2,
    "interval": Client.KLINE_INTERVAL_1HOUR
}

device = "cuda" if torch.cuda.is_available() else "cpu"

epochs = 1

load_dotenv()
web_server = 'https://app.community.clear.ml'
api_server = 'https://api.community.clear.ml'
files_server = 'https://files.community.clear.ml'
access_key = os.environ.get("CLEARML_ACCESS")
secret_key = os.environ.get("CLEARML_SECRET")

binance_api_key = os.environ.get("BINANCE_API_KEY")
binance_api_secret = os.environ.get("BINANCE_API_SECRET")

client = Client(binance_api_key, binance_api_secret)


quantity = 100
