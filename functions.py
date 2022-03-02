import time

import torch
import numpy
from binance.exceptions import BinanceAPIException, BinanceOrderException
from datetime import datetime

from clearml import Task
from dateutil.relativedelta import *


from hidden_module.Conformer.ConformerOrdinal import ConformerOrdinal
from hidden_module.conformer_ordinal_trainer import ConformerOrdinalTrainer
from hidden_module.constants import model_config, data_config, trainer_config, epochs, client, device, test_client, \
    pricePrecision, money_to_play, quantityPrecision, web_server, api_server, files_server, access_key, secret_key, \
    project_name
from hidden_module.read_financial_leaf_sliding_window import read_financial_leaf_sliding_window
from hidden_module.statistics import get_triple_barrier_statistics


Task.set_credentials(web_host=web_server,
                     api_host=api_server,
                     files_host=files_server,
                     key=access_key,
                     secret=secret_key)


def train_latest():
    date = datetime.now()
    time_config = {
            "train_date_from": (date - relativedelta(day=1)).strftime("%d %b, %Y"),
            "train_date_to": date.strftime("%d %b, %Y")
        }

    node, class_count, _, _ = read_financial_leaf_sliding_window(time_config=time_config, data_config=data_config)

    trainer = ConformerOrdinalTrainer(node, trainer_config, model_config, class_count, device,
                                      get_triple_barrier_statistics())

    task = Task.init(project_name=project_name,
                     task_name=node.node_name,
                     tags=["latest"],
                     reuse_last_task_id=False)

    trainer.logger = task.get_logger()
    trainer.plot_hist_classes()
    clear_dict = {**data_config, **trainer_config, **model_config}

    task.connect(clear_dict)

    trainer.train_model(epochs)

    task.upload_artifact("model", trainer.model.state_dict())
    task.close()


def trade():
    task = Task.get_task(task_id="9b51fa51344d4d009738220eb8bd4c6b")
    node, class_count, high, low = read_financial_leaf_sliding_window(data_config=data_config)
    model = ConformerOrdinal(
        device,
        model_config["d_model"],
        model_config["n"],
        model_config["heads"],
        model_config["dropout"],
        node.features_count(),
        node.sequence_length(),
        class_count
    )
    model.load_state_dict(task.artifacts["model"].get())
    prev = None
    while True:
        node, class_count, high, low = read_financial_leaf_sliding_window(data_config=data_config)
        if prev != node.test[0, 0, 0]:
            predict = model(torch.tensor(node.test.astype(float)).type(torch.FloatTensor).to(device)).detach().cpu().numpy()
            play(numpy.argmax(predict), high, low)
        prev = node.test[0, 0, 0]
        time.sleep(10)


def play(action, high, low):
    client = test_client
    curr_price = float(client.get_avg_price(symbol=data_config["dataset_name"])["price"])
    quantity = "{:0.0{}f}".format(
        (money_to_play / curr_price),
        quantityPrecision
    )
    curr_future_price = client.futures_mark_price(symbol=data_config["dataset_name"])
    print("Curr price: {}".format(curr_future_price["markPrice"]))
    position_side = "SHORT" if action == 0 else ("LONG" if action == 2 else None)
    low = "{:0.0{}f}".format(low, pricePrecision)
    high = "{:0.0{}f}".format(high, pricePrecision)
    loss_price = low if position_side == "LONG" else high
    profit_price = high if position_side == "LONG" else low
    print("Side: {} Low: {} High: {}".format(position_side, loss_price, profit_price))
    if position_side is not None:
        try:
            main_order = client.futures_create_order(
                symbol=data_config["dataset_name"], side="BUY", type="MARKET", quantity=quantity,
                positionSide=position_side)
            print(main_order["orderId"])
            sm_order = client.futures_create_order(
                symbol=data_config["dataset_name"], side="SELL", type="STOP_MARKET", quantity=quantity,
                positionSide=position_side, stopPrice=loss_price, closePosition=True)
            print(sm_order["orderId"])
            tpm_order = client.futures_create_order(
                symbol=data_config["dataset_name"], side="SELL", type="TAKE_PROFIT_MARKET", quantity=quantity,
                positionSide=position_side, stopPrice=profit_price, closePosition=True)
            print(tpm_order["orderId"])
        except (BinanceAPIException, BinanceOrderException) as e:
            print(e)

