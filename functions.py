import time

import torch
import numpy
from binance.enums import ContractType
from binance.exceptions import BinanceAPIException, BinanceOrderException
from binance import BinanceSocketManager, AsyncClient
from clearml import Task

from hidden_module.Conformer.ConformerOrdinal import ConformerOrdinal
from hidden_module.constants import model_config, data_config, client, device,\
    pricePrecision, money_to_play, quantityPrecision, web_server, api_server, files_server, access_key, secret_key, \
    test_binance_api_key, test_binance_api_secret
from hidden_module.read_financial_leaf_sliding_window import read_financial_leaf_sliding_window


Task.set_credentials(web_host=web_server,
                     api_host=api_server,
                     files_host=files_server,
                     key=access_key,
                     secret=secret_key)


async def trade():
    orderIds = []
    # async_client = AsyncClient(binance_api_key, binance_api_secret)
    async_client = AsyncClient(test_binance_api_key, test_binance_api_secret, testnet=True)
    node, class_count, _, _= read_financial_leaf_sliding_window(data_config=data_config)
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
    task = Task.get_task(task_id="7765c33a66ce430c85093658fef3533f")
    model.load_state_dict(task.artifacts["model"].get())

    while True:
        open_orders = [x["orderId"] for x in client.futures_get_open_orders(symbol=data_config["dataset_name"])]
        i = 0
        while i < len(orderIds):
            for j in range(2):
                if orderIds[i][j] not in open_orders and orderIds[i][1-j] in open_orders:
                    client.future_cancel_order(
                        symbol=data_config["dataset_name"],
                        orderId=str(orderIds[i][1-j])
                    )
            if orderIds[i][0] not in open_orders and orderIds[i][1] not in open_orders:
                orderIds = orderIds.pop(i)
            else:
                i += 1
        node, class_count, high, low = read_financial_leaf_sliding_window(data_config=data_config)
        predict = model(
            torch.tensor(node.test.astype(float)).type(torch.FloatTensor).to(device)
        ).detach().cpu().numpy()
        ids = play(numpy.argmax(predict), high, low)
        if ids is not None:
            orderIds.append(ids)
        time.sleep(5 * 60)


def play(action, high, low):
    curr_price = float(client.get_avg_price(symbol=data_config["dataset_name"])["price"])
    quantity = "{:0.0{}f}".format(
        (money_to_play / curr_price),
        quantityPrecision
    )
    position_side = "SHORT" if action == 0 else ("LONG" if action == 2 else None)
    low = "{:0.0{}f}".format(low, pricePrecision)
    high = "{:0.0{}f}".format(high, pricePrecision)
    loss_price = low if position_side == "LONG" else high
    profit_price = high if position_side == "LONG" else low
    print("Side: {} Low: {} High: {}".format(position_side, low, high))
    if position_side is not None:
        try:
            client.futures_create_order(
                symbol=data_config["dataset_name"], side="BUY", type="MARKET", quantity=quantity,
                positionSide=position_side)
            time.sleep(1)
            sm_order = client.futures_create_order(
                symbol=data_config["dataset_name"], side="SELL", type="STOP_MARKET", quantity=quantity,
                positionSide=position_side, stopPrice=loss_price)
            tpm_order = client.futures_create_order(
                symbol=data_config["dataset_name"], side="SELL", type="TAKE_PROFIT_MARKET", quantity=quantity,
                positionSide=position_side, stopPrice=profit_price)
            return [sm_order["orderId"], tpm_order["orderId"]]
        except (BinanceAPIException, BinanceOrderException) as e:
            print(e)
    return None
