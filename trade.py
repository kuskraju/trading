import torch
import numpy
from clearml import Task
from datetime import datetime
from dateutil.relativedelta import *


from Conformer.ConformerOrdinal import ConformerOrdinal
from conformer_ordinal_trainer import ConformerOrdinalTrainer
from constants import model_config, data_config, api_server, web_server, files_server, access_key, secret_key, \
    trainer_config, epochs
from read_financial_leaf_sliding_window import read_financial_leaf_sliding_window
from statistics import get_triple_barrier_statistics


def trade():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Task.set_credentials(web_host=web_server,
                         api_host=api_server,
                         files_host=files_server,
                         key=access_key,
                         secret=secret_key)
    # Train
    date = datetime.now()
    time_config = {
            "train_date_from": (date - relativedelta(months=2)).strftime("%d %b, %Y"),
            "train_date_to": date.strftime("%d %b, %Y")
        }

    node, class_count, _, _ = read_financial_leaf_sliding_window(time_config=time_config, data_config=data_config)

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

    trainer = ConformerOrdinalTrainer(node, trainer_config, model_config, class_count, device,
                                      get_triple_barrier_statistics())

    task = Task.init(project_name="trade_{}".format(data_config["dataset_name"]),
                     task_name=node.node_name,
                     reuse_last_task_id=False)

    trainer.logger = task.get_logger()
    trainer.plot_hist_classes()
    clear_dict = {**data_config, **trainer_config, **model_config}

    task.connect(clear_dict)

    trainer.train_model(epochs)
    task.close()

    node, class_count, high, low = read_financial_leaf_sliding_window(data_config=data_config)

    predict = model(torch.tensor(node.test.astype(float)).type(torch.FloatTensor).to(device)).detach().cpu().numpy()
    print(node.test[0, 0, 0], numpy.argmax(predict), high, low)
    """play(predict, high, low)"""
