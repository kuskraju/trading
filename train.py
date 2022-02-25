import torch
from datetime import datetime
import numpy as np
from clearml import Task
from dotenv import load_dotenv
from binance.client import Client
import os

from conformer_ordinal_trainer import ConformerOrdinalTrainer
from statistics import get_triple_barrier_statistics
from read_financial_leaf_sliding_window import read_financial_leaf_sliding_window

load_dotenv()

web_server = 'https://app.community.clear.ml'
api_server = 'https://api.community.clear.ml'
files_server = 'https://files.community.clear.ml'
access_key = os.environ.get("CLEARML_ACCESS")
secret_key = os.environ.get("CLEARML_SECRET")
project_name = "X_project"
Task.set_credentials(web_host=web_server,
                     api_host=api_server,
                     files_host=files_server,
                     key=access_key,
                     secret=secret_key)

trainer_config = {
    "optimizer_step_at_each_batch": False,
    "initial_learning_rate": 0.000004,
    "batch_size": 64,
}

model_config = {
    "d_model": 64,
    "n": 6,
    "heads": 8,
    "dropout": 0.1,
    "kernel_size": 16
}

sequence_length = 13
quantile = 0.2
future_range = 5

epochs = 180

dataset_names = ["ETHBTC"]

interval = Client.KLINE_INTERVAL_1DAY

cont = False
done_num = 0

device = "cuda" if torch.cuda.is_available() else "cpu"

steps = 0
if cont:
    task_sum = Task.get_task(project_name=project_name, task_name="Summary")
    good_count = task_sum.artifacts["good_count"].get()
    all_played = task_sum.artifacts["all_played"].get()
    task_sum.close()
else:
    task = Task.init(project_name=project_name,
                     task_name="Summary",
                     reuse_last_task_id=False)
    task.close()
    good_count = np.zeros((len(dataset_names)))
    all_played = np.zeros((len(dataset_names)))

for yit, year in enumerate(range(2018, 2020, 1)):
    train_date_from = datetime(year, 1, 1).strftime("%d %b, %Y")
    train_date_to = datetime(year + 1, 12, 30).strftime("%d %b, %Y")
    test_date_from = datetime(year + 2, 1, 1).strftime("%d %b, %Y")
    test_date_to = datetime(year + 2, 12, 30).strftime("%d %b, %Y")
    clear_dict = {
        "train_date_from": train_date_from,
        "train_date_to": train_date_to,
        "test_date_from": test_date_from,
        "test_date_to": test_date_to
    }

    for di, dataset_name in enumerate(dataset_names):

        node, class_count = read_financial_leaf_sliding_window(
            dataset_name=dataset_name,
            interval=interval,
            train_date_from=train_date_from,
            train_date_to=train_date_to,
            test_date_from=test_date_from,
            test_date_to=test_date_to,
            sequence_length=sequence_length,
            quantile=quantile,
            future_range=future_range
        )

        clear_dict.update(node.get_config())

        steps += 1
        if cont and steps <= done_num:
            continue

        trainer = ConformerOrdinalTrainer(node, trainer_config, model_config, class_count,
                                          device, "ConformerOrdinal", get_triple_barrier_statistics())

        task = Task.init(project_name=project_name,
                         task_name=node.node_name,
                         reuse_last_task_id=False)

        clear_dict.update(trainer.model_config)
        clear_dict.update(trainer.trainer_config)

        trainer.logger = task.get_logger()
        trainer.plot_hist_classes()
        task.connect(clear_dict)

        trainer.train_model(epochs)
        good_count[di] += trainer.register["test_good_count"][-1]
        all_played[di] += trainer.register["test_all_played"][-1]

        task.close()

        task_sum = Task.get_task(project_name=project_name,
                                 task_name="Summary")

        task_sum.upload_artifact(name='good_count', artifact_object=good_count)
        task_sum.upload_artifact(name='all_played', artifact_object=all_played)

        logger_sum = task_sum.get_logger()
        val = trainer.register["test_good_count"][-1] / trainer.register["test_all_played"][-1] if \
            trainer.register["test_all_played"][-1] > 0 else 0
        logger_sum.report_scalar(title="Play_Accuracy_{}".format(dataset_name),
                                 series="play_acc", iteration=yit,
                                 value=val)

        task_sum.close()

task_sum = Task.get_task(project_name=project_name,
                         task_name="Summary")
logger_sum = task_sum.get_logger()

logger_sum.report_histogram(
    "Play accuracy",
    "play acc",
    values=good_count / all_played if all_played > 0 else 0,
    labels=dataset_names
)
logger_sum.report_histogram(
    "Population size",
    "played",
    values=all_played,
    labels=dataset_names
)

task_sum.close()
