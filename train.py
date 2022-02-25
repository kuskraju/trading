from datetime import datetime
import numpy as np
from clearml import Task
from dotenv import load_dotenv
from binance.client import Client
import os

from individual_z_norm_node import IndividualZNormNode
from cofnormer_ordinal_trainer import ConformerOrdinalTrainer
from statistics import get_triple_barrier_statistics
from read_financial_leaf_sliding_window import read_financial_leaf_sliding_window

load_dotenv()

web_server = 'https://app.community.clear.ml'
api_server = 'https://api.community.clear.ml'
files_server = 'https://files.community.clear.ml'
access_key = os.environ.get("CLEARML_ACCESS")
secret_key = os.environ.get("CLEARML_SECRET")
project_name = "X"
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

epochs = 300

dataset_names = ["ETHBTC"]

interval = Client.KLINE_INTERVAL_1HOUR

cont = False
done_num = 0


def financial_experiment(device):
    steps = 0
    if cont:
        task_sum = Task.get_task(project_name=project_name, task_name="Summary_{}".format(net))
        good_count = task_sum.artifacts["good_count"].get()
        all_played = task_sum.artifacts["all_played"].get()
        task_sum.close()
    else:
        task = Task.init(project_name=project_name,
                         task_name="Summary_{}".format(net),
                         reuse_last_task_id=False)
        task.close()
        good_count = np.zeros((len(dataset_names)))
        all_played = np.zeros((len(dataset_names)))

    for yit, year in enumerate(range(2012, 2018, 2)):
        train_date_from = datetime(year, 1, 1)
        train_date_to = datetime(year + 2, 12, 30)
        test_date_from = datetime(year + 3, 1, 1)
        test_date_to = datetime(year + 3, 12, 30)
        clear_dict = {
            "train_date_from": train_date_from.strftime("%m/%d/%Y"),
            "train_date_to": train_date_to.strftime("%m/%d/%Y"),
            "test_date_from": test_date_from.strftime("%m/%d/%Y"),
            "test_date_to": test_date_to.strftime("%m/%d/%Y")
        }

        for di, dataset_name in enumerate(dataset_names):

            node, class_count = read_financial_leaf_sliding_window(
                dataset_name=dataset_name,
                train_date_from=train_date_from,
                train_date_to=train_date_to,
                test_date_from=test_date_from,
                test_date_to=test_date_to,
                sequence_length=sequence_length,
                quantile=quantile
            )

            clear_dict.update(node.get_confih())

            steps += 1
            if cont and steps <= done_num:
                continue

            trainer = ConformerOrdinalTrainer(node_shuffled, trainer_config, class_count,
                                              device, get_triple_barrier_statistics(), model_config)

            task = Task.init(project_name=project_name,
                             task_name="{}_{}".format(node.node_name, net),
                             reuse_last_task_id=False)

            clear_dict.update(trainer.model_config)
            clear_dict.update(trainer.trainer_config)

            trainer.logger = task.get_logger()
            trainer.plot_hist_classes()
            task.connect(clear_dict)

            trainer.train_model(epochs, net)
            good_count[di] += trainer.first_run_register["validation_good_count"][-1]
            all_played[di] += trainer.first_run_register["validation_all_played"][-1]

            task.close()

            task_sum = Task.get_task(project_name=project_name,
                                     task_name="Summary_{}".format(net))

            task_sum.upload_artifact(name='good_count', artifact_object=good_count)
            task_sum.upload_artifact(name='all_played', artifact_object=all_played)

            logger_sum = task_sum.get_logger()
            logger_sum.report_scalar(title="Play_Accuracy_{}".format(dataset_name),
                                     series="{}_{}".format(param["quantile"], learning_window_size), iteration=yit,
                                     value=trainer.first_run_register["validation_good_count"][-1] /
                                           trainer.first_run_register["validation_all_played"][-1])

            task_sum.close()

    task_sum = Task.get_task(project_name=project_name,
                             task_name="Summary_{}".format(net))
    logger_sum = task_sum.get_logger()

    logger_sum.report_histogram(
        "Play accuracy",
        "played",
        values=good_count / all_played,
        labels=dataset_names
    )
    logger_sum.report_histogram(
        "Population size",
        "played",
        values=all_played,
        labels=dataset_names
    )

    task_sum.close()
