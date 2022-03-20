from datetime import datetime
from clearml import Task
from dateutil.relativedelta import relativedelta

from hidden_module.conformer_ordinal_trainer import ConformerOrdinalTrainer
from hidden_module.constants import data_config, web_server, api_server, files_server, access_key, secret_key, \
    trainer_config, \
    model_config, device, epochs, project_name
from hidden_module.statistics import get_triple_barrier_statistics
from hidden_module.read_financial_leaf_sliding_window import read_financial_leaf_sliding_window

Task.set_credentials(web_host=web_server,
                     api_host=api_server,
                     files_host=files_server,
                     key=access_key,
                     secret=secret_key)


def train():
    task_sum = Task.init(project_name=project_name,
                         task_name="Summary",
                         tags=["test_training"],
                         reuse_last_task_id=False)
    task_sum.close()

    for mit in range(1, 10, 1):
        time_config = {
            "train_date_from": datetime(2021, mit, 1).strftime("%d %b, %Y"),
            "train_date_to": datetime(2021, mit + 2, 30).strftime("%d %b, %Y"),
            "test_date_from": datetime(2021, mit + 3, 1).strftime("%d %b, %Y"),
            "test_date_to": datetime(2021, mit + 3, 30).strftime("%d %b, %Y")
        }

        node, class_count, high_train, low_train, high_test, low_test = read_financial_leaf_sliding_window(
            data_config=data_config,
            time_config=time_config
        )

        clear_dict = {**data_config, **time_config, **trainer_config, **model_config}
        trainer = ConformerOrdinalTrainer(node, trainer_config, model_config, class_count, device,
                                          get_triple_barrier_statistics(), high_train, low_train, high_test, low_test)

        task = Task.init(project_name=project_name,
                         task_name=node.node_name,
                         tags=["test_training"],
                         reuse_last_task_id=False)

        trainer.logger = task.get_logger()
        trainer.plot_hist_classes()
        task.connect(clear_dict)

        trainer.train_model(epochs)

        task.close()

        task_sum = Task.get_task(project_name=project_name,
                                 task_name="Summary")

        logger_sum = task_sum.get_logger()
        val = trainer.register["test_gain"][-1]
        logger_sum.report_scalar(title="Gain", series="gain", iteration=mit, value=val)
        task_sum.close()


def train_latest():
    date = datetime.now()
    time_config = {
        "train_date_from": (date - relativedelta(day=1)).strftime("%d %b, %Y"),
        "train_date_to": date.strftime("%d %b, %Y")
    }

    node, class_count, _, _, _, _ = read_financial_leaf_sliding_window(time_config=time_config,
                                                                    data_config=data_config)

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
