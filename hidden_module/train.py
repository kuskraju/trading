from datetime import datetime
from clearml import Task

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
    good_count = 0
    all_played = 0

    task_sum = Task.init(project_name=project_name,
                         task_name="Summary",
                         tags=["test_training"],
                         reuse_last_task_id=False)
    task_sum.close()

    for mit in range(1, 2, 20):
        time_config = {
            "train_date_from": datetime(2022, 1, mit).strftime("%d %b, %Y"),
            "train_date_to": datetime(2022, 1, mit + 1).strftime("%d %b, %Y"),
            "test_date_from": datetime(2022, 1, mit + 2).strftime("%d %b, %Y"),
            "test_date_to": datetime(2022, 1, mit + 3).strftime("%d %b, %Y")
        }

        node, class_count, _, _ = read_financial_leaf_sliding_window(
            data_config=data_config,
            time_config=time_config
        )

        clear_dict = {**data_config, **time_config, **trainer_config, **model_config}

        trainer = ConformerOrdinalTrainer(node, trainer_config, model_config, class_count, device,
                                          get_triple_barrier_statistics())

        task = Task.init(project_name=project_name,
                         task_name=node.node_name,
                         tags=["test_training"],
                         reuse_last_task_id=False)

        trainer.logger = task.get_logger()
        trainer.plot_hist_classes()
        task.connect(clear_dict)

        trainer.train_model(epochs)

        good_count += trainer.register["test_good_count"][-1]
        all_played += trainer.register["test_all_played"][-1]

        task.close()

        task_sum = Task.get_task(project_name=project_name,
                                 task_name="Summary")

        task_sum.upload_artifact(name='good_count', artifact_object=good_count)
        task_sum.upload_artifact(name='all_played', artifact_object=all_played)

        logger_sum = task_sum.get_logger()
        val = trainer.register["test_good_count"][-1] / trainer.register["test_all_played"][-1] if \
            trainer.register["test_all_played"][-1] > 0 else 0
        logger_sum.report_scalar(title="Play_Accuracy", series="play_acc", iteration=mit, value=val)
        task_sum.close()

    task_sum = Task.get_task(project_name=project_name,
                             task_name="Summary")
    logger_sum = task_sum.get_logger()

    logger_sum.report_histogram(
        "Play accuracy",
        "play acc",
        values=[good_count / all_played],
        labels=[data_config["dataset_name"]]
    )
    logger_sum.report_histogram(
        "Population size",
        "played",
        values=[all_played],
        labels=[data_config["dataset_name"]]
    )

    task_sum.close()
