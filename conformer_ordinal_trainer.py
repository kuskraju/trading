import numpy
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from Conformer.ConformerOrdinal import ConformerOrdinal
from abstract_data_sets import AbstractDataSet


class ConformerOrdinalTrainer:
    def __init__(self, data_sets_tree: AbstractDataSet, trainer_config, model_config, class_count, device, model_name,
                 statistics):
        self.data_sets = data_sets_tree
        self.trainer_config = trainer_config
        self.model_config = model_config
        self.optimizer_step_at_each_batch = trainer_config["optimizer_step_at_each_batch"]
        self.initial_learning_rate = trainer_config["initial_learning_rate"]
        self.batch_size = trainer_config["batch_size"]
        self.class_count = class_count
        self.device = device
        self.model_name = model_name
        self.statistics = statistics
        self.logger = None
        self.model = None
        pass

    def _get_data_loaders(self):
        return DataLoader(self.data_sets.get_wrapped_train(), batch_size=self.batch_size, shuffle=True),\
               DataLoader(self.data_sets.get_wrapped_test(), batch_size=self.batch_size)

    def _get_optimiser(self, model):
        return optim.Adam(model.parameters(), lr=self.initial_learning_rate)

    def _get_scheduler(self, optimizer):
        return optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

    def _get_model(self):
        return ConformerOrdinal(
            self.device,
            self.model_config["d_model"],
            self.model_config["n"],
            self.model_config["heads"],
            self.model_config["dropout"],
            self.data_sets.features_count(),
            self.data_sets.sequence_length(),
            self.class_count
        )

    def _get_loss_function(self):
        return nn.CrossEntropyLoss(reduction='sum')

    def _per_epoch_register(self, key, value):
        if key not in self.epoch_register.keys():
            self.epoch_register[key] = []

        self.epoch_register[key].append(value)
        pass

    def _optimizer_operations(self):
        torch.cuda.empty_cache()
        self.optimizer.zero_grad()
        self.gradient_loss.backward()
        self.optimizer.step()
        self.gradient_loss = 0.0
        pass

    def _train_model_single_run(self, epoch_count, name):
        print(name + " starting training: %d epochs" % epoch_count)
        self.model = self._get_model()
        self.optimizer = self._get_optimiser(self.model)
        self.scheduler = self._get_scheduler(self.optimizer)
        self.loss_function = self._get_loss_function()
        self.train_loader, self.test_loader = self._get_data_loaders()
        for epoch_no in range(epoch_count):
            self._train_epoch(epoch_no)
            self._test_model(epoch_no)

    def _train_epoch(self, epoch_no):
        self.gradient_loss = 0.0
        self.epoch_loss = 0.0
        self.model.train()
        predictions = []
        targets = []
        for batch_no, data in enumerate(self.train_loader):
            source = data["x"].to(self.device)
            target = data["y"].to(self.device)
            prediction = self.model(source)

            batch_loss = self.loss_function(prediction, target)
            self.gradient_loss += batch_loss
            self.epoch_loss += batch_loss

            predictions.extend(prediction.cpu().detach().numpy())
            targets.extend(target.cpu().detach().numpy())
            if self.optimizer_step_at_each_batch:
                self._optimizer_operations()  # what may happen here is that gradient_loss may get zeroed

        self.logger.report_scalar(title='epoch_loss', series='Train', iteration=epoch_no, value=self.epoch_loss)
        if not self.optimizer_step_at_each_batch:
            self._optimizer_operations()
        self.scheduler.step()

        self._per_epoch_register("train_loss", self.epoch_loss.item())
        classifications = numpy.argmax(predictions, axis=1)
        for key in self.statistics.keys():
            stat = self.statistics[key](classifications, targets)
            self._per_epoch_register("train_%s" % key, stat)
            if stat is not None:
                self.logger.report_scalar(title=key, series='Train', iteration=epoch_no, value=stat)
        pass

    @torch.no_grad()
    def _eval_model(self, loader, prefix, epoch_no=0, one_batch_only=False):
        self.epoch_loss = 0.0
        self.model.eval()
        predictions = []
        targets = []
        for batch_no, data in enumerate(loader):
            source = data["x"].to(self.device)
            target = data["y"].to(self.device)
            prediction = self.model(source)
            batch_loss = self.loss_function(prediction, target)
            self.epoch_loss += batch_loss
            predictions.extend(prediction.cpu().detach().numpy())
            targets.extend(target.cpu().detach().numpy())
            self.model_right = (numpy.argmax(predictions, axis=1) == targets)
            if one_batch_only:
                break
        self.logger.report_scalar(title='epoch_loss', series=prefix, iteration=epoch_no, value=self.epoch_loss)
        self.eval_predictions = predictions
        self.eval_targets = targets
        if prefix is not None and len(loader) > 0:
            classifications = numpy.argmax(predictions, axis=1)
            self._per_epoch_register("%s_loss" % prefix, self.epoch_loss.item())
            for key in self.statistics.keys():
                stat = self.statistics[key](classifications, targets)
                self._per_epoch_register("%s_%s" % (prefix, key), stat)

                if stat is not None:
                    self.logger.report_scalar(title=key, series=prefix, iteration=epoch_no, value=stat)
        pass

    @torch.no_grad()
    def _test_model(self, epoch_no):
        self._eval_model(self.test_loader, "test", epoch_no)

    def train_model(self, epoch_count, name=""):
        self.epoch_register = {}
        self._train_model_single_run(epoch_count, name=name)
        self.register = self.epoch_register

    def plot_hist_classes(self):
        self.logger.report_histogram(
            "Train Classes distribution",
            "labels",
            values=numpy.unique(self.data_sets.target_train, return_counts=True)[1],
            xaxis="class no",
            yaxis="class count",
        )
        self.logger.report_histogram(
            "Test Classes distribution",
            "labels",
            values=numpy.unique(self.data_sets.target_test, return_counts=True)[1],
            xaxis="class no",
            yaxis="class count",
        )