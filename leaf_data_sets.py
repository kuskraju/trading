import numpy

from classification_benchmark.execution.preprocessing.tree_structure.abstract_data_sets import AbstractDataSets
from classification_benchmark.execution.preprocessing.tree_structure.node_exception import NodeException


class LeafDataSets(AbstractDataSets):
    def __init__(self, train: numpy.ndarray, val: numpy.ndarray, test: numpy.ndarray, target_train: numpy.ndarray, target_val: numpy.ndarray, target_test: numpy.ndarray, sequence_wise = False, available_classes_count = 0, node_name = "", variable_length = False):
        super(LeafDataSets, self).__init__(None)
        #all sets are arrays of order 3
        #1st index is the time series number (or sequence number, the word 'time series' and 'sequence' will be used interchangably)
        #2nd index is the position in the time series
        #3rd index is the feature number (i.e. only one value for univariate time series and so on)

        #if sequence_wise is True, then the arrangement is different (time series number, feature number, position) and thus the transpose must follow
            #it is only used when initiating the object

        # for an empty set (0 sequences) a value of None may be supplied for validation or test sets or both (but NOT for a training set)

        if available_classes_count == 0:
            available_classes_count = round(max(available_classes_count, numpy.max(target_train) + 1 if target_train is not None else 0))
            available_classes_count = round(max(available_classes_count, numpy.max(target_val) + 1 if target_val is not None else 0))
            available_classes_count = round(max(available_classes_count, numpy.max(target_test) + 1 if target_test is not None else 0))

        if train is None:
            raise NodeException("Train set must not be empty(==None) in LeafDataSets creation")
        self.train = numpy.float_(numpy.array(train)) #in leaves it is not permittable that the train set is empty
        self.val = numpy.float_(numpy.array(val)) if val is not None else None
        self.test = numpy.float_(numpy.array(test)) if test is not None else None
        self.target_train = target_train
        self.target_val = target_val
        self.target_test = target_test

        if sequence_wise:
            self.train = self.train.transpose((0, 2, 1))
            self.val = self.val.transpose((0, 2, 1)) if self.val is not None else None
            self.test = self.test.transpose((0, 2, 1)) if self.test is not None else None

        self.available_classes_count = available_classes_count
        self.node_name = node_name

        self.variable_length = variable_length

        if variable_length:
            if self.train is not None:
                self.train_effective_length = numpy.count_nonzero(numpy.any(~numpy.isnan(self.train), axis=2), axis=1)
            if self.val is not None:
                self.val_effective_length = numpy.count_nonzero(numpy.any(~numpy.isnan(self.val), axis=2), axis=1)
            if self.test is not None:
                self.test_effective_length = numpy.count_nonzero(numpy.any(~numpy.isnan(self.test), axis=2), axis=1)
        print("Leaf node '%s' created: (%i+%i+%i)x%ix%i" % (self.node_name, self.train_sequences_count(), self.val_sequences_count(), self.test_sequences_count(), self.sequence_length(), self.features_count()))
        pass

    def features_count(self):
        return numpy.size(self.train, axis=2)

    def sequence_length(self):
        return numpy.size(self.train, axis=1)

    def train_sequences_count(self):
        return numpy.size(self.train, axis=0)

    def val_sequences_count(self):
        return numpy.size(self.val, axis=0) if self.val is not None else 0

    def test_sequences_count(self):
        return numpy.size(self.test, axis=0) if self.test is not None else 0
