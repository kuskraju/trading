import numpy

from hidden_module.abstract_data_sets import AbstractDataSet


class LeafDataSet(AbstractDataSet):
    def __init__(self, train, test, target_train, target_test, node_name=""):
        super(LeafDataSet, self).__init__(None)

        self.target_test = target_test

        self.train = numpy.float_(numpy.array(train))
        self.test = numpy.float_(numpy.array(test))
        self.target_train = target_train
        self.target_test = target_test

        self.available_classes_count = 3
        self.node_name = node_name

        print("Leaf node '%s' created" % self.node_name)

    def features_count(self):
        return numpy.size(self.test, axis=2) if not numpy.isnan(self.test).any() else numpy.size(self.train, axis=2)

    def sequence_length(self):
        return numpy.size(self.train, axis=1)

    def train_sequences_count(self):
        return numpy.size(self.train, axis=0)

    def test_sequences_count(self):
        return numpy.size(self.test, axis=0) if self.test is not None else 0
