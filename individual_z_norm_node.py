from abstract_data_sets import AbstractDataSet
import numpy


class IndividualZNormNode(AbstractDataSet):
    def __init__(self, subtree: AbstractDataSet):
        super(IndividualZNormNode, self).__init__(subtree)
        self._copy_target(subtree)
        self.train = numpy.full(subtree.train.shape, None, dtype=numpy.float64)
        self.test = numpy.full(subtree.test.shape, None, dtype=numpy.float64)
        if not numpy.isnan(subtree.train).any():
            self._normalize_single_sequence(subtree.train, self.train, subtree.train_sequences_count(),
                                            subtree.features_count())
        if not numpy.isnan(subtree.test).any():
            self._normalize_single_sequence(subtree.test, self.test, subtree.test_sequences_count(),
                                            subtree.features_count())

        self.node_name = "%s_ind-znorm" % subtree.node_name
        pass

    @staticmethod
    def _normalize_single_sequence(source_set, target_set, seq_count, features_count):
        epsilon = 1e-16

        for seq in range(seq_count):
            mean = numpy.mean(source_set[seq], axis=0)
            sd = numpy.std(source_set[seq], axis=0)
            if (sd > epsilon).all():
                target_set[seq] = (source_set[seq] - mean) / sd
            else:
                for f in range(features_count):
                    if sd[f] > epsilon:
                        target_set[seq] = (source_set[seq] - mean[f]) / sd[f]
                    else:
                        raise FloatingPointError

    def _copy_target(self, subtree):
        self.target_train = subtree.target_train
        self.target_test = subtree.target_test
        self.available_classes_count = subtree.available_classes_count
