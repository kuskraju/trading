from abstract_data_sets import AbstractDataSets
import numpy


class IndividualZNormNode(AbstractDataSets):
    def __init__(self, subtree: AbstractDataSets):
        super(IndividualZNormNode, self).__init__(subtree)
        self._copy_target(subtree)
        self._manage_offset_and_variable_length(0, subtree)
        self.train = numpy.full(subtree.train.shape, None, dtype=numpy.float64) if subtree.train_sequences_count() > 0 else None
        self.test = numpy.full(subtree.test.shape, None, dtype=numpy.float64) if subtree.test_sequences_count() > 0 else None

        self._normalize_single_sequence(subtree.train, self.train, "train", subtree.train_sequences_count(), subtree.features_count())
        self._normalize_single_sequence(subtree.test, self.test, "test", subtree.test_sequences_count(), subtree.features_count())

        self.node_name = "%s_ind-znorm" % subtree.node_name
        self.print("Individual Z-Norm")
        pass

    def _normalize_single_sequence(self, source_set, target_set, set_name, seq_count, features_count):
        epsilon = 1e-6

        for seq in range(seq_count):
            mean = numpy.mean(source_set[seq, :self.get_effective_sequence_length(set_name, seq), :], axis=0)
            sd = numpy.std(source_set[seq, :self.get_effective_sequence_length(set_name, seq), :], axis=0)

            if (sd > epsilon).all():
                target_set[seq, :self.get_effective_sequence_length(set_name, seq), :] = (source_set[seq, :self.get_effective_sequence_length(set_name, seq), :] - mean) / sd
            else:
                for f in range(features_count):
                    if sd[f] > epsilon:
                        target_set[seq, :self.get_effective_sequence_length(set_name, seq), f] = (source_set[seq, :self.get_effective_sequence_length(set_name, seq), f] - mean[f]) / sd[f]
                    else:
                        target_set[seq, :self.get_effective_sequence_length(set_name, seq), f] = numpy.zeros(self.get_effective_sequence_length(set_name, seq))
