import numpy


def precision_maker(class_num):
    return lambda classification, targets, _1, _2: precision(classification, targets, class_num)


def recall_maker(class_num):
    return lambda classification, targets, _1, _2: recall(classification, targets, class_num)


def light_precision_maker(class_num):
    return lambda classification, targets, _1, _2: light_precision(classification, targets, class_num)


def light_accuracy(classification, targets, _1, _2):
    targets = numpy.asarray(targets)
    good = 0
    good += numpy.count_nonzero(numpy.all([classification != 2 - targets, targets != 1], axis=0))
    good += numpy.count_nonzero(numpy.all([classification == 1, targets == 1], axis=0))
    return float(good / len(targets))


def precision(classification, targets, class_num):
    targets = numpy.asarray(targets)
    good_per_class = 0
    num_of_class = 0
    good_per_class += numpy.count_nonzero(numpy.all([classification == targets, targets == class_num], axis=0))
    num_of_class += numpy.count_nonzero(classification == class_num)
    return float(good_per_class / num_of_class) if num_of_class > 0 else None


def light_precision(classification, targets, class_num):
    targets = numpy.asarray(targets)
    good_per_class = 0
    num_of_class = 0
    good_per_class += numpy.count_nonzero(numpy.all([classification == class_num, targets != 2 - class_num], axis=0))
    num_of_class += numpy.count_nonzero(classification == class_num)
    return float(good_per_class / num_of_class) if num_of_class > 0 else None


def recall(classification, targets, class_num):
    targets = numpy.asarray(targets)
    good_per_class = numpy.count_nonzero(numpy.all([classification == targets, targets == class_num], axis=0))
    num_of_class = numpy.count_nonzero(class_num == targets)
    return float(good_per_class / num_of_class) if num_of_class > 0 else None


def accuracy(classification, targets, _1, _2):
    targets = numpy.asarray(targets)
    return sum(classification == targets) / len(classification)


def good_count_0_2(classification, targets, _1, _2):
    targets = numpy.asarray(targets)
    good = numpy.count_nonzero(numpy.all([classification == targets, targets != 1], axis=0))
    return good


def all_played_0_2(classification, targets, _1, _2):
    targets = numpy.asarray(targets)
    count = numpy.count_nonzero(numpy.all([classification != 1, targets != 1], axis=0))
    return count


def good_count(classification, targets, _1, _2):
    targets = numpy.asarray(targets)
    good = numpy.count_nonzero(numpy.all([classification == targets, targets == 0], axis=0)) + \
           numpy.count_nonzero(numpy.all([classification == targets, targets == 1], axis=0))
    return good


def all_played(_, targets, _1, _2):
    targets = numpy.asarray(targets)
    return len(targets)


def earn_to_lose_ratio(classification, targets, _1, _2):
    targets = numpy.asarray(targets)
    earn = numpy.count_nonzero(numpy.all([classification == 2, targets == 2], axis=0)) + \
           numpy.count_nonzero(numpy.all([classification == 0, targets == 0], axis=0))
    lose = numpy.count_nonzero(numpy.all([classification == 2, targets == 0], axis=0)) + \
           numpy.count_nonzero(numpy.all([classification == 0, targets == 2], axis=0))
    return earn / lose if lose != 0 else None


def gain(classification, targets, high, low):
    targets = numpy.asarray(targets)
    earn = numpy.sum(high[numpy.all([classification == 2, targets == 2], axis=0)] - \
                     low[numpy.count_nonzero(numpy.all([classification == 0, targets == 0], axis=0))])

    lose = numpy.sum(high[numpy.all([classification == 0, targets == 2], axis=0)] - \
                     low[numpy.count_nonzero(numpy.all([classification == 2, targets == 0], axis=0))])

    return earn - lose


# get rid of class 1
def count_future_targets(targets):
    future_targets = []
    empty = 1
    for t in targets:
        if t == 1:
            empty += 1
        else:
            future_targets.append(numpy.full(empty, t))
            empty = 1
    return numpy.concatenate(future_targets, axis=0)


def get_triple_barrier_statistics():
    stat_dict = {
        "accuracy": accuracy,
        "light_accuracy": light_accuracy,
        "earn_to_lose_ratio": earn_to_lose_ratio,
        "all_played": all_played_0_2,
        "good_count": good_count_0_2,
        "gain": gain
    }
    for i in range(3):
        stat_dict.update({
            "precision_{}".format(i): precision_maker(i),
            "recall_{}".format(i): recall_maker(i),
        })
        if i != 1:
            stat_dict.update({
                "light_precision_{}".format(i): light_precision_maker(i),
            })
    return stat_dict
