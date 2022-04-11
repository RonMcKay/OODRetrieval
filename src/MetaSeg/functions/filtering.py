import logging

import numpy as np


class QuantileFilter(object):
    def __init__(self, quantile, lower=False, values=None):
        """When called returns a boolean array that can be used to filter an array.

        Bool values are calculated based on the specified quantile. If 'values' is not
        None, the quantile is calculated based on these. Otherwise the quantile is
        calculated on the fly in the call method.

        Args:
            quantile (float): Quantile value to use for filtering the ious. Should be
                in the interval (0,1)
            lower (bool): If True takes the quantile from the lower end of the
                distribution. Else from the upper end. Default: False
            values (numpy array): Values to compute the quantile from.
        """
        self.quantile = quantile
        self.lower = lower
        self.log = logging.getLogger(__name__ + ".QuantileFilter")

        if values is not None:
            self.q = np.quantile(values, self.quantile)
        else:
            self.q = None

    def __call__(self, values):
        self.log.debug("Applying QuantileFilter")
        if self.q is None:
            q = np.quantile(values, self.quantile)
        else:
            q = self.q
        return values <= q if self.lower else values >= q


class PredFilter(object):
    def __init__(self, valid_classes):
        """When called returns a boolean array that can be used to filter an array.

        Bool values are calculated based on the specified valid classes.

        Args:
            valid_classes (sequence of ints): Sequence of classes that are used
                for filtering.
        """
        self.valid_classes = valid_classes
        self.log = logging.getLogger(__name__ + ".PredFilter")

    def __call__(self, pred):
        self.log.debug("Applying PredFilter")
        return np.isin(pred, self.valid_classes)


class ValueFilter(object):
    def __init__(self, value, lower=True):
        self.value = value
        self.lower = lower
        self.log = logging.getLogger(__name__ + ".ValueFilter")

    def __call__(self, values):
        self.log.debug("Applying ValueFilter")
        return values < self.value if self.lower else values >= self.value
