import numpy as np
import logging


def iou_numpy(pred: np.ndarray,
              target: np.ndarray,
              n_classes: int,
              update_matrix: np.ndarray = None,
              ignore_index: int = 255,
              mask: np.ndarray = None,
              average: bool = True,
              eps: float = 1e-6):
    log = logging.getLogger('iou_numpy')
    if mask is None:
        mask = target != ignore_index
    else:
        mask = mask.astype(np.bool) & (target != ignore_index) & (pred != ignore_index)

    log.debug('mask shape: {}'.format(mask.shape))
    pred = pred[mask]
    target = target[mask]

    log.debug('pred shape: {}'.format(pred.shape))
    log.debug('target shape: {}'.format(target.shape))

    if target.shape[0] > 0:
        confusion_matrix = np.zeros((n_classes, n_classes))

        for i, j in zip(pred, target):
            confusion_matrix[i, j] += 1

        tp = np.diag(confusion_matrix)
        fn = confusion_matrix.sum(0) - tp
        fp = confusion_matrix.sum(1) - tp

        if average:
            iou_value = (tp / (tp + fp + fn + eps)).mean()
        else:
            iou_value = tp / (tp + fp + fn + eps)
    else:
        confusion_matrix = np.zeros((n_classes, n_classes))
        if average:
            iou_value = 1.0
        else:
            iou_value = np.ones(n_classes)

    if update_matrix is not None:
        return iou_value, update_matrix + confusion_matrix
    else:
        return iou_value
