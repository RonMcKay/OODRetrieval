import logging
import numpy as np
from scipy.stats import gaussian_kde


def estimate_kernel_density(pc):
    log = logging.getLogger('estimate_kernel_density')
    log.debug('Computing Gaussian Kernel...')
    values = pc.T
    return gaussian_kde(values)


def get_grid(pc, n_components, n_steps=100):
    if n_components == 1:
        x = pc[:, 0]
        delta_x = (max(x) - min(x)) / 10

        xmin = min(x) - delta_x
        xmax = max(x) + delta_x

        xx = np.linspace(xmin, xmax, num=n_steps)
        return xx, xx, xx.ravel()
    elif n_components == 2:
        x = pc[:, 0]
        y = pc[:, 1]
        delta_x = (max(x) - min(x)) / 10
        delta_y = (max(y) - min(y)) / 10

        xmin = min(x) - delta_x
        xmax = max(x) + delta_x
        ymin = min(y) - delta_y
        ymax = max(y) + delta_y

        xx, yy = np.meshgrid(np.linspace(xmin, xmax, num=n_steps), np.linspace(ymin, ymax, num=n_steps))

        return xx, yy, np.vstack([xx.ravel(), yy.ravel()])
