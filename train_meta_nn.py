from sacred import Experiment
import logging
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from os.path import join, dirname

from src.MetaSeg.functions.meta_nn import MetaNN, MetricDataset
from configuration import CONFIG
from src.MetaSeg.functions.helper import load_data
from src.log_utils import log_config

ex = Experiment('train_meta_nn')

log = logging.getLogger()
log.handlers = []

log_format = logging.Formatter('%(asctime)s || %(name)s - [%(levelname)s] - %(message)s')

streamhandler = logging.StreamHandler(sys.stdout)
streamhandler.setFormatter(log_format)
log.addHandler(streamhandler)

log.setLevel('INFO')
ex.logger = log


@ex.config
def config():
    args = dict(
        dataset='cityscapes',
        dataset_val='cityscapes_val',
        epochs=50,
        learning_rate=1e-4,
        weight_decay=5e-4,
        batch_size=256,
        n_jobs=CONFIG.NUM_CORES,
        gpu=CONFIG.GPU_ID,
        save_folder=join(CONFIG.metaseg_io_path, 'meta_networks'),
        net_name='meta_nn.pth',
    )


@ex.automain
def train(args, _run, _log):
    log_config(_run, _log)
    os.makedirs(dirname(args['save_folder']), exist_ok=True)

    _log.info('Loading data...')
    xa, ya, _, _, xa_mean, xa_std, classes_mean, classes_std, *_ = load_data(args['dataset'])
    xa_val, ya_val, *_ = load_data(args['dataset_val'],
                                   xa_mean=xa_mean,
                                   xa_std=xa_std,
                                   classes_mean=classes_mean,
                                   classes_std=classes_std)

    dat = MetricDataset([xa, ya])
    dat_val = MetricDataset([xa_val, ya_val])

    _log.info('Training dataset size: {}'.format(len(dat)))
    _log.info('Validation dataset size: {}'.format(len(dat_val)))

    datloader = DataLoader(dat,
                           args['batch_size'],
                           shuffle=True,
                           num_workers=args['n_jobs'])
    valloader = DataLoader(dat_val,
                           args['batch_size'],
                           shuffle=True,
                           num_workers=args['n_jobs'])

    _log.info('Initializing network...')
    net = MetaNN(xa.shape[1]).cuda(args['gpu'])

    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=args['learning_rate'],
                                 weight_decay=args['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
    crit = nn.BCEWithLogitsLoss().cuda(args['gpu'])
    crit_val = nn.BCEWithLogitsLoss(reduction='none')

    min_loss = float('inf')
    for e in range(args['epochs']):
        _log.info('Epoch {}/{}'.format(e + 1, args['epochs']))

        _log.info('Training phase...')
        net.train()
        avg_loss = []
        for x, y in datloader:
            optimizer.zero_grad()
            x, y = x.cuda(args['gpu']), y.cuda(args['gpu'])
            out = net(x)

            loss = crit(out, y)
            loss.backward()
            optimizer.step()
            avg_loss.append(loss.item())
            # _run.log_scalar('batch_loss', loss.item())
        avg_loss = sum(avg_loss) / len(avg_loss)
        _run.log_scalar('train_loss', avg_loss)

        _log.info('Validation phase...')
        net.eval()
        avg_val_loss = []
        with torch.no_grad():
            for x, y in valloader:
                x = x.cuda(args['gpu'])
                out = net(x).data.cpu()
                avg_val_loss.append(crit_val(out, y))
        avg_val_loss = torch.cat(avg_val_loss).mean().item()
        _run.log_scalar('val_loss', avg_val_loss)
        if avg_val_loss < min_loss:
            min_loss = avg_val_loss
            _log.info('Average validation loss decreased, saved model.')
            torch.save(net.state_dict(), join(args['save_folder'], args['net_name']))

        scheduler.step()
