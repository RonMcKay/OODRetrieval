from sacred import Experiment
import sys
import logging
from os.path import join, basename, splitext, abspath

from src.discover import Discovery
from configuration import CONFIG
from src.log_utils import log_config

ex = Experiment('discover_embedding_space')

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
        embeddings_file=join(CONFIG.metaseg_io_path, 'embeddings_128_128_densenet201.p'),
        distance_metric='euclid',
        embedding_size=2,
        overwrite_embeddings=False,
        method='TSNE',
        n_jobs=CONFIG.NUM_CORES,
    )
    args['save_dir'] = abspath(join(CONFIG.metaseg_io_path, 'vis_' + splitext(basename(args['embeddings_file']))[0]))

    mainplot = dict(
        legend=False
    )

    tsne = dict(
        perplexity=30,
        learning_rate=200.0,
        early_exaggeration=12.0,
        verbose=3,
    )


@ex.automain
def main(args, mainplot, tsne, _run, _log):
    log_config(_run, _log)
    Discovery(**args,
              main_plot_args=mainplot,
              tsne_args=tsne)
