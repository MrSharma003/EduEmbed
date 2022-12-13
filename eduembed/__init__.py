

import logging.config
import pkg_resources

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

__version__ = '1.4.0'
__all__ = ['datasets', 'latent_features', 'discovery', 'evaluation', 'utils']

logging.config.fileConfig(pkg_resources.resource_filename(__name__, 'logger.conf'), disable_existing_loggers=False)
