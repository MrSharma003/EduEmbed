from .EmbeddingModel import EmbeddingModel, register_model
from eduembed.latent_features import constants as constants
from eduembed.latent_features.initializers import DEFAULT_XAVIER_IS_UNIFORM
import tensorflow as tf


@register_model("TransH",
                ["norm", "normalize_ent_emb", "negative_corruption_entities"])
class TransH(EmbeddingModel):
    

    def __init__(self,
                 k=constants.DEFAULT_EMBEDDING_SIZE,
                 eta=constants.DEFAULT_ETA,
                 epochs=constants.DEFAULT_EPOCH,
                 batches_count=constants.DEFAULT_BATCH_COUNT,
                 seed=constants.DEFAULT_SEED,
                 embedding_model_params={'norm': constants.DEFAULT_NORM_TRANSE,
                                         'normalize_ent_emb': constants.DEFAULT_NORMALIZE_EMBEDDINGS,
                                         'negative_corruption_entities': constants.DEFAULT_CORRUPTION_ENTITIES,
                                         'corrupt_sides': constants.DEFAULT_CORRUPT_SIDE_TRAIN},
                 optimizer=constants.DEFAULT_OPTIM,
                 optimizer_params={'lr': constants.DEFAULT_LR},
                 loss=constants.DEFAULT_LOSS,
                 loss_params={},
                 regularizer=constants.DEFAULT_REGULARIZER,
                 regularizer_params={},
                 initializer=constants.DEFAULT_INITIALIZER,
                 initializer_params={'uniform': DEFAULT_XAVIER_IS_UNIFORM},
                 verbose=constants.DEFAULT_VERBOSE):
       
        super().__init__(k=k, eta=eta, epochs=epochs,
                         batches_count=batches_count, seed=seed,
                         embedding_model_params=embedding_model_params,
                         optimizer=optimizer, optimizer_params=optimizer_params,
                         loss=loss, loss_params=loss_params,
                         regularizer=regularizer, regularizer_params=regularizer_params,
                         initializer=initializer, initializer_params=initializer_params,
                         verbose=verbose)

    def _fn(self, e_s, e_p, e_o):
        
        return tf.negative(
            tf.norm(e_s + e_p - e_o, ord=self.embedding_model_params.get('norm', constants.DEFAULT_NORM_TRANSE),
                    axis=1))

    # pass entire matrix as input X and then call parent fit method
    def fit(self, X, early_stopping=False, early_stopping_params={}, focusE_numeric_edge_values=None,
            tensorboard_logs_path=None):
        
        super().fit(X, early_stopping, early_stopping_params, focusE_numeric_edge_values,
                    tensorboard_logs_path=tensorboard_logs_path)

    def predict(self, X, from_idx=False):
        __doc__ = super().predict.__doc__  # NOQA
        return super().predict(X, from_idx=from_idx)

    def calibrate(self, X_pos, X_neg=None, positive_base_rate=None, batches_count=100, epochs=50):
        __doc__ = super().calibrate.__doc__ # NOQA
        super().calibrate(X_pos, X_neg, positive_base_rate, batches_count, epochs)

    def predict_proba(self, X):
        __doc__ = super().predict_proba.__doc__ # NOQA
        return super().predict_proba(X)
