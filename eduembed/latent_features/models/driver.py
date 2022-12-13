import numpy as np
from eduembed.latent_features import TransE
# from ampligraph.evaluation import evaluate_performance, mrr_score, hits_at_n_score
from eduembed.evaluation import evaluate_performance, mrr_score, hits_at_n_score

model = TransE(batches_count=1, seed=555, epochs=20,
               k=10, loss='pairwise',
               loss_params={'margin':5})
X = np.array([['a', 'y', 'b'],
              ['b', 'y', 'a'],
              ['a', 'y', 'c'],
              ['c', 'y', 'a'],
              ['a', 'y', 'd'],
              ['c', 'y', 'd'],
              ['b', 'y', 'c'],
              ['f', 'y', 'e']])

model.fit(X)
model.predict(np.array([['f', 'y', 'e'], ['b', 'y', 'd']]))

