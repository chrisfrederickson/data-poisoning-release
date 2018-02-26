import scipy.sparse as sparse

import IPython

from certpoison import data_utils as data, datasets

### This just thresholds and rounds IMDB
### Not guaranteed to actually be feasible

dataset_name = 'imdb'

weight_decay = datasets.DATASET_WEIGHT_DECAYS[dataset_name]
weight_decay = 0.17 ### HACK, need to rerun on proper weight_decay

epsilons = datasets.DATASET_EPSILONS[dataset_name]
norm_sq_constraint = datasets.DATASET_NORM_SQ_CONSTRAINTS[dataset_name]

for epsilon in epsilons:
    if epsilon == 0: continue

    attack_npz_path = datasets.get_attack_npz_path(dataset_name, weight_decay, epsilon, norm_sq_constraint)
    X_modified, Y_modified, X_test, Y_test, idx_train, idx_poison = datasets.load_attack_npz(dataset_name, attack_npz_path)

    X_modified = sparse.csr_matrix(data.rround(data.threshold(X_modified)))
    IPython.embed()

    break