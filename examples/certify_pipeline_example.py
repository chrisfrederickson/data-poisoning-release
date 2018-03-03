""" Pipeline Example """

from certml.pipeline import Pipeline
from certml.classifiers import LinearSVM
from certml.defenses import DataOracle
from certml.certify.poison import UpperBound
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

x, y = make_blobs(n_samples=100, n_features=2, centers=2)

steps = [
    ('Data Oracle', DataOracle(mode='sphere', radius=5)),
    ('Linear SVM', LinearSVM(upper_params_norm_sq=1, use_bias=True))
]

pipeline = Pipeline(steps)

pipeline.fit_trusted(x, y)
pipeline.fit(x, y)

pred = pipeline.predict(x)

cert_params = pipeline.cert_params()

bounds = UpperBound(pipeline=pipeline, norm_sq_constraint=1,
                    max_iter=100, num_iter_to_throw_out=10,
                    learning_rate=1, verbose=True, print_interval=500)

epsilons = np.array([0, 0.1, 0.2, 0.3])
total_loss, good_loss, bad_loss = bounds.certify(epsilons)

plt.subplot(1, 2, 1)
plt.title('True Labels')
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.subplot(1, 2, 2)
plt.title('Prediction')
plt.scatter(x[:, 0], x[:, 1], c=pred)
plt.show()

plt.plot(epsilons, total_loss, label='Total Loss')
plt.plot(epsilons, good_loss, label='Good Loss')
plt.plot(epsilons, bad_loss, label='Bad Loss')
plt.legend()
plt.title('Upper Bounds')
plt.xlabel('Epsilons')
plt.ylabel('Loss')
plt.show()
