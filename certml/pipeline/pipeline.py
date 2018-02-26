"""Pipeline"""


class Pipeline(object):
    """Pipeline"""

    def __init__(self, steps):
        self.steps = steps

    def predict(self, X):
        """Apply the transformations and predict the final estimator"""
        x_step = X
        for name, step in self.steps[:-1]:
            x_step, used = step.transform(x_step)
        return self._final_estimator.predict(x_step)

    def fit(self, X, y):
        """Apply the transformations and fit the final estimator"""
        x_step = X
        y_step = y
        for name, step in self.steps[:-1]:
            x_step, y_step, used = step.transform(x_step, y_step)
        return self._final_estimator.fit(x_step, y_step)

    def partial_fit(self, X, y):
        """Apply the transformations and then partial fit the final estimator"""
        x_step = X
        y_step = y
        for name, step in self.steps[:-1]:
            x_step, y_step, used = step.transform(x_step, y_step)
        return self._final_estimator.partial_fit(x_step, y_step)

    @property
    def _final_estimator(self):
        return self.steps[-1][-1]
