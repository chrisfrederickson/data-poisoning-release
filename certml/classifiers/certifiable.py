"""Certifiable Classifier"""


class CertifiableMixin:
    """Certifiable Classifier"""

    def __init__(self):
        pass

    def cert_x(self):
        raise NotImplementedError('Training features not implemented!')

    def cert_y(self):
        raise NotImplementedError('Training labels not implemented!')

    def cert_loss(self, X, y):
        raise NotImplementedError('Classifier loss not implemented!')

    def cert_loss_grad(self, X, y):
        raise NotImplementedError('Classifier loss gradient not implemented!')
