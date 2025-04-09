"""
Implements a truncated Lévy distribution so that we can sample from it.
"""
import scipy.stats as ss
import numpy as np
from scipy.special import gamma, gammainc, exp1


def upper_gamma(a, x):
    """
    Calculate upper gamma function.
    """
    return gamma(a) * (1 - gammainc(a, x))


def exp_integral_e(n, z):
    """
    Calculate exponential n integral.
    :param n:
    :param z:
    :return:
    """
    return z ** (n - 1) * upper_gamma(1 - n, z)


class TruncatedLevy(ss.rv_continuous):
    """
    Distribution for an exponentially weighted Levy distribution.
    """

    def __init__(self, beta=0.75, x0=1.5, k=80):
        super().__init__(a=0, b=np.inf)
        self.beta = beta
        self.x0 = x0
        self.k = k

        self.A = self.normalisation_constant()

    def normalisation_constant(self):
        """
        Calculate the normalising constant for the PDF.
        :return:
        """
        return np.exp(self.x0 / self.k) * self.k ** (1 - self.beta) * upper_gamma(1 - self.beta, self.x0 / self.k, )

    def _get_support(self, *args, **kwargs):
        """
        Return the support of the distribution.
        """
        return 0, np.inf

    def _pdf(self, x):
        return (self.x0 + x) ** (-self.beta) * np.exp(-x / self.k) / self.A
