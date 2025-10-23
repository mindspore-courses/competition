import mindspore
from mindspore import ops
from mindspore import Tensor
from mindspore.nn.probability.distribution import Normal, Distribution, Bernoulli
from mindspore.ops import BroadcastTo
import mindspore.nn.probability.distribution as msd
import numpy as np
from mindspore.ops import operations as P


class ObservationNormalizationStatistics:
    def __init__(self):
        self.x_mean = None
        self.x_std = None
        self.log_x_mean = None
        self.log_x_std = None

    def set_statistics(self, x_mean, x_std, log_x_mean, log_x_std):
        self.x_mean = x_mean
        self.x_std = x_std
        self.log_x_mean = log_x_mean
        self.log_x_std = log_x_std

class RelaxedBernoulli():
    def __init__(self, temperature, probs=None, logits=None):
        super(RelaxedBernoulli, self).__init__()
        if probs is not None:
            self.logits = ops.log(probs) - ops.log(1 - probs)
        else:
            self.logits = logits
        self.temperature = temperature

    def sample(self, sample_shape=()):
        shape = sample_shape + self.logits.shape
        minval = mindspore.Tensor(0, dtype=mindspore.float32) 
        maxval = mindspore.Tensor(1, dtype=mindspore.float32)
        uniform = ops.uniform(shape, minval, maxval)
        gumbel = -ops.log(-ops.log(uniform))
        y = (self.logits + gumbel) / self.temperature
        return ops.sigmoid(y)
    
    def log_prob(self, value):
        log_p = ops.log(value) * (self.logits - ops.log(self.temperature)) \
                - ops.log(1 + ops.exp(self.logits - ops.log(self.temperature)))
        return log_p


class RelaxedBernoulliStraightThrough(RelaxedBernoulli):
    def __init__(self, temperature, probs=None, logits=None):
        super(RelaxedBernoulliStraightThrough, self).__init__(temperature, probs, logits)

    def rsample(self, sample_shape=()):
        # Soft sample from the RelaxedBernoulli distribution
        soft_sample = super(RelaxedBernoulliStraightThrough, self).sample(sample_shape)
        # Clamp the probabilities to avoid numerical issues
        soft_sample = ops.clip_by_value(soft_sample, 1e-6, 1 - 1e-6)
        # Quantize the soft sample to get a hard sample
        hard_sample = ops.round(soft_sample)
        # Straight-through estimator
        # hard_sample = ops.stop_gradient(hard_sample - soft_sample) + soft_sample
        return hard_sample

    def log_prob(self, value):
        # During backprop, we use the unquantized sample
        return super(RelaxedBernoulliStraightThrough, self).log_prob(value)


class GumbelSoftmaxBernoulliStraightThrough(RelaxedBernoulliStraightThrough):
    def __init__(self, temperature, probs=None, logits=None):
        super(GumbelSoftmaxBernoulliStraightThrough, self).__init__(temperature, probs, logits)
        self.probs = ops.sigmoid(self.logits)

    def log_prob(self, value):
        if self.probs is None:
            raise ValueError("probs must be defined to use this method.")
        return Bernoulli(probs=self.probs).log_prob(value)

    @property
    def mode(self):
        if self.probs is None:
            raise ValueError("probs must be defined to use this method.")
        mode = (self.probs > 0.5).astype(self.probs.dtype)
        return mode

class GammaPoisson():
    """
    Compound distribution comprising of a gamma-poisson pair, also referred to as
    a gamma-poisson mixture. The `rate` parameter for the Poisson distribution
    is unknown and randomly drawn from a Gamma distribution.
    
    :param concentration: Shape parameter (alpha) of the Gamma distribution.
    :param rate: Rate parameter (beta) for the Gamma distribution.
    """

    def __init__(self, concentration, rate, validate_args=None):
        super(GammaPoisson, self).__init__()
        concentration, rate = self._broadcast_all(concentration, rate)
        self._gamma = msd.Gamma(concentration, rate)
        self._validate_args = validate_args

    @staticmethod
    def _broadcast_all(*args):
        shape = np.broadcast_shapes(*[arg.shape for arg in args])
        return tuple(BroadcastTo(shape)(arg) for arg in args)

    @property
    def concentration(self):
        return self._gamma.concentration

    @property
    def rate(self):
        return self._gamma.rate

    def sample(self, sample_shape=()):
        rate_sample = self._gamma.sample(sample_shape)
        poisson_dist = msd.Poisson(rate_sample)
        return poisson_dist.sample()

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        post_value = self.concentration + value
        log_beta = ops.Lgamma()(self.concentration) + ops.Lgamma()(value + 1) - ops.Lgamma()(post_value)
        return (
            -log_beta
            - P.Log()(post_value)
            + P.Log()(self.concentration) * P.Log()(self.rate)
            - P.Log()(post_value) * P.Log()(1 + self.rate)
        )

    @property
    def mean(self):
        return self.concentration / self.rate

    @property
    def variance(self):
        return self.concentration / (self.rate ** 2) * (1 + self.rate)

    def expand(self, batch_shape):
        new = GammaPoisson(
            broadcast_to(self.concentration, batch_shape),
            broadcast_to(self.rate, batch_shape),
        )
        return new

    def _validate_sample(self, value):
        if not mindspore.numpy.all(value >= 0):
            raise ValueError("Sample values must be non-negative")

