import jax
from jax import numpy as jnp
import distrax as dx
from jax.nn import sigmoid


def logit(p):
    return jnp.log(p) - jnp.log1p(-p)

class UniformIntervalSigmoid:
    '''
    Take a 1D sample on R and map to (low, high), i.e. constrained prior space.
    Contains forward and inverse functions with log det associated with this bijection.
    '''
    _eps = 1e-6

    def __init__(self, low: float, high: float) -> None:
        self.low = low
        self.high = high
        self.span = high - low

    # base z -> theta
    def forward_and_log_det(self, z: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        p = sigmoid(z) # z is unconstrained (-inf, inf)
        theta = self.low + self.span * p # constrained [low, high]
        log_det_jac = jnp.log(self.span) + jnp.log(p) + jnp.log1p(-p)
        return theta, log_det_jac
    
    # theta -> base z
    def inverse_and_log_det(self, theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        logit_argument = (theta - self.low) / self.span # [0, 1]
        p = jnp.clip( logit_argument, self._eps, 1.0 - self._eps ) # [a bit > 0, a bit < 1]
        z = logit(p) # unconstrained (-inf, inf)
        log_det_jac = -jnp.log(self.span) - jnp.log(p) - jnp.log1p(-p)
        return z, log_det_jac

class LatitudeBijector:
    '''
    Transformations from [-pi / 2, pi / 2] to -inf, inf via sin and sigmoid functions.
    '''
    _eps = 1e-6

    def __init__(self) -> None:
        pass

    def forward_and_log_det(self, z: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        sigma = sigmoid(z)              # (-inf, inf) -> [0, 1]
        t = 2. * sigma - 1.             # [-1, 1]
        log_det_jac = (
          - 0.5 * jnp.log1p(-t * t)
          + jnp.log(2)
          + jnp.log(sigma)
          + jnp.log1p(-sigma)
        )
        b = jnp.arcsin(t)               # [-pi/2, pi/2]
        return b, log_det_jac

    def inverse_and_log_det(self, b: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        logit_argument = 0.5 * (jnp.sin(b) + 1)
        sigma = jnp.clip(logit_argument, self._eps, 1 - self._eps)
        t = 2. * sigma - 1.
        z = logit(sigma)
        log_det_jac = (
          + 0.5 * jnp.log1p(-t * t)
          - jnp.log(2)
          - jnp.log(sigma)
          - jnp.log1p(-sigma)
        )
        return z, log_det_jac

