from typing import Callable, Optional
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import gammaln, xlogy


EPS = 1e-8

class PoissonDist:
    """Poisson with rate lam > 0; supports broadcasting over batch/event dims."""
    def __init__(self, rate_parameters: jnp.ndarray) -> None:
        self.lam = jnp.clip(rate_parameters, EPS, None)  # (..., n_drop)
        self.event_shape = self.lam.shape[-1:]

    def log_prob(self, y: jnp.ndarray) -> jnp.ndarray:
        y = jnp.asarray(y)

        # log pmf: y*log(lam) - lam - lgamma(y+1)
        lp = xlogy(y, self.lam) - self.lam - gammaln(y + 1.0)

        return lp  # (..., n_drop)

    def sample(self, seed, sample_shape=()):
        return jr.poisson(seed, lam=self.lam, shape=sample_shape + self.lam.shape)

class NegBinomDist:
    r"""Negative Binomial in (mu, r) parametrization:
    Var[Y] = mu + mu^2 / r,  r > 0  (inverse dispersion)
    Equivalent to Poisson-Gamma mixture: lam ~ Gamma(r, rate=r/mu),  Y|lam ~ Poisson(lam).
    """
    def __init__(self, mu, r):
        mu = jnp.clip(mu, EPS, None)
        r  = jnp.clip(r,  EPS, None)
        self.mu, self.r = mu, r
        self.event_shape = mu.shape[-1:]
        # p = r/(r+mu)
        self.p = r / (r + mu)

    def log_prob(self, y):
        y = jnp.asarray(y)
        r, p = self.r, self.p
        # log pmf: log Γ(y+r) - log Γ(r) - log Γ(y+1) + r*log p + y*log(1-p)
        lp = (gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
              + r * jnp.log(p) + y * jnp.log1p(-p))
        return lp  # (..., n_drop)

    def sample(self, seed, sample_shape=()):
        # sample via Gamma-Poisson mixture to avoid relying on jr.negative_binomial
        r, p = self.r, self.p
        # Gamma shape=r, rate= r/mu  -> equivalently scale = mu/r
        lam = jr.gamma(seed, a=r, shape=sample_shape + r.shape) * (self.mu / r)
        return jr.poisson(seed, lam=lam)

class StudentT:
    """
    Univariate Student's t with parameters:
      df   > 0
      loc  ∈ R
      scale> 0
    Broadcasting supported. This is a *base* dist; use your IndependentWrapper
    to reinterpret last dim(s) as event dims if needed.
    """

    def __init__(self, df, loc=0.0, scale=1.0, eps=1e-8, dtype=jnp.float32):
        self.df    = jnp.asarray(df,   dtype=dtype)
        self.loc   = jnp.asarray(loc,  dtype=dtype)
        self.scale = jnp.asarray(scale,dtype=dtype)
        self.eps   = eps
        # guardrails
        self.scale = jnp.maximum(self.scale, self.eps)
        self.df    = jnp.maximum(self.df,    self.eps)

    @property
    def event_shape(self):
        return ()  # scalar event per component

    def _log_norm(self):
        nu = self.df
        return (
            gammaln(0.5*(nu + 1.0))
            - gammaln(0.5*nu)
            - 0.5*(jnp.log(nu) + jnp.log(jnp.pi))
            - jnp.log(self.scale)
        )

    def log_prob(self, x):
        x = jnp.asarray(x, dtype=self.loc.dtype)
        z = (x - self.loc) / self.scale
        nu = self.df
        log_kernel = -0.5*(nu + 1.0) * jnp.log1p((z*z) / nu)
        return self._log_norm() + log_kernel  # shape = broadcast(x, params)

    def sample(self, key, sample_shape=()):
        """
        Reparameterization:
          Z ~ Normal(0,1)
          G ~ Gamma(nu/2, 1)  (shape α=ν/2, scale=1)  →  V=2G ~ ChiSquare(ν)
          T = Z / sqrt(V/ν)
          X = loc + scale * T
        """
        key_z, key_g = jax.random.split(key, 2)
        out_shape = sample_shape + jnp.shape(self.loc)
        z  = jax.random.normal(key_z, shape=out_shape, dtype=self.loc.dtype)
        g  = jax.random.gamma(key_g, a=0.5*self.df, shape=out_shape, dtype=self.loc.dtype)
        t  = z / jnp.sqrt((2.0*g) / self.df + self.eps)
        return self.loc + self.scale * t

class IndependentWrapper:
    """Minimal 'Independent' wrapper: sums over the last `ndims` dims."""
    def __init__(
            self, 
            base_dist, 
            reinterpreted_batch_ndims=1, 
            integer_transform: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
    ) -> None:
        self.base = base_dist
        self.ndims = reinterpreted_batch_ndims
        self.integer_transform = integer_transform

    def log_prob(self, value):
        # jax.debug.print("value before: {val}", val=value)
        if self.integer_transform is not None:
            value = self.integer_transform(value)
        lp = self.base.log_prob(value)
        lp_sum = lp.sum(axis=tuple(range(-self.ndims, 0)))  # -> (...,)
        # jax.debug.print("lp: {l}", l=lp_sum)
        # jax.debug.print("value after: {val}", val=value)
        return lp_sum

    def sample(self, seed, sample_shape=()):
        return self.base.sample(seed, sample_shape)

    def sample_and_log_prob(self, seed, sample_shape=()):
        """
        Returns (x, log_prob(x)) with:
          x.shape       = sample_shape + batch_shape + event_shape
          log_prob.shape= sample_shape + batch_shape
        """
        x = self.sample(seed, sample_shape=sample_shape)
        lp = self.log_prob(x)
        return x, lp
