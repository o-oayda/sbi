import unittest
import numpy as np
from dipolesbi.tools.np_rngkey import (
    NPKey, prng_key, npkey_from_jax, split, fold_in, normal, integers, poisson,
    NPKeySequence
)
import jax
from jax import numpy as jnp


class TestNPKey(unittest.TestCase):

    def test_split_deterministic(self):
        key = prng_key(123)
        k1, k2 = split(key, 2)
        # Re-split from same parent -> identical children
        k1b, k2b = split(prng_key(123), 2)
        self.assertTrue(np.allclose(k1.normal((5,)), k1b.normal((5,))))
        self.assertTrue(np.allclose(k2.normal((5,)), k2b.normal((5,))))

    def test_split_independent(self):
        key = prng_key(0)
        k1, k2 = split(key, 2)
        x1 = k1.normal((1000,))
        x2 = k2.normal((1000,))
        # Independence: correlation near zero
        corr = np.corrcoef(x1, x2)[0, 1]
        self.assertLess(abs(corr), 0.1)

    def test_fold_in_changes_stream(self):
        base = prng_key(42)
        k0 = fold_in(base, 0)
        k1 = fold_in(base, 1)
        x0 = k0.integers(0, 100, shape=(10,))
        x1 = k1.integers(0, 100, shape=(10,))
        # Should differ
        self.assertFalse(np.all(x0 == x1))
        # Reproducibility
        x0b = fold_in(base, 0).integers(0, 100, shape=(10,))
        self.assertTrue(np.all(x0 == x0b))

    def test_reproducibility_normal(self):
        k = prng_key(999)
        x1 = normal(k, (5,))
        x2 = normal(prng_key(999), (5,))
        self.assertTrue(np.allclose(x1, x2))

    def test_integer_bounds(self):
        k = prng_key(7)
        vals = integers(k, 0, 10, shape=(1000,))
        self.assertTrue(np.all((0 <= vals) & (vals < 10)))

class TestNPKeyPoisson(unittest.TestCase):

    def test_poisson_deterministic(self):
        k1 = prng_key(123)
        k2 = prng_key(123)
        x1 = poisson(k1, lam=5.0, shape=(1000,))
        x2 = poisson(k2, lam=5.0, shape=(1000,))
        self.assertTrue(np.array_equal(x1, x2))

    def test_poisson_fold_in_repro(self):
        base = prng_key(7)
        a1 = poisson(fold_in(base, 0), lam=3.0, shape=(1000,))
        a2 = poisson(fold_in(base, 0), lam=3.0, shape=(1000,))
        b  = poisson(fold_in(base, 1), lam=3.0, shape=(1000,))
        self.assertTrue(np.array_equal(a1, a2))   # reproducible
        self.assertFalse(np.array_equal(a1, b))   # distinct stream

    def test_poisson_split_independence(self):
        parent = prng_key(0)
        k1, k2 = split(parent, 2)
        x1 = poisson(k1, lam=8.0, shape=(2000,))
        x2 = poisson(k2, lam=8.0, shape=(2000,))
        # Independence heuristic: correlation near zero
        corr = np.corrcoef(x1, x2)[0, 1]
        self.assertLess(abs(corr), 0.1)

    def test_poisson_moments(self):
        # Law of large numbers sanity: mean≈λ, var≈λ
        k = prng_key(999)
        lam = 12.5
        x = poisson(k, lam=lam, shape=(200_000,))
        m, v = float(np.mean(x)), float(np.var(x, ddof=0))
        self.assertAlmostEqual(m, lam, delta=0.15 * lam)  # 15% tolerance
        self.assertAlmostEqual(v, lam, delta=0.15 * lam)

    def test_poisson_broadcast_and_shape(self):
        k = prng_key(321)
        lam = np.array([2.0, 5.0, 10.0], dtype=np.float32)  # broadcast over last dim
        x = poisson(k, lam=lam, shape=(4, 3))
        self.assertEqual(x.shape, (4, 3))
        # Means should rank roughly with lambda
        means = x.astype(np.float32).mean(axis=0)
        self.assertTrue(means[0] <= means[1] <= means[2] or means[2] <= means[1] <= means[0])

    def test_poisson_dtype(self):
        k = prng_key(11)
        x = poisson(k, lam=4.0, shape=(5,), dtype=np.int32)
        self.assertEqual(x.dtype, np.int32)

class TestFromJax(unittest.TestCase):

    def test_deterministic_conversion(self):
        jk = jax.random.PRNGKey(123)
        nk1 = npkey_from_jax(jk)
        nk2 = npkey_from_jax(jk)
        x1 = nk1.normal((1000,))
        x2 = nk2.normal((1000,))
        self.assertTrue(np.allclose(x1, x2))  # same JAX key -> same NPKey stream

    def test_split_independence(self):
        jk = jax.random.PRNGKey(0)
        jk1, jk2 = jax.random.split(jk, 2)
        nk1 = NPKey.from_jax(jk1)
        nk2 = NPKey.from_jax(jk2)
        a = nk1.normal((4000,))
        b = nk2.normal((4000,))
        corr = float(np.corrcoef(a, b)[0, 1])
        self.assertLess(abs(corr), 0.05)  # heuristic independence check

    def test_fold_in_bridge(self):
        base = jax.random.PRNGKey(7)
        # fold_in on the JAX side should yield distinct NPKey streams per tag
        nk_e0 = NPKey.from_jax(jax.random.fold_in(base, 0))
        nk_e1 = NPKey.from_jax(jax.random.fold_in(base, 1))
        s0a = nk_e0.poisson(lam=5.0, shape=(2000,))
        s0b = NPKey.from_jax(jax.random.fold_in(base, 0)).poisson(lam=5.0, shape=(2000,))
        s1  = nk_e1.poisson(lam=5.0, shape=(2000,))
        # reproducible for same fold-in
        self.assertTrue(np.array_equal(s0a, s0b))
        # different for different fold-in values
        self.assertFalse(np.array_equal(s0a, s1))

    def test_shape_guard(self):
        with self.assertRaises(ValueError):
            NPKey.from_jax(jnp.array([1,2,3], dtype=jnp.uint32))  # wrong shape

class TestNPKeySequence(unittest.TestCase):

    def test_reproducible_stream_same_seed(self):
        seq1 = NPKeySequence(123)
        seq2 = NPKeySequence(123)

        # Draw a few keys and sample from each to ensure identical streams
        xs1 = []
        xs2 = []
        for _ in range(5):
            xs1.append(next(seq1).normal((1000,)).mean())
            xs2.append(next(seq2).normal((1000,)).mean())

        self.assertTrue(np.allclose(xs1, xs2))

    def test_independent_streams_different_seeds(self):
        seq1 = NPKeySequence(0)
        seq2 = NPKeySequence(1)

        a = np.concatenate([next(seq1).normal((4000,)) for _ in range(3)])
        b = np.concatenate([next(seq2).normal((4000,)) for _ in range(3)])

        corr = float(np.corrcoef(a, b)[0, 1])
        self.assertLess(abs(corr), 0.05)  # heuristic independence

    def test_constructor_accepts_various_seed_types(self):
        # int
        s_int = NPKeySequence(42)
        # Sequence[int]
        s_seq = NPKeySequence([1, 2, 3, 4])
        # NPKey
        base_key = NPKey.from_seed(7)
        s_key = NPKeySequence(base_key)

        # Just sanity: each yields a key and can sample
        _ = next(s_int).normal((3,))
        _ = next(s_seq).normal((3,))
        _ = next(s_key).normal((3,))

    def test_take_matches_next_iteration(self):
        seq = NPKeySequence(999)
        taken = seq.take(4)  # first 4 keys via take()

        # Now restart another sequence and get the first 4 via next()
        seq2 = NPKeySequence(999)
        via_next = [next(seq2) for _ in range(4)]

        # Keys should produce identical draws
        for k1, k2 in zip(taken, via_next):
            x1 = k1.poisson(lam=5.0, shape=(2000,))
            x2 = k2.poisson(lam=5.0, shape=(2000,))
            self.assertTrue(np.array_equal(x1, x2))

    def test_progression_is_stable(self):
        # If we advance one sequence further, earlier outputs remain reproducible.
        seq_a = NPKeySequence(31415)
        seq_b = NPKeySequence(31415)

        k0_a = next(seq_a); k1_a = next(seq_a)  # advance further
        k0_b = next(seq_b)

        # First key’s samples match regardless of later iteration on seq_a
        x_a = k0_a.normal((2048,))
        x_b = k0_b.normal((2048,))
        self.assertTrue(np.array_equal(x_a, x_b))

        # And second key is stable across fresh sequence
        k1_b = next(seq_b)
        y_a = k1_a.normal((2048,))
        y_b = k1_b.normal((2048,))
        self.assertTrue(np.array_equal(y_a, y_b))

if __name__ == "__main__":
    unittest.main()
