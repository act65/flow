import jax.numpy as jnp
from jax import random
from sinterp import exact_si_gmm, exact_si_gaussian
from sinterp.interpolants import LinearDeterministic

key = random.PRNGKey(0)

def test_si():
    n = 200
    z = jnp.linspace(-5, 5, n)
    t = jnp.linspace(0, 1, n)

    d = 1
    K = 3
    m1s = [jnp.array([i]) for i in jnp.linspace(-K, K, K)]
    m2s = [jnp.array([i]) for i in jnp.linspace(-K, K, K)]

    c1s = [jnp.array([[0.05]]) for _ in range(K)]
    c2s = [jnp.array([[0.05]]) for _ in range(K)]

    w1s = jnp.ones(K) / K
    w2s = jnp.ones(K) / K

    p, v, s = exact_si_gmm.construct_p_b_s(LinearDeterministic(), m1s, m2s, c1s, c2s, w1s, w2s)

    p_val = p(z[0:1], t[0])
    v_val = v(z[0:1], t[0])
    s_val = s(z[0:1], t[0])

    assert p_val is not None
    assert v_val is not None
    assert s_val is not None

def test_g_v_gmm():
    """
    Check that the exact gaussian SI gives results equal to exact GMM with 1 mode.
    """
    global key
    m1 = jnp.array([0.0])
    m2 = jnp.array([0.5])
    c1 = jnp.array([[1.0]])
    c2 = jnp.array([[0.5]])
    w1 = jnp.array([1.0])
    w2 = jnp.array([1.0])

    interp = LinearDeterministic()

    p0, v0, s0 = exact_si_gaussian.construct_p_b_s(m1, m2, c1, c2, interp)
    p1, v1, s1 = exact_si_gmm.construct_p_b_s(interp, [m1], [m2], [c1], [c2], w1, w2)

    for _ in range(100):
        key, subkey = random.split(key)
        z = random.uniform(subkey, shape=m1.shape) * 10 - 5
        key, subkey = random.split(key)
        t = random.uniform(subkey, ())

        assert jnp.isclose(p0(z, t), p1(z, t), atol=1e-8)
        assert jnp.isclose(v0(z, t), v1(z, t), atol=1e-8)
        assert jnp.isclose(s0(z, t), s1(z, t), atol=1e-8)