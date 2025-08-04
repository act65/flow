import pytest
import jax.numpy as jnp
from sinterp.interpolants import interpolators, ConstantNoise

def check_boundaries(interp):
    t = jnp.linspace(0, 1, 100)
    a = interp.alpha(t)
    b = interp.beta(t)
    c = interp.gamma(t)
    assert jnp.isclose(a[0], 1.0, atol=1e-7)
    assert jnp.isclose(a[-1], 0.0, atol=1e-7)
    assert jnp.isclose(b[0], 0.0, atol=1e-7)
    assert jnp.isclose(b[-1], 1.0, atol=1e-7)
    assert jnp.isclose(c[0], 0.0, atol=1e-7)
    assert jnp.isclose(c[-1], 0.0, atol=1e-7)

@pytest.mark.parametrize("interp_class", interpolators)
def test_interp(interp_class):
    i = interp_class()
    if isinstance(i, ConstantNoise):
        # This interpolator has different boundary conditions for gamma
        return
    check_boundaries(i)

@pytest.mark.parametrize("interp_class", interpolators)
def test_grad(interp_class):
    i = interp_class()
    t = jnp.linspace(0, 1, 100)
    da = i.dalphadt(t[0])
    assert da.shape == ()
    shape = (28, 28, 1)
    di = i.didt(jnp.ones(shape), jnp.ones(shape), jnp.ones(shape), 0.0)
    assert di.shape == shape