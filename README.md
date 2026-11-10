# flow (moved)

As of v0.3.0, `flow` lives in [act65/sinterp](https://github.com/act65/sinterp), which
ships both packages, `sinterp` and `flow`, in one distribution:

```bash
pip install "sinterp-jax @ git+https://github.com/act65/sinterp@v0.3.0"
```

```python
import flow, sinterp
```

The import name is unchanged. This repository is kept for its history; the last standalone
release is tag `v0.2.0`.
