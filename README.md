# grouped-query-attention-pytorch


## Install

```bash
pip install "grouped-query-attention-pytorch @ git+ssh://git@github.com/fkodom/grouped-query-attention-pytorch.git"

# Install all dev dependencies (tests etc.)
pip install "grouped-query-attention-pytorch[test,t5] @ git+ssh://git@github.com/fkodom/grouped-query-attention-pytorch.git"

# Setup pre-commit hooks
pre-commit install
```


## Test

Tests run automatically through GitHub Actions.
* Fast tests run on each push.
* Slow tests (decorated with `@pytest.mark.slow`) run on each PR.

You can also run tests manually with `pytest`:
```bash
pytest grouped-query-attention-pytorch
```


## Release

[Optional] Requires either PyPI or Docker GHA workflows to be enabled.

Just tag a new release in this repo, and GHA will automatically publish Python wheels and/or Docker images.
