# {{REPO_NAME}}


## Install

```bash
pip install "{{REPO_NAME}} @ git+ssh://git@github.com/{{REPO_OWNER}}/{{REPO_NAME}}.git"

# Install all dev dependencies (tests etc.)
pip install "{{REPO_NAME}}[all] @ git+ssh://git@github.com/{{REPO_OWNER}}/{{REPO_NAME}}.git"

# Setup pre-commit hooks
pre-commit install
```


## Test

Tests run automatically through GitHub Actions.
* Fast tests run on each push.
* Slow tests (decorated with `@pytest.mark.slow`) run on each PR.

You can also run tests manually with `pytest`:
```bash
pytest {{REPO_NAME}}

# For all tests, including slow ones:
pytest --slow {{REPO_NAME}}
```


## Release

[Optional] Requires either PyPI or Docker GHA workflows to be enabled.

Just tag a new release in this repo, and GHA will automatically publish Python wheels and/or Docker images.
