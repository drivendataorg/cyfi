# Maintainers documentation

This page contains documentation for `cyfi` maintainers.

## Release instructions

To release a new version of `cyfi`:

1. Bump the version in [`pyproject.toml`](https://github.com/drivendataorg/cyfi/blob/main/pyproject.toml). The version number should follow [semantic versioning](https://semver.org/).
2. Specify the release date in the [CHANGELOG.md](https://github.com/drivendataorg/cyfi/blob/main/CHANGELOG.md) and ensure relevant PRs are captured.
3. Run `make docs` and commit the changes from the previous two steps.
4. Create a new release using the [GitHub releases UI](https://github.com/drivendataorg/cyfi/releases/new). The tag version must have a prefix `v`, e.g., `v1.0.1`.

On publishing of the release, the [`release`](https://github.com/drivendataorg/cyfi/blob/main/.github/workflows/release.yml) GitHub action workflow will be triggered. This workflow builds the package and publishes it to PyPI. You will be able to see the workflow status in the [Actions tab](https://github.com/drivendataorg/cyfi/actions/workflows/release.yml).
