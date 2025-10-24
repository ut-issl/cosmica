# Releasing COSMICA

This page describes the process of releasing a new version of the COSMICA.

## Prerequisites

- [ ] The build is passing on the `main` branch.

## Release

COSMICA follows [Semantic Versioning](https://semver.org/). Determine the new version number based on the changes since the last release:
> MAJOR version when you make incompatible API changes;
> MINOR version when you add functionality in a backward compatible manner;
> PATCH version when you make backward compatible bug fixes.

### Create a release branch

1. Create a new branch for the release (`git switch -c release/v<version>`). The branch name should be `release/v<version>`, where `<version>` is the new version number.

### Bump the version number

1. Bump the `project.version` field in the `pyproject.toml` file.
2. Commit the changes by running `git commit -am "Bump version to <version>"`.

### Merge the release branch into `main`

1. Create a pull request to merge the release branch into `main`.
2. Wait for the CI to pass.
3. Merge the pull request.

### Create a Git tag

1. Create a Git tag for the new version by running `git tag v<version>`.
2. Push the tag to GitHub by running `git push origin v<version>`.

### Create a GitHub release

1. Go to the [GitHub releases page](https://github.com/ut-issl/cosmica/releases).
2. Click "Draft a new release".
3. Choose `v<version>` in the "Choose a tag" dropdown.
4. Enter `v<version>` as the title of the release in the "Release title" field.
5. Enter the release notes in the "Write" field. The release notes should be a summary of the changes in the new version. You can click "Generate release notes" to generate the release notes based on the pull requests merged since the last release.
6. Click "Publish release".

### Update the documentation

1. Run `uv run -- mike deploy --push --update-aliases <version> latest`.

!!! tip
    The version in the `mike deploy` command is not prefixed with `v`.
