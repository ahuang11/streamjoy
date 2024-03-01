# streamjoy

<p align="center">
    <em>Animate, fast and easy</em>
</p>

[![build](https://github.com/ahuang11/streamjoy/workflows/Build/badge.svg)](https://github.com/ahuang11/streamjoy/actions)
[![codecov](https://codecov.io/gh/ahuang11/streamjoy/branch/master/graph/badge.svg)](https://codecov.io/gh/ahuang11/streamjoy)
[![PyPI version](https://badge.fury.io/py/streamjoy.svg)](https://badge.fury.io/py/streamjoy)

---

**Documentation**: <a href="https://ahuang11.github.io/streamjoy/" target="_blank">https://ahuang11.github.io/streamjoy/</a>

**Source Code**: <a href="https://github.com/ahuang11/streamjoy" target="_blank">https://github.com/ahuang11/streamjoy</a>

---

## Development

### Setup environment

We use [Hatch](https://hatch.pypa.io/latest/install/) to manage the development environment and production build. Ensure it's installed on your system.

### Run unit tests

You can run all the tests with:

```bash
hatch run test
```

### Format the code

Execute the following command to apply linting and check typing:

```bash
hatch run lint
```

### Publish a new version

You can bump the version, create a commit and associated tag with one command:

```bash
hatch version patch
```

```bash
hatch version minor
```

```bash
hatch version major
```

Your default Git text editor will open so you can add information about the release.

When you push the tag on GitHub, the workflow will automatically publish it on PyPi and a GitHub release will be created as draft.

## Serve the documentation

You can serve the Mkdocs documentation with:

```bash
hatch run docs-serve
```

It'll automatically watch for changes in your code.

## License

This project is licensed under the terms of the MIT license.
# streamjoy
