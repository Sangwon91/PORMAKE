# Changelog

All notable changes to **PORMAKE** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Documented per-slot linker assignment for edges in the README: edge slots that
  share an edge type are not locked together, so individual edge slots can be
  given different linkers. ([#57])

## [0.2.3] - 2026-05-30

### Fixed
- Relaxed the `pymatgen` dependency constraint by removing the `< 2024.0.0` upper
  bound (now `pymatgen >= 2023.8.10`). This resolves the unsatisfiable dependency
  set on Python 3.12, where `pymatgen >= 2024` is required. ([#36])

### Added
- NumPy-style docstrings across `src/pormake`.
- A comprehensive `pytest` test suite (unit, integration, and smoke tests) running
  on a CI matrix of Python 3.10–3.12 on Ubuntu and macOS.
- Tag-based PyPI publishing via GitHub Actions using OIDC Trusted Publishing:
  pushing a `v*` tag builds and publishes the release with no API token. ([#58], [#59])

### Changed
- Migrated project management from `poetry` to `uv`.
- Dropped support for Python 3.8 and 3.9; the minimum supported version is now
  Python 3.10.
- Routine dependency updates via Dependabot.

## [0.2.2] - 2024-06-27

_Released on PyPI. This release predates the changelog — see the
[commit history](https://github.com/Sangwon91/PORMAKE/commits/main) for details._

## [0.2.1] - 2024-01-13

### Added
- Building blocks with partial charge. Thanks to [@aniruddha-seal](https://github.com/aniruddha-seal)!

## [0.2.0] - 2023-09-24

### Added
- A module for extracting building blocks from MOFs (MOF Decomposer).

## [0.1.2] - 2023-09-07

_Released on PyPI._

## [0.1.1] - 2022-12-31

_Released on PyPI._

## [0.1.0] - 2022-12-31

- Initial public release on PyPI.

[Unreleased]: https://github.com/Sangwon91/PORMAKE/compare/v0.2.3...HEAD
[0.2.3]: https://pypi.org/project/pormake/0.2.3/
[0.2.2]: https://pypi.org/project/pormake/0.2.2/
[0.2.1]: https://pypi.org/project/pormake/0.2.1/
[0.2.0]: https://pypi.org/project/pormake/0.2.0/
[0.1.2]: https://pypi.org/project/pormake/0.1.2/
[0.1.1]: https://pypi.org/project/pormake/0.1.1/
[0.1.0]: https://pypi.org/project/pormake/0.1.0/
[#36]: https://github.com/Sangwon91/PORMAKE/issues/36
[#57]: https://github.com/Sangwon91/PORMAKE/pull/57
[#58]: https://github.com/Sangwon91/PORMAKE/issues/58
[#59]: https://github.com/Sangwon91/PORMAKE/pull/59
