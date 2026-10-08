# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PHYSBO (v3.2.1) is a pure-Python Bayesian optimization library (NumPy/SciPy only) for physics and
materials science, derived from COMBO. It scales to large candidate sets via Thompson sampling on a
random-feature Bayesian linear model, rank-one Cholesky updates, and automatic hyperparameter tuning.
License is MPL-2.0; every source file starts with the SPDX header block used in `src/physbo/__init__.py`.
Supported Python is 3.9 through 3.13 (CI also tests NumPy 1.26.4), so avoid syntax newer than 3.9
(`from __future__ import annotations` is used where needed).

## Commands

Development uses `uv` (recommended by README) or plain pip. The package is `src/`-layout.

```bash
uv sync --extra tests            # or: python3 -m pip install -e ".[tests]"
uv run pytest tests              # full suite (CI runs: uv run pytest --capture=sys tests)
uv run pytest tests/unit         # fast unit tests only
uv run pytest tests/integrated/test_discrete.py::TestDiscrete::test_random_search   # single test
```

Important: every test file begins with `physbo = pytest.importorskip("physbo")`, and `tests/` has no
`conftest.py` on purpose (see `tests/README.md`): tests exercise the *installed* package. If physbo is
not installed in the active environment, the whole suite is silently skipped and reports success.
Install (editable is fine) before trusting a green run.

```bash
uv sync --extra docs && uv run bash docs/make_docs.sh    # builds en+ja HTML into docs/built (needs pandoc)
uv run bash docs/evaluate_notebooks.sh [--inplace] [-j N] # executes tutorial notebooks (nbsphinx never executes them)
```

`tests/benchmark/*.py` are standalone argparse scripts (need matplotlib), not pytest tests.
Lint config is ruff with only `E741` ignored.

Releasing: the version string lives in four places that must agree: `pyproject.toml`,
`src/physbo/__init__.py` (`__version__`), and `version`/`release` in both
`docs/sphinx/manual/en/source/conf.py` and `docs/sphinx/manual/ja/source/conf.py`.
Pushing a tag publishes to PyPI; docs deploy to gh-pages for `master`, `develop`, and tags.

## Architecture

### Data container: `physbo.Variable` (`src/physbo/_variable.py`)
Holds `X` (N x d inputs), `t` (N x k objective values, **always 2-D**, even for k=1), and `Z`
(k x N x n random-feature basis, **always 3-D**). `normalize_t` / `normalize_Z` coerce 1-D inputs
to these shapes. Policies and predictors pass `objective_index` to pick a column of `t` / slab of `Z`.
Any code that reads `training.t` must handle the (N, k) shape.

### Surrogate layer: predictors and models
- `physbo.predictor.BasePredictor` defines the interface (`fit`, `prepare`, `update`, `get_post_fmean`,
  `get_post_fcov`, `get_post_samples`, `get_predict_samples`, `get_basis`, `save`/`load`).
- `physbo.gp.Predictor` wraps `gp.core.Model` (exact GP: `cov.Gauss` + `mean.Const`/`Zero` + `lik.Gauss`).
  Hyperparameters are learned in `Model.fit` by `gp.core.learning.Adam` (config `learning.method="adam"`,
  default) or `learning.Batch` (`"bfgs"`/`"batch"`), with `init_params_search` random restarts.
- `physbo.blm.Predictor` is selected when `num_rand_basis > 0`: it fits the GP, then calls
  `gp.core.Model.export_blm(num_basis)` to produce a `blm.core.Model` with Fourier random features
  (`blm.basis`), which supports cheap incremental `update_stats` (rank-one updates) and sampling.
- `misc.SetConfig` (`search`: `multi_probe_num_sampling`, `alpha`; `learning`: Adam/Batch settings)
  is the shared configuration object; it can also be loaded from an INI file.

### Search layer: six `Policy` classes in `physbo.search`
The grid is {discrete, range} x {single, multi, unified}:

| module | base class | predictors | acquisition (`score`) |
|---|---|---|---|
| `discrete` | root | one | `score.py`: EI, PI, TS |
| `discrete_multi` | `discrete.Policy` | `predictor_list`, one per objective | `score_multi.py`: EHVI, HVPI, TS (uses `pareto.py`) |
| `discrete_unified` | `discrete.Policy` | one, on `unify_method(training.t)` | `score.py` |
| `range` | root (continuous, `min_X`/`max_X`) | one | `score.py`, maximized by an `optimizer` |
| `range_multi` | `range.Policy` | `predictor_list` | `score_multi.py` |
| `range_unified` | `range.Policy` | one, unified | `score.py` |

- Discrete policies take `test_X` (candidate matrix) and work on integer **actions** (row indices);
  `self.actions` is the not-yet-evaluated set and is split across MPI ranks when `comm` is given.
- Range policies take `min_X`/`max_X` and maximize the acquisition with an optimizer object:
  `search.optimize.random.Optimizer` (default, random sampling) or `search.optimize.odatse.Optimizer`
  (wraps ODAT-SE algorithms: exchange, pamc, minsearch, mapper, bayes). The odatse module imports
  `odatse` at module import time and is not imported by `physbo.search`, so import it explicitly.
- Unified policies collapse k objectives to one scalar via `search.unify.ParEGO` or `search.unify.NDS`
  (objectives are min-max scaled to [-1, 0] first) and then reuse the single-objective machinery.
- Each policy module has its own `_history.py` (`History`): fixed-size preallocated arrays
  (`MAX_SEARCH = 30000`), per-step timings, and `export_sequence_best_fx` /
  `export_all_sequence_best_fx` (multi variants also track the Pareto set).

### The `bayes_search` loop (same shape in every policy)
Per probe: learn hyperparameters when `utility.is_learning(n, interval)` (`interval<0` never,
`0` first step only), otherwise `_update_predictor()` pushes `self.new_data` incrementally; then
`_get_actions(score, N, K, alpha)` (marginal scores for `num_search_each_probe > 1`), run the
`simulator`, and `write()` results into `training` + `history`. Calling with `simulator=None`
(or `max_num_probes=None`) returns the proposed actions instead of looping, which is the
"interactive mode" used in the tutorials. `write()` adds to `training` but not to the predictor;
`_update_predictor()` does that.

### Other pieces
- `physbo.test_functions` has `single_objective` and `multi_objective` benchmark functions with
  `min_X`, `max_X`, `constraint`, used by `search.utility.make_grid` / `utility.Simulator` and
  the `tests/benchmark` scripts.
- Lowercase factory names (`physbo.variable`, `physbo.gp.model`, `physbo.set_config`, ...) are
  deprecated shims that print a warning once; new code uses the CapWords classes.
- Checkpoints go through `Policy.save/load` (history `.npz`, training/predictor `.dump`); these
  extensions are gitignored.

## Documentation layout

`docs/sphinx/manual/{en,ja}/source` are two parallel, hand-maintained Sphinx trees (not gettext
translations; `_locales/ja` only covers Sphinx UI strings). Tutorials are the notebooks under
`*/source/notebook/`; both language sets are executed in CI on Python 3.9 and 3.13, so changes to
the API must keep all notebooks runnable. API pages are regenerated by `sphinx-apidoc` in
`make_docs.sh` (output `source/api` is gitignored). `copy_tutorials.py` exports the English
notebooks to the external PHYSBO Gallery repository in CI.

The README's "simple example" link points to `examples/simple.py`, which does not exist; the
equivalent script is `examples/single_objective/bayes_search.py`.
