# Repository Guidelines

## Project Structure & Module Organization
- Core library code lives in `src/attzoo/` (one module per attention variant, plus shared helpers like `base.py`, `masks.py`, and `utils.py`).
- Public package exports are defined in `src/attzoo/__init__.py`.
- Tests live in `tests/` and generally mirror module names (for example, `src/attzoo/lsh.py` -> `tests/test_lsh.py`).
- Documentation sources are in `docs/` with MkDocs config in `mkdocs.yml`.
- Generated outputs (`site/`, `dist/`) are build artifacts; avoid manual edits.

## Build, Test, and Development Commands
- `uv sync --frozen --all-extras --dev`: install pinned runtime and dev dependencies.
- `uv run pytest`: run the full test suite.
- `uv run ruff check src tests`: run lint checks.
- `uv run mypy src`: run static type checks.
- `uv run -m mkdocs serve`: run docs locally with live reload.
- `uv run -m mkdocs build`: build static docs (also enforced in CI).

## Coding Style & Naming Conventions
- Target Python `3.12` and keep code fully type-annotated.
- Follow Ruff and Mypy rules configured in `pyproject.toml` (strict typed definitions are expected).
- Use `snake_case` for functions/variables/modules, `PascalCase` for classes.
- Keep attention class names descriptive and consistent (for example, `LongformerSelfAttention`).
- Prefer absolute imports from `attzoo` package modules.

## Testing Guidelines
- Framework: `pytest` with files named `test_*.py`.
- Add or extend tests for every behavior change, prioritizing shape contracts, mask semantics, dtype/device handling, and deterministic behavior in `eval()` mode.
- For targeted runs, use patterns like `uv run pytest tests/test_longformer.py -k global`.

## Commit & Pull Request Guidelines
- Recent history favors short, imperative, lowercase subjects (for example, `fix typing`, `update readme`, `lsh attention test`).
- Keep commits focused to one logical change.
- PRs should include what changed and why, affected modules/files, validation steps with command output (`pytest`, `ruff`, `mypy`), and linked issue(s) when applicable.
- If public API changes, update exports, docs, and README in the same PR.
