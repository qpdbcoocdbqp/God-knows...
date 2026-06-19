# Repository Guidelines

## Project Structure & Module Organization
This repository is currently minimal and contains a single top-level notes file:

- `READMD.md`: project notes describing a Python/Kaggle/Jane Street direction.

Keep new files organized by purpose as the project grows. Recommended layout:

- `src/`: reusable Python source code and package modules.
- `notebooks/`: exploratory Kaggle or analysis notebooks.
- `data/`: local datasets or sample inputs. Do not commit large raw datasets.
- `tests/`: automated tests mirroring `src/` module names.
- `assets/`: images, charts, or documentation media.

## Build, Test, and Development Commands
No build system, dependency file, or test runner is configured yet. When adding Python code, prefer standard commands and document them here:

- `python -m venv .venv`: create a local virtual environment.
- `.venv\Scripts\Activate.ps1`: activate the environment on Windows PowerShell.
- `pip install -r requirements.txt`: install dependencies once `requirements.txt` exists.
- `pytest`: run the test suite once tests are added.

If notebooks are added, include any required setup steps and data paths in the notebook header or a dedicated README section.

## Coding Style & Naming Conventions
Use Python 3 conventions for future source files. Follow PEP 8, use 4-space indentation, and prefer descriptive `snake_case` names for files, functions, and variables. Use `PascalCase` for classes and `UPPER_SNAKE_CASE` for constants.

Keep modules focused and small. Prefer names such as `feature_engineering.py`, `train_model.py`, and `evaluate.py` over vague names like `utils2.py`.

## Testing Guidelines
Place tests under `tests/` and name files with the `test_*.py` pattern. Test functions should describe behavior, for example `test_train_model_rejects_empty_dataset`.

For data-science code, include tests for data validation, feature transformations, metric calculations, and edge cases with small in-memory fixtures. Avoid tests that require large Kaggle datasets unless clearly marked as integration tests.

## Commit & Pull Request Guidelines
This directory is not currently initialized as a Git repository, so no existing commit convention is available. If Git is added, use short imperative commit messages such as `Add Jane Street baseline notebook` or `Document data setup`.

Pull requests should include a brief summary, changed files or modules, validation steps run, and screenshots or notebook outputs when visual results change. Link related issues or Kaggle experiments when applicable.

## Security & Configuration Tips
Do not commit Kaggle API tokens, private datasets, `.env` files, or generated model artifacts. Store secrets locally and document required environment variables without including their values.

## Python Interpreter

* venv: `$HOME/.venv/Scripts/activate`
* python path: `$HOME/.venv/Scripts/python.exe`
* uv path: `$HOME/.local/bin/uv.exe`
