# Development and Testing

What this page covers:
This page outlines development standards, testing workflows, and contribution guidelines for Farmer Eye.
It explains how to run the automated test suite, describes continuous integration checks, and defines coding conventions.

---

## Running Automated Tests

The repository includes a comprehensive unit testing suite in the `tests/` directory.
Tests run on standard laptops without requiring a Raspberry Pi, camera module, or motor controllers:

```bash
# Run the entire test suite with verbose output
pytest -v
```
*(Verified on this machine: 27 passed, 1 skipped gracefully when TensorFlow is absent).*

### Hardware Mocking Architecture
In [`tests/conftest.py`](../tests/conftest.py), pytest injects virtual module stubs into Python's `sys.modules` before test execution:
- `RPi.GPIO`: Mocks General Purpose Input/Output pin state changes and event listeners.
- `gpiozero`: Mocks high-level hardware device interfaces (LED, OutputDevice).
- `picamera2`: Mocks camera initialization, configuration dictionaries, and frame capture arrays.

These stubs allow developers to test frame preprocessing, JSON serialization, and motor direction mapping without physical hardware.

---

## Continuous Integration (CI)

Every commit and pull request triggers automated continuous integration through GitHub Actions:
- **Workflow File**: [`.github/workflows/tests.yml`](../.github/workflows/tests.yml).
- **Environment**: Ubuntu latest, Python 3.10.
- **Caching**: Python packages are cached with `actions/cache` using `hashFiles('requirements-dev.txt')` to speed up build times.
- **Execution**: Installs development dependencies and executes `pytest -v`.

---

## Code Style and Tooling

### Linter and Formatter
We recommend using **Ruff** for fast code analysis and formatting:

```bash
# Check code for linting errors
ruff check .

# Automatically apply safe formatting fixes
ruff format .
```

### Python Type Annotations
All modules in `src/` use standard Python 3.9+ type annotations:
- Built-in generics: `list[str]`, `dict[str, Any]`, `tuple[int, int]`.
- Explicit return types on all public functions and class methods.

### Docstring Conventions
All functions, classes, and modules follow the **Google Python Style Guide** in plain English:
- **One-line summary**: Concise description ending in a period.
- **Args**: Typed parameter names with plain explanations.
- **Returns**: Explanation of output values and structures.
- **Raises**: Explicitly documented only when exceptions are intentionally raised.
- **Module Headers**: Every file begins with Purpose, How It Is Run, Inputs/Outputs, and Pipeline Placement.

---

## How to Contribute

1. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/your-improvement-name
   ```
2. **Make Changes**:
   - Write clear, concise code adhering to existing style patterns.
   - Preserve all Arabic translation text in `data/plant_disease_data.xlsx`.
3. **Verify Zero Regressions**:
   - Compile all modified files: `python -m py_compile <file>`.
   - Run the test suite: `pytest -v`.
4. **Submit a Pull Request**:
   - Describe what changed and why. Ensure GitHub Actions CI passes.

---

## Next Steps

- Review the technical terminology in [Glossary](glossary.md).
- Follow a single image frame through the pipeline in [How It Works](how-it-works.md).
- Review all documents from the [Documentation Index](README.md).
