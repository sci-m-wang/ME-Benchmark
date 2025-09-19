# ME-Benchmark Tests

This directory contains tests for the ME-Benchmark framework.

## Running Tests

To run the basic tests:

```bash
python tests/test_basic.py
```

## Test Coverage

The current tests cover:

1. **Component Registry**: Verifies that all built-in components are properly registered
2. **Configuration Utilities**: Tests saving and loading of configuration files
3. **Results Collector**: Tests the results collection functionality

## Adding New Tests

To add new tests:

1. Create a new test file in this directory
2. Follow the pattern in `test_basic.py`
3. Run the tests to ensure they pass

## Continuous Integration

For CI/CD integration, you can run all tests with:

```bash
python -m pytest tests/
```