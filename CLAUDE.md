# Trading Bot Development Guide

## Commands
- Install: `pipenv install`
- Run all tests: `python run_all_tests.py [--verbose]`
- Run single test: `python -m unittest tests/path/to/test_file.py`
- Run specific test class: `python -m unittest tests.path.to.test_file.TestClass`
- Run specific test method: `python -m unittest tests.path.to.test_file.TestClass.test_method`
- Generate data: `make get_data`

## Code Style
- **Imports**: Standard library → Third-party → Local modules (separated by line)
- **Types**: Use type annotations for all function parameters and returns
- **Naming**:
  - Classes: CamelCase (e.g., `PPOAgent`, `TradingEnvironment`)
  - Functions/variables: snake_case (e.g., `calculate_reward`, `batch_size`)
  - Constants: UPPER_SNAKE_CASE
- **Documentation**: Use docstrings for modules, classes, and functions
- **Error handling**: Use try/except with specific exceptions and informative messages
- **Organization**: Maintain separation of concerns between agents, environments, execution
- **Testing**: Write unit tests for components, integration tests for interactions

## Key Patterns
- Always check device compatibility for PyTorch (CPU/CUDA/MPS)
- Save/load models with comprehensive state preservation
- Use NumPy/PyTorch for numerical operations