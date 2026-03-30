# Coding Conventions

**Analysis Date:** 2026-03-29

## Naming Patterns

**Files:**
- Python modules use PascalCase with capital first letter: `MyPlot.py`, `PlotResults.py`
- Preference for descriptive names indicating functionality

**Functions:**
- Regular methods use camelCase: `random_color()`, `load_data()`, `show_config()`
- Plot/visualization methods use camelCase starting with "plot": `plotBandwidth()`, `plotScenarios()`, `plotBlocks()`, `plotExecutionTime()`, `plotScatter()`
- One method inconsistently uses PascalCase: `PlotHistogram()` (should follow camelCase pattern as `plotHistogram()`)
- Abbreviations mixed: `computerMetrics()`, `escreveCsv()` (Portuguese naming)

**Variables:**
- Instance variables initialized in `__init__` as camelCase or snake_case: `number_nodes`, `number_scenarios_per_nodes`, `bestScenario`, `worstScenario`
- Data structure variables descriptive: `df_vec`, `df_master`, `df_times`, `df_mpiOpenTimes`
- Array collections often suffixed with number: `X1`, `X2`, `X3`, `X4`, `records1`, `records2`, `records3`, `records4`
- Dictionary-based states: `colors`, `bandwidths`, `dicionarioScenarios` (Portuguese)

**Types/Enums:**
- Enum class in PascalCase: `TypeEvaluation` with values like `JUST_COMUNICATION`, `COMUNICATION_AND_IO`, `JUST_SEND`
- Note: Enum name and values use UPPER_SNAKE_CASE

## Code Style

**Formatting:**
- No detected linting or formatting configuration (no .pylintrc, .flake8, or .black config)
- Inconsistent spacing around operators and assignments: `self.limit=limit` (no spaces) vs `self.number_nodes = number_nodes` (with spaces)
- Inconsistent spacing after colons in loops and conditionals observed in code

**Linting:**
- No active linting configuration detected
- No type hints or annotations in codebase

## Import Organization

**Order:**
1. Standard library imports: `os`, `csv`, `random`, `statistics`, `enum`
2. Third-party data/visualization libraries: `pandas`, `numpy`, `matplotlib`
3. Local imports: `from MyPlot import *`

**Path Aliases:**
- No path aliases detected
- Relative imports used: `from MyPlot import *`

## Error Handling

**Patterns:**
- Minimal error handling - only one explicit error condition found in codebase
- Generic exception raising: `raise Exception("Bad configuration.")` in `load_data()` at line 72 of `MyPlot.py`
- No try/except blocks detected in production code
- File operations (CSV reads/writes) lack error handling for missing files or I/O failures
- Defensive checks: `if os.path.exists()` before reading optional files (e.g., `mpiio-open-{rank}.log`)

## Logging

**Framework:** `print()` statements only

**Patterns:**
- Direct console output via `print()` for progress tracking
- Print statements in `load_data()`: `print(f'Loading Experiment {experiment+1}')`, `print(f'Loading Processes {i+1}')`
- Extensive metric printing in `computerMetrics()` method
- CSV output status: `print("Writing CSV file...")`
- No structured logging framework (no logging module, no log levels)
- All output is informational/diagnostic

## Comments

**When to Comment:**
- Minimal comments in codebase
- Occasional inline comments for clarity on data transformations
- String comments not treated as comments: `'#initialize data structures'` at line 78 of `MyPlot.py` (anti-pattern)
- Portuguese comments used: `# as 2 contagens nao sao relevantes...` at line 144

**JSDoc/TSDoc:**
- No docstrings used except minimal class docstring: `"""description of class"""` for `MyPlot` class
- No function-level docstrings
- No type hints or documentation of parameter types/return values

## Function Design

**Size:**
- Large functions with multiple responsibilities:
  - `load_data()` spans ~69 lines (lines 67-135): data loading, filtering, categorization, time tracking
  - `computerMetrics()` spans ~130 lines (lines 141-271): metric computation and CSV output
  - `plotBlocks()` spans ~67 lines (lines 410-468): data preparation and visualization

**Parameters:**
- Constructor (`__init__`) takes 6 parameters with one default
- Methods take 2-3 parameters on average
- Parameters not type-annotated
- Default parameter values used: `typeEvaluation=TypeEvaluation.JUST_COMUNICATION`, `number_blocks=4`, `max_size_kb=0`

**Return Values:**
- Most methods perform side effects (printing, plotting, writing) rather than returning values
- No return type hints
- Methods return `None` implicitly in most cases

## Module Design

**Exports:**
- Star imports used: `from MyPlot import *` in `PlotResults.py` and notebook
- Single main class `MyPlot` serves as API
- Enum `TypeEvaluation` exported for configuration

**Barrel Files:**
- Not applicable - no barrel/index pattern
- Direct imports from single module file

## Spacing and Whitespace

**Observations:**
- Inconsistent blank lines between methods (0-1 lines)
- Inconsistent blank lines within methods
- Line continuation not wrapped consistently
- Long lines (100+ characters) not wrapped

## Language Mix

**Observation:**
- Mixed Portuguese/English naming and comments throughout:
  - Portuguese method names: `escreveCsv()` (should be `write_csv()`)
  - Portuguese comments: `# as 2 contagens nao sao relevantes porque...`
  - Portuguese variable names: `dicionarioScenarios`
  - Portuguese comments in plots: "Messagem", "Tempo envio"
- English method names dominate: `load_data()`, `random_color()`, `computerMetrics()`

---

*Convention analysis: 2026-03-29*
