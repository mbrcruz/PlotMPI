# Testing Patterns

**Analysis Date:** 2026-03-29

## Test Framework

**Status:** No testing framework detected

**Overview:**
- No test files found in codebase (no `*test*.py` or `*spec*.py` files)
- No test configuration files: `pytest.ini`, `setup.cfg`, `tox.ini`
- No test runner dependencies in `requirements.txt`: pytest, unittest, nose, etc.
- No test fixtures or mocking libraries detected

**Recommendation for Implementation:**
- Project uses pandas, numpy, matplotlib - would benefit from pytest for data transformation testing
- Consider pytest as primary framework (modern, widely used for Python data science)

## Manual Testing Approach

**Current Approach:**
- Interactive testing via Jupyter notebook: `PlotResults.ipynb`
- Direct script execution: `PlotResults.py`
- Manual verification of outputs through console prints and visual plots

**Notebook Testing Pattern:**
The notebook at `PlotResults.ipynb` serves as test/demo document with cells executing:
1. Initialization and configuration setup
2. Data loading for different node configurations (2, 4, 8, 16, 32 nodes)
3. Metric computation via `computerMetrics()`
4. Visualization generation via plot methods

Example from notebook:
```python
p =  MyPlot(os.path.join(path_dados_aws, "2-nodes"),
            2 , 16, 1024 ,TypeEvaluation.JUST_SEND)
p.show_config()
p.load_data()
p.computerMetrics()
```

## Testing Gaps

**Critical Areas Untested:**
1. **Data Loading (`load_data()`)** - No validation that:
   - CSV files are correctly parsed
   - Filtering logic (by experiment/scenario/size) works correctly
   - Category assignments (records1-4 by size) are accurate
   - Edge cases: missing files, malformed data, empty datasets

2. **Data Transformations (`computerMetrics()`)** - No validation that:
   - Aggregations (groupby) produce correct results
   - Statistical calculations (mean, std dev) are accurate
   - Bandwidth calculations use correct formulas
   - Categories and thresholds correctly separate data

3. **File I/O (`escreveCsv()`)** - No tests for:
   - CSV header writing on first write
   - Append mode on subsequent writes
   - CSV format correctness
   - File path resolution errors

4. **Visualization Methods** - No validation that:
   - Plots render without errors for various data shapes
   - File loading for CSV data succeeds
   - Edge cases (empty data, single record) handled

5. **Enum Configuration (`TypeEvaluation`)** - No validation:
   - Only one enum type, used minimally
   - No enforcement of type checking

## Error Handling Observations

**Current State:**
- `load_data()` method contains only one explicit error check:
  ```python
  if ( not self.typeEvaluation == TypeEvaluation.JUST_COMUNICATION
      and not self.typeEvaluation == TypeEvaluation.COMUNICATION_AND_IO
      and not self.typeEvaluation == TypeEvaluation.JUST_SEND):
          raise Exception("Bad configuration.")
  ```

**Missing Error Handling:**
- No try/except blocks around:
  - `pd.read_csv()` calls - missing files not caught
  - File path operations in `escreveCsv()`
  - Index/key access on DataFrames
  - Statistical operations on empty datasets

- Defensive file existence check exists:
  ```python
  if os.path.exists(os.path.join(self.base_directory , str(experiment+1), f"mpiio-open-{rank}.log")):
      self.df_mpiOpenTimes= pd.read_csv(...)
  ```
  But other files lack similar checks.

## Suggested Testing Structure

**If pytest implemented, recommended pattern:**

```python
# tests/test_myplot.py
import pytest
import pandas as pd
import numpy as np
from PlotResults.MyPlot import MyPlot, TypeEvaluation
import tempfile
import os

class TestDataLoading:
    """Test data loading and filtering functionality"""

    def test_load_data_with_valid_directory(self):
        """Verify load_data processes CSV files correctly"""
        pass

    def test_load_data_filters_by_scenario(self):
        """Test filter_scenario parameter works"""
        pass

    def test_load_data_categorizes_records_by_size(self):
        """Test records1-4 separated correctly by size thresholds"""
        pass

class TestMetricsComputation:
    """Test metric calculations"""

    def test_bandwidth_calculation(self):
        """Verify bandwidth computed as (sizeBytes * 8 / 1e9) / timeSec"""
        pass

    def test_statistics_with_empty_records(self):
        """Test mean/stdev handle empty record sets"""
        pass

class TestCSVOutput:
    """Test CSV file generation"""

    def test_csv_header_on_first_write(self):
        """Verify header written once on first call"""
        pass

    def test_csv_append_on_subsequent_writes(self):
        """Verify rows appended on subsequent calls"""
        pass

class TestVisualization:
    """Test plotting functions"""

    def test_plot_with_empty_dataframe(self):
        """Verify plots handle empty data gracefully"""
        pass
```

## Requirements for Testing

**If testing framework added, add to `requirements.txt`:**
```
pytest==7.x.x          # Test runner
pytest-cov==4.x.x      # Coverage reporting
pandas==2.3.1          # Already present
numpy==2.3.1           # Already present
matplotlib==3.10.3     # Already present
```

## Current Test Execution

**Manual Testing Method:**
1. Run notebook cells sequentially: `PlotResults.ipynb`
2. Verify console output shows expected metrics
3. Visually inspect generated plots
4. Check CSV output file existence and format

**Limitations:**
- No automated regression detection
- Manual verification tedious and error-prone
- No CI/CD integration
- Difficult to test edge cases systematically

## Data-Driven Testing Considerations

**For HPC benchmark data validation:**
- Use fixture data: representative MPI I/O benchmark logs
- Test with controlled dataset sizes (small, medium, large)
- Validate statistical correctness with known test data
- Consider parameterized tests for different `TypeEvaluation` enum values

---

*Testing analysis: 2026-03-29*
