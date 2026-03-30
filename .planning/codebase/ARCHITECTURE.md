# Architecture

**Analysis Date:** 2026-03-29

## Pattern Overview

**Overall:** Single-class analytical pipeline for HPC I/O benchmark processing

**Key Characteristics:**
- Monolithic class design (`MyPlot`) that handles entire data pipeline
- Stateful object accumulation pattern for log processing
- CSV aggregation layer for cross-experiment comparisons
- Jupyter notebook-driven workflow with manual execution steps

## Layers

**Data Loading Layer:**
- Purpose: Parse raw MPI-IO and simulation timing logs from HPC benchmark experiments
- Location: `MyPlot.load_data()` in `/d/Marcelo/Projetos/Python/PlotResults/PlotResults/MyPlot.py` (lines 67-135)
- Contains: CSV file reading, record extraction, categorization by message size
- Depends on: pandas, os, numpy for file I/O and data structures
- Used by: `computerMetrics()` for statistics computation

**Metrics Computation Layer:**
- Purpose: Calculate aggregate statistics (bandwidth, latency, throughput) across experiments and scenarios
- Location: `MyPlot.computerMetrics()` in `MyPlot.py` (lines 141-271)
- Contains: pandas groupby operations, CSV aggregation, CSV file writing
- Depends on: pandas, statistics, csv modules; output from `load_data()`
- Used by: Plotting layer for rendering charts

**Visualization Layer:**
- Purpose: Render matplotlib charts comparing performance across node configurations
- Location: Multiple `plot*()` methods in `MyPlot.py` (lines 296-568)
  - `plotBandwidth()` (lines 296-343)
  - `plotBlocks()` (lines 410-467)
  - `plotExecutionTime()` (lines 471-504)
  - `plotScenarios()` (lines 382-407)
  - `PlotHistogram()` (lines 506-534)
  - `plotScatter()` (lines 536-568)
- Contains: matplotlib figure creation, axis configuration, legend management
- Depends on: pandas, matplotlib, numpy; reads from `../plot.csv`
- Used by: Jupyter notebook for interactive exploration

## Data Flow

**Single Experiment Analysis (per node configuration):**

1. **Initialization** (`__init__`)
   - Caller provides: base directory, node count, scenarios per node, total scenarios, evaluation type
   - Initializes state dictionaries: `self.X1-X4` (by scenario), `self.records` (all I/O), `self.records1-4` (by size category)

2. **Data Loading** (`load_data()`)
   - For each experiment subdirectory: `{base_directory}/1/`, `{base_directory}/2/`, etc.
   - For each rank (starting from initial_rank=2):
     - Read `mpiio-{rank}.log`: extract I/O operation records (stage, scenario, file, block, duration in seconds, size in bytes)
     - Read `sddptimer{rank:04d}.log`: extract simulation wall-time
     - Read `mpiio-open-{rank}.log` (optional): extract file open/close times
   - Accumulate in `self.records` list and categorize into `self.records1-4` by message size
   - Track minimum start timestamp across all processes

3. **Metrics Computation** (`computerMetrics()`)
   - Convert records lists → pandas DataFrames
   - Group by (experiment, scenario) → sum time and size per scenario
   - Calculate: mean/stdev across scenarios, min/max scenario, per-process averages
   - Categorize statistics by size bucket (records1-4)
   - Calculate bandwidth: (total_bytes × 8 bits/byte) / (total_time × 1e9) → Gb/s
   - Write single CSV row to `../plot.csv` with ~25 metric columns

4. **Multi-Configuration Plotting** (repeated after all `computerMetrics()` calls)
   - Load accumulated `../plot.csv` indexed by node count
   - Create comparison plots: bandwidth vs. node count, latency by message size, execution time composition

**State Management:**
- `self.records`: Master list of all I/O records across all ranks/experiments
- `self.records1-4`: Filtered subsets by message size category
- `self.colors`: Cached hex colors for reproducible plot coloring
- `self.start_moment`: Global minimum timestamp (unused in calculations)
- `self.bandwidths`, `self.dicionarioScenarios`: Declared but unused
- `self.df_vec`, `self.df_master`: Declared but never populated

## Key Abstractions

**TypeEvaluation Enum:**
- Purpose: Control whether to include all processes or filter by locality
- Location: `MyPlot.py` lines 14-18
- Variants:
  - `JUST_SEND` (value 3): Load all processes — most common in notebook usage
  - `JUST_COMUNICATION` (value 1): Alternative control flag (unused distinction from COMUNICATION_AND_IO)
  - `COMUNICATION_AND_IO` (value 2): Alternative control flag (same load behavior as JUST_COMUNICATION)
- Pattern: Enum checked in `load_data()` (line 69-72); `onlyRemote` flag separately filters first N processes

**Message Size Categories:**
- Purpose: Bucket I/O records for separate statistical analysis
- Location: `self.categories = [0.001, 0.128, 1, 50]` MB
- Boundaries:
  - record1: < 0.001 MB (< ~1 KB)
  - record2: 0.001–0.128 MB (< ~128 KB)
  - record3: 0.128–1 MB
  - record4: ≥ 1 MB
- Pattern: Applied in `load_data()` (lines 114-121) during record accumulation

**CSV Aggregation Pattern:**
- Purpose: Accumulate metrics across multiple node configurations for cross-comparison
- Location: `escreveCsv()` in `MyPlot.py` (lines 273-294)
- Behavior: Append one row per `computerMetrics()` call; indexed by `Nodes` column in read operations
- Assumption: Rows written in increasing node-count order for logical plotting

## Entry Points

**Direct Python Script (`PlotResults.py`):**
- Location: `/d/Marcelo/Projetos/Python/PlotResults/PlotResults/PlotResults.py`
- Triggers: `python PlotResults.py`
- Responsibilities:
  - Import `MyPlot` class
  - Instantiate with hardcoded path to benchmark results
  - Call `show_config()`, `load_data()`, `computerMetrics()`, optionally `PlotHistogram()`
- Limitations: Single hardcoded configuration; requires manual modification for different experiments

**Jupyter Notebook (`PlotResults.ipynb`):**
- Location: `/d/Marcelo/Projetos/Python/PlotResults/PlotResults/PlotResults.ipynb`
- Triggers: `jupyter notebook PlotResults.ipynb`
- Responsibilities:
  - Interactive execution of multiple MyPlot instantiations for different node counts (2, 4, 8, 16, 32 nodes)
  - Sequential: load_data() → computerMetrics() for each config
  - Then: load_data() with filters (experiment, scenario) → multiple plotting calls
  - Supports manual reordering and filtering of analysis steps
- Pattern: Cells separated by markdown headings for AWS vs. SDumont experiments

## Error Handling

**Strategy:** Minimal — relies on pandas and matplotlib raising native exceptions

**Patterns:**
- Configuration validation: `load_data()` checks `TypeEvaluation` enum value (lines 69-72), raises generic `Exception("Bad configuration.")`
- File I/O: No explicit error handling; missing logs cause `FileNotFoundError` from pandas `read_csv()`
- Missing optional files: Conditional check for `mpiio-open-{rank}.log` (line 127); skips if absent
- CSV header writing: Conditional logic checks file existence and size (line 288)

## Cross-Cutting Concerns

**Logging:** No structured logging — uses `print()` statements
- Data load progress: `print(f'Loading Experiment {experiment+1}')` (line 77)
- Process-level detail: `print(f'Loading Processes {i+1}')` conditional (line 90)
- Metric summaries: Printed directly to stdout (lines 176-253)
- CSV writes: `print("Writing CSV file...")` (line 257)

**Validation:** Limited
- TypeEvaluation enum membership check (lines 69-72)
- DataFrame non-empty checks before statistics (lines 191, 199, 207, 214)
- No schema validation on input CSV columns

**Authentication:** Not applicable (file-system only)

---

*Architecture analysis: 2026-03-29*
