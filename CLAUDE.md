# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Python project managed via Visual Studio (`.sln`/`.pyproj`). The virtual environment lives in `PlotResults/.venv/`.

To install dependencies:
```bash
cd PlotResults
.venv/Scripts/pip install -r requirements.txt
```

To run the script directly:
```bash
cd PlotResults
.venv/Scripts/python PlotResults.py
```

To launch Jupyter for the notebook:
```bash
cd PlotResults
.venv/Scripts/jupyter notebook PlotResults.ipynb
```

## Architecture

This is a research tool for analyzing MPI I/O performance benchmark results from HPC experiments (AWS and SDumont supercomputer). There are no tests.

### Core class: `MyPlot` (`MyPlot.py`)

Single class that handles the full pipeline:

1. **Load** — `load_data()` reads raw CSV log files from a directory tree structured as `{base_directory}/{experiment_number}/`
2. **Compute** — `computerMetrics()` calculates statistics and appends a row to `../plot.csv` relative to `base_directory`
3. **Plot** — multiple `plot*()` methods read from `../plot.csv` and render matplotlib charts

### Log file formats expected by `load_data()`

Each experiment directory must contain per-rank files:
- `mpiio-{rank}.log` — I/O records; columns: `[0, stage, scenario, file, block, time_start, time_end, 7, size_bytes]`
- `sddptimer{rank:04d}.log` — simulation timers; rows with `"Simulation"` in column 0 are parsed
- `mpiio-open-{rank}.log` — file open/close times; columns `[0, 1, open_time, close_time]`

Rank numbering starts at `initial_rank = 2` and iterates `number_nodes × number_scenarios_per_node` processes.

### `TypeEvaluation` enum

Controls which records are loaded:
- `JUST_SEND` — all processes (most common in notebook)
- `JUST_COMUNICATION` / `COMUNICATION_AND_IO` — same load path currently
- `onlyRemote=True` skips the first `number_scenarios_per_nodes` ranks (local processes)

### `plot.csv` accumulation pattern

`computerMetrics()` **appends** to `plot.csv`. The plotting methods (`plotBandwidth`, `plotBlocks`, `plotExecutionTime`, `plotScenarios`) read this CSV indexed by `Nodes`. To redo a full multi-node comparison, delete `plot.csv` first and re-run `computerMetrics()` for each node configuration in order.

### Typical notebook workflow

```python
# 1. Load and compute metrics for each node count (appends to plot.csv)
p = MyPlot(path_to_node_dir, num_nodes, scenarios_per_node, total_scenarios, TypeEvaluation.JUST_SEND)
p.load_data()
p.computerMetrics()

# 2. After all configs are processed, plot using the last MyPlot instance
p1 = MyPlot(path_2nodes, 2, 16, 1024, TypeEvaluation.JUST_SEND)
p1.load_data(filter_experiment=1, filter_scenario=1024)
p1.plotBandwidth(path_2nodes, "AWS")
p1.plotBlocks(base_path, "AWS")
p1.plotExecutionTime(path_2nodes, "AWS")
```

Note: `plotBlocks` takes the **base** directory (containing `plot.csv`), while `plotBandwidth` and `plotExecutionTime` take the specific **node** subdirectory.

### Size categories

Messages are bucketed in `self.categories = [0.001, 0.128, 1, 50]` MB:
- record1: `< 0.001 MB` (~1 KB)
- record2: `0.001–0.128 MB` (~128 KB)
- record3: `0.128–1 MB`
- record4: `>= 1 MB`
