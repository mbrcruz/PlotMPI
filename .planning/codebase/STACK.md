# STACK.md — Technology Stack

## Runtime

| Component | Value |
|-----------|-------|
| Language | Python 3.13.1 |
| Environment | Virtual environment at `PlotResults/.venv/` |
| OS (dev) | Windows 11 Pro |
| IDE | Visual Studio (`.sln` / `.pyproj` project files) |
| Notebook | Jupyter (`PlotResults/PlotResults.ipynb`) |

## Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pandas` | 2.3.1 | DataFrame operations, CSV I/O, groupby aggregations |
| `numpy` | 2.3.1 | Numerical arrays for plot data |
| `matplotlib` | 3.10.3 | All chart rendering (bar, scatter, histogram) |

## Jupyter Infrastructure

| Package | Version | Purpose |
|---------|---------|---------|
| `ipykernel` | 6.29.5 | Jupyter kernel |
| `ipython` | 9.4.0 | Interactive shell |
| `jupyter_client` | 8.6.3 | Client/server communication |
| `jupyter_core` | 5.8.1 | Core Jupyter framework |
| `tornado` | 6.5.1 | Async web server for notebooks |
| `pyzmq` | 27.0.0 | ZeroMQ messaging |

## Supporting Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `debugpy` | 1.8.14 | VS Code/Visual Studio debugger |
| `pillow` | 11.3.0 | Image handling (matplotlib backend) |
| `pywin32` | 310 | Windows API bindings |
| `contourpy` | 1.3.2 | Matplotlib contour rendering |
| `fonttools` | 4.58.5 | Font handling for plots |
| `python-dateutil` | 2.9.0.post0 | Date parsing |
| `pytz` / `tzdata` | 2025.2 | Timezone data |

## Configuration

- **No config files** — all parameters passed at runtime via Python constructor and method arguments
- **No environment variables** — fully self-contained
- **Dependency pinning** — all 40+ packages hard-pinned in `PlotResults/requirements.txt`
- **No `pyproject.toml` or `setup.py`** — not a distributable package

## Install

```bash
cd PlotResults
.venv/Scripts/pip install -r requirements.txt
```

## Key Source Files

| File | Role |
|------|------|
| `PlotResults/MyPlot.py` | Core library — `MyPlot` class and `TypeEvaluation` enum |
| `PlotResults/PlotResults.py` | Script entrypoint — instantiates `MyPlot` with a specific dataset |
| `PlotResults/PlotResults.ipynb` | Interactive notebook for multi-node analysis runs |
| `PlotResults/requirements.txt` | Pinned dependency list |
| `PlotResults/PlotResults.pyproj` | Visual Studio project file |
| `PlotResults/PlotResults.sln` | Visual Studio solution file |

---
*Mapped: 2026-03-29*
