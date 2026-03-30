# CONCERNS.md — Technical Debt & Issues

## Tech Debt

| Issue | Location | Severity | Notes |
|-------|----------|----------|-------|
| God object design | `MyPlot.py:21` — `MyPlot` class | High | Single class handles loading, metrics, plotting, CSV output |
| Dead/commented code | `MyPlot.py:343-379` — `plotBandwidth` | Medium | Large dead block after `return` statement |
| Unused attributes | `MyPlot.__init__` | Low | `X1`-`X4`, `df_master`, `df_vec` initialized but unused |
| Hardcoded values | `MyPlot.__init__:47` | Medium | `categories=[0.001,0.128,1,50]` not configurable |
| Inconsistent data types | `records` list | Medium | Mixes raw dicts then converts to DataFrame; no schema |
| Missing error handling | `load_data` | High | No try/except on file reads; missing files crash silently |
| Platform-specific paths | `PlotResults.py:6`, notebook cells | High | Hardcoded Windows absolute paths (`D:\Marcelo\...`) |
| Hardcoded rank offset | `load_data:85` | Medium | `initial_rank = 2` is a magic constant with no explanation |

## Known Bugs

| Bug | Location | Impact |
|-----|----------|--------|
| Enum typo | `TypeEvaluation.JUST_COMUNICATION` | Low — misspelled but functional |
| DataFrame assignment logic | `computerMetrics:146-153` | Medium — `df1=df` then immediately reassigned; wasted assignments |
| Division by zero risk | `computerMetrics` — `stdev` with single experiment | Low — `std()` returns NaN, not crash |
| Inconsistent CSV paths | `escreveCsv` uses `"../plot.csv"`, `plotBlocks` uses `"plot.csv"` (no `../`) | High — different methods expect CSV in different locations |
| Redundant `start_moment` init | `__init__:54,58` — set to 0 then immediately to `1000000000000000` | Low |

## Security Risks

| Risk | Details |
|------|---------|
| Personal paths in notebook | `PlotResults.ipynb` contains absolute paths with personal directory structure (`D:\Marcelo\PSR Dropbox\Marcelo Barros\...`) — leaked if repo is public |
| Non-configurable output | CSV written relative to input directory with no override — could overwrite data in unexpected locations |

## Performance Bottlenecks

| Bottleneck | Location | Details |
|-----------|----------|---------|
| Unbounded memory growth | `load_data` | All records accumulated in `self.records` list (millions of rows) |
| Multiple DataFrame reconstructions | `computerMetrics` | `pd.DataFrame(self.records)` called; then `records1-4` each reconstructed separately |
| Multiple CSV passes | `plotBandwidth`, `plotBlocks`, etc. | Each plot method re-reads `plot.csv` independently |
| Matplotlib memory leaks | All `plot*` methods | No `plt.close()` calls between plots |

## Fragile Areas

| Area | Details |
|------|---------|
| Magic column indices | `load_data` — `df.iloc[k,1]` through `df.iloc[k,8]`; no column names, breaks on format change |
| Implicit state accumulation | `records`, `mpiOpenTimes`, `Simulations` accumulate across `load_data` calls; no reset |
| Metrics → CSV → Plot coupling | `computerMetrics` must run before any plot method; not enforced programmatically |
| `number_experiments` hardcoded to 10 in CSV stats | `computerMetrics:191` — divides count by 10 regardless of actual experiment count |

## Scaling Limits

| Limit | Details |
|-------|---------|
| Memory at 1M+ records | Single `MyPlot` instance for 1024 scenarios × 10 experiments × 32 nodes loads ~15M records into RAM |
| Sequential file I/O | `load_data` reads files one at a time in a nested loop |
| No streaming or chunked processing | Entire dataset must fit in memory before any metric can be computed |

## Dependencies at Risk

| Package | Pin | Risk |
|---------|-----|------|
| `numpy==2.3.1` | Hard-pinned | NumPy 2.x has breaking API changes vs 1.x |
| `pandas==2.3.1` | Hard-pinned | Recent version; behavior changes in groupby/agg |
| `matplotlib==3.10.3` | Hard-pinned | Fine but locked |
| Python version | Unspecified | No `python_requires` anywhere |

## Missing Infrastructure

- **Zero tests** — no unit, integration, or regression tests
- **No logging** — `print()` used throughout; no log levels or file output
- **No configuration** — all parameters passed manually at construction/call time
- **No data validation** — CSV column counts/types assumed; corrupt files crash the process
- **No output versioning** — `plot.csv` is always overwritten/appended with no timestamp or run ID

---
*Mapped: 2026-03-29*
