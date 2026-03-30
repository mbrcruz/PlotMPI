# INTEGRATIONS.md — External Services & Integrations

## Summary

This is a fully **offline, local-filesystem-only** tool. There are no external APIs, network calls, cloud services, databases, or authentication systems.

## Data Inputs (Filesystem)

All inputs are CSV log files produced by MPI I/O benchmark runs, organized as:

```
{base_directory}/
  {experiment_number}/          # 1-based, e.g. 1/, 2/, ... 10/
    mpiio-{rank}.log            # Per-process I/O timing records
    sddptimer{rank:04d}.log     # Per-process simulation timer records
    mpiio-open-{rank}.log       # Per-process MPI file open/close times
```

### `mpiio-{rank}.log` columns (0-indexed)

| Index | Field | Type | Notes |
|-------|-------|------|-------|
| 0 | (unused) | — | — |
| 1 | stage | int | Processing stage |
| 2 | scenario | int | Scenario number (1-based) |
| 3 | file | int | File identifier |
| 4 | block | int | Block number |
| 5 | time_start | float | Epoch timestamp (seconds) |
| 6 | time_end | float | Epoch timestamp (seconds) |
| 7 | (unused) | — | — |
| 8 | size_bytes | int | Buffer size in bytes |

### `sddptimer{rank:04d}.log` columns

Rows where column 0 == `"Simulation"` are parsed; column 1 is the simulation duration (float seconds).

### `mpiio-open-{rank}.log` columns (0-indexed)

| Index | Field |
|-------|-------|
| 0-1 | (unused) |
| 2 | open_time (epoch seconds) |
| 3 | close_time (epoch seconds) |

## Data Output (Filesystem)

| File | Path | Format | Notes |
|------|------|--------|-------|
| `plot.csv` | `{base_directory}/../plot.csv` | CSV, append mode | Aggregated metrics per node configuration; must be deleted before fresh multi-node runs |

**Note:** `plotBlocks()` reads from `{base_directory}/plot.csv` (no `../`), which is inconsistent with how `escreveCsv()` writes it. See CONCERNS.md.

## No External Integrations

| Category | Status |
|----------|--------|
| REST APIs | None |
| Databases | None |
| Auth providers | None |
| Cloud storage | None |
| Message queues | None |
| Webhooks | None |
| Environment variables | None |

---
*Mapped: 2026-03-29*
