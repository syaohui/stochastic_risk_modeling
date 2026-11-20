# Contributing

Corrections, extensions and new topics are welcome. This is a research and teaching repository, so the bar is that every claim can be checked.

## Ground rules

1. **Every model comes with an independent check.** A closed form is tested against Monte Carlo, an exact distribution, a second algorithm, or a published value. An estimator or test is run on simulated data where the truth is known.
2. **Every simulated number states its sample size,** and agreement with theory is reported in standard errors or as a relative error.
3. **Units first.** Docstrings state every symbol and its unit. Code names that differ from the slide symbols are mapped in the module docstring or in [docs/api.md](docs/api.md).
4. **No hand-typed results.** Numbers on slides come from `results/*.tex`, written by the experiment scripts.
5. **Regulation and published facts are cited and quoted as published.** If a number cannot be traced to a source, it does not go in.
6. **No bank or market data in the repository.** Code that reads data is welcome; the data stay with their owner.

## Workflow

```bash
pip install -e ".[test]"
```

```bash
python -m pytest tests
```

For a new figure, add `experiments/eNN_name.py` with `compute()` and `plot()` and list it in `experiments/run_all.py`. For new slides, add the `\input` line for its results to `slides/preamble.tex` and run `python build_slides.py`. See [docs/reproducibility.md](docs/reproducibility.md) for the full checklist.

## Style

- Figures: PNG at 300 dpi through `plotting.save`, with the journal style (inward ticks on four sides, Title Case axis labels with units, legends clear of data).
- Prose: plain words and short sentences; define each symbol where it first appears; no dashes as clause connectors.
- Python: NumPy and SciPy only in the package; docstrings on public functions; fixed seeds in experiments.

## Reporting a problem

Open an issue with the file and line, what you expected, and what you got. For a disputed number, say which experiment produced it and which independent check disagrees.
