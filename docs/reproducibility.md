# Reproducibility

## How a number gets onto a slide

1. Each experiment script in `experiments/` (one per figure) has two steps. `compute()` runs the simulation and saves raw arrays to `results/raw/eNN_name.npz`. `plot()` reads them, draws the figure and calls `write_numbers`.
2. `write_numbers` writes `results/eNN_name.json` and `results/eNN_name.tex`. The `.tex` file defines one LaTeX macro per number, for example `\EThreeAmberOne` (the share of amber outcomes for a correct model).
3. `slides/preamble.tex` inputs every `results/*.tex`, and the slides use the macros. No simulated number on a slide is typed by hand. Typed numbers are model inputs, regulatory constants and numbers quoted from papers.

Change the code, rerun the experiment, rebuild the slides, and the text follows.

## Commands

```bash
pip install -e ".[test]"
```

```bash
python -m pytest tests
```

```bash
python experiments/run_all.py --compute --jobs 8
```

```bash
python experiments/run_all.py --plot
```

```bash
python build_slides.py
```

- **Tests.** 24 checks in `tests/test_risk.py`, about 20 seconds. Each compares a closed form or estimator with an independent result: Monte Carlo, an exact distribution, a second algorithm, or a published value.
- **Compute.** 19 experiments with fixed seeds. About half an hour of CPU time in total; `--jobs N` runs them in `N` parallel processes. The numbers in this repository came from one run on a 16-core Linux node (Python 3.12, NumPy 2.5, SciPy 1.18), which took eight minutes. `python experiments/run_all.py --compute e14` reruns one experiment.
- **Plot.** About fifteen seconds for the 19 experiments and the gallery. The raw arrays are in the repository, so this step works without the compute step.
- **Slides.** `build_slides.py` compiles the overview deck to `slides/stochastic-risk-slides.pdf` and the nine deep dives to `slides/decks/`. It needs LaTeX with `latexmk` and the Metropolis Beamer theme. `python build_slides.py t5` builds one deck; `--keep` keeps the LaTeX auxiliary files.

Results are reproducible for a given NumPy version. A different NumPy or SciPy version can change random streams or optimizer paths and move the last digit of some numbers; the conclusions do not depend on it.

## Gallery and animations

- `experiments/gallery.py` makes one showcase figure per topic in `figures/gallery/` and writes its numbers to `results/gallery.{json,tex}`. It runs with the experiments (`run_all.py` includes it) and its simulation takes about ten seconds.
- `experiments/animations.py` renders five GIFs to `figures/animations/` (20 to 24 frames per second; 1080 pixels wide, the rotating 3D view 750). Frames are drawn at 150 dpi and encoded with one shared 255-colour palette and no dithering, so lines and text stay sharp. Rendering takes several minutes of CPU time; the GIFs in this repository were rendered on a Linux cluster node (job of 20 November 2026), with Liberation Serif standing in for Times New Roman. `python experiments/animations.py a2` renders one, and `--poster DIR` saves the last frame of each as a PNG for the figure audit, and `--sample DIR` writes 12 evenly spaced frames of each as a quick test.

## Figures

All figures are PNG at 300 dpi, made with the journal style in `src/stochastic_risk/journal.mplstyle`: inward ticks on all four sides, Title Case axis labels with units where the quantity has one, type sized for print, and legends clear of data. Set the environment variable `FIGCHECK` to a folder containing a `check_figure.py` audit script, and every `save()` call runs the audit before writing.

## Statistics conventions

- Simulated rejection rates, coverages and probabilities come with their sample size (number of repetitions) on the slide that reports them.
- Agreement between theory and simulation is reported in standard errors or as a relative error.
- Where an exact answer exists (binomial zone probabilities, the binomial mixture, Jamshidian prices, the FFT compound distribution), the experiments use it instead of simulation.

## Adding a topic

1. A module in `src/stochastic_risk/` with docstrings that state the model, its symbols and their units.
2. A test in `tests/` against an independent result.
3. An experiment `experiments/eNN_name.py` with `compute()` and `plot()`, using `save_raw`, `load_raw` and `write_numbers("eNN_name", {...})` from `common.py` with letters-only macro names. Add it to `EXPERIMENTS` in `experiments/run_all.py`.
4. An `\input` line in `slides/preamble.tex`, then slides in `slides/parts/` or a new deck in `slides/topics/` (copy an existing deck's header and add it to `TOPIC_DECKS` in `build_slides.py`).
5. A page in `docs/topics/`.
