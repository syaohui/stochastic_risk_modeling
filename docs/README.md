# Documentation

This folder explains the models, the statistics and the code behind the slides. Each topic page follows the order of its deck: the question, the model and its math, the tests or estimators, what this repository's experiments show, and takeaways for a risk or validation team.

## Topic pages

| # | Topic | Deck | Page |
|---|---|---|---|
| 1 | Market risk: VaR and ES | [`01-var-es-market-risk.pdf`](../slides/decks/01-var-es-market-risk.pdf) | [topics/01-var-es-market-risk.md](topics/01-var-es-market-risk.md) |
| 2 | Backtesting VaR and ES | [`02-backtesting-var-es.pdf`](../slides/decks/02-backtesting-var-es.pdf) | [topics/02-backtesting-var-es.md](topics/02-backtesting-var-es.md) |
| 3 | Model risk and estimation risk | [`03-model-risk-estimation-risk.pdf`](../slides/decks/03-model-risk-estimation-risk.pdf) | [topics/03-model-risk-estimation-risk.md](topics/03-model-risk-estimation-risk.md) |
| 4 | Credit portfolio risk | [`04-credit-portfolio-risk.pdf`](../slides/decks/04-credit-portfolio-risk.pdf) | [topics/04-credit-portfolio-risk.md](topics/04-credit-portfolio-risk.md) |
| 5 | PD model validation | [`05-pd-model-validation.pdf`](../slides/decks/05-pd-model-validation.pdf) | [topics/05-pd-model-validation.md](topics/05-pd-model-validation.md) |
| 6 | Counterparty risk and CVA | [`06-counterparty-risk-cva.pdf`](../slides/decks/06-counterparty-risk-cva.pdf) | [topics/06-counterparty-risk-cva.md](topics/06-counterparty-risk-cva.md) |
| 7 | Monte Carlo methods | [`07-monte-carlo-methods.pdf`](../slides/decks/07-monte-carlo-methods.pdf) | [topics/07-monte-carlo-methods.md](topics/07-monte-carlo-methods.md) |
| 8 | Operational risk and EVT | [`08-operational-risk-evt.pdf`](../slides/decks/08-operational-risk-evt.pdf) | [topics/08-operational-risk-evt.md](topics/08-operational-risk-evt.md) |
| 9 | Stress testing | [`09-stress-testing.pdf`](../slides/decks/09-stress-testing.pdf) | [topics/09-stress-testing.md](topics/09-stress-testing.md) |

The overview deck, [`slides/stochastic-risk-slides.pdf`](../slides/stochastic-risk-slides.pdf), covers all of these in one pass, plus the regulatory setting in 2026, open problems and backup derivations.

## Using the code

- [api.md](api.md): every module and its main functions.
- [reproducibility.md](reproducibility.md): how numbers flow from code to slides, seeds, run times, and how to add an experiment.
- [reading_list.md](reading_list.md): the regulation and literature, grouped by question.
- [`examples/quickstart.py`](../examples/quickstart.py): one call per module, with printed values.
- [`figures/gallery/`](../figures/gallery/) and [`figures/animations/`](../figures/animations/): one showcase figure per topic and five GIF animations of the simulations.

## Conventions

- Losses are positive when money is lost; returns are fractions of value and daily unless stated.
- Every symbol is defined where it first appears on a page, with its unit.
- Simulated numbers come with their sample size. Numbers quoted from regulation or papers are as published; this repository contains no market or bank data.
- Where a result is derived here rather than taken from a source, the page says so.
- Parameters are illustrative. Nothing here is regulatory, investment or legal advice.
