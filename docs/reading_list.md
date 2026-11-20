# Reading list

Every entry here is cited in the slides and was checked against a publisher, regulator, arXiv or RePEc record in November 2026. The full BibTeX is in [`slides/refs.bib`](../slides/refs.bib). Grouped by the question each source answers.

## What the rules require

- Basel Committee on Banking Supervision. **MAR32**: Internal models approach: backtesting and profit and loss attribution test requirements. Basel Framework, in force from 1 January 2023. *Traffic light, desk eligibility, PLA zones (topic 2).*
- Basel Committee on Banking Supervision. **MAR33**: Internal models approach: capital requirements calculation. Basel Framework, in force from 1 January 2023. *ES at 97.5%, liquidity horizons, stress calibration, default risk charge (topics 1 and 9).*
- Basel Committee on Banking Supervision (2005). **An Explanatory Note on the Basel II IRB Risk Weight Functions.** *The IRB formula and its one-factor foundation (topic 4).*
- Board of Governors of the Federal Reserve System (2026). **SR 26-2: Revised Guidance on Model Risk Management**, issued with the OCC and FDIC; supersedes SR 11-7 and SR 21-8. *Model definition, materiality, validation (topic 3).*
- European Central Bank (2019). **Instructions for Reporting the Validation Results of Internal Models: IRB Pillar I Models for Credit Risk.** *Jeffreys test and AUC comparison (topic 5).*

## Which risk measure

- Artzner, Delbaen, Eber and Heath (1999). Coherent measures of risk. *Mathematical Finance* 9(3), 203-228.
- Acerbi and Tasche (2002). On the coherence of expected shortfall. *Journal of Banking & Finance* 26(7), 1487-1503.
- Gneiting (2011). Making and evaluating point forecasts. *Journal of the American Statistical Association* 106(494), 746-762. *ES is not elicitable.*
- Fissler and Ziegel (2016). Higher order elicitability and Osband's principle. *Annals of Statistics* 44(4), 1680-1707. *(VaR, ES) is jointly elicitable.*

## How to estimate and backtest it

- McNeil and Frey (2000). Estimation of tail-related risk measures for heteroscedastic financial time series: an extreme value approach. *Journal of Empirical Finance* 7(3-4), 271-300. *GARCH-EVT.*
- McNeil, Frey and Embrechts (2015). *Quantitative Risk Management: Concepts, Techniques and Tools*, revised edition. Princeton University Press. *The standard textbook for most of this repository.*
- Kupiec (1995). Techniques for verifying the accuracy of risk measurement models. Finance and Economics Discussion Series 95-24, Federal Reserve Board.
- Christoffersen (1998). Evaluating interval forecasts. *International Economic Review* 39(4), 841-862.
- Acerbi and Székely (2014). Backtesting expected shortfall. *Risk*, December 2014; MSCI research paper.
- Bayer and Dimitriadis (2022). Regression-based expected shortfall backtesting. *Journal of Financial Econometrics* 20(3), 437-471.

## How wrong the model can be

- Glasserman and Xu (2014). Robust risk measurement and model risk. *Quantitative Finance* 14(1), 29-58.

## Credit portfolios and PD models

- Gordy (2003). A risk-factor model foundation for ratings-based bank capital rules. *Journal of Financial Intermediation* 12(3), 199-232.
- Glasserman and Li (2005). Importance sampling for portfolio credit risk. *Management Science* 51(11), 1643-1656.
- DeLong, DeLong and Clarke-Pearson (1988). Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics* 44(3), 837-845.

## Counterparty risk

- Brigo and Mercurio (2006). *Interest Rate Models: Theory and Practice*, second edition. Springer. *Hull-White bond formula, Jamshidian's decomposition.*
- Hull and White (2012). CVA and wrong-way risk. *Financial Analysts Journal* 68(5), 58-69.

## Simulation

- Glasserman (2003). *Monte Carlo Methods in Financial Engineering*. Springer.
- Gordy and Juneja (2010). Nested simulation in portfolio risk measurement. *Management Science* 56(10), 1833-1848.

## Operational risk

- Böcker and Klüppelberg (2005). Operational VaR: a closed-form approximation. *Risk*, 90-93.

## Generative scenarios

- Cont, Cucuringu, Xu and Zhang. Tail-GAN: learning to simulate tail risk scenarios. arXiv:2203.01664, version 4 (May 2025).
- Chen, Xu, Xu and Zhang. Diffusion factor models: generating high-dimensional returns with factor structure. arXiv:2504.06566, version 5 (January 2026).
