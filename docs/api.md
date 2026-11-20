# API reference

The package `stochastic_risk` needs only NumPy, SciPy and Matplotlib. Install it from the repository root with `pip install -e .`. Losses are positive when money is lost. Probabilities, rates and correlations are decimals. Every function that draws random numbers takes `rng` (a seed or a `numpy.random.Generator`).

## Market risk

### `measures`: VaR, ES and FRTB liquidity horizons (topic 1)

| Function | What it does |
|---|---|
| `var_es_empirical(losses, alpha)` | VaR and ES of the empirical distribution; ES is exact for the empirical measure, including the fractional tail weight |
| `normal_var_es(alpha, mu, sigma)`, `t_var_es(alpha, nu, scale, loc)` | closed forms; `std_t_scale(nu)` gives the scale for unit variance |
| `parametric_normal(losses, alpha)`, `parametric_t(losses, alpha)` | fit, then closed forms |
| `sqrt_time(risk_1d, horizon)` | square-root-of-time scaling |
| `lh_adjusted_es(es_base, es_subsets, lh, T)` | FRTB liquidity-adjusted ES (MAR33.4); `FRTB_LIQUIDITY_HORIZONS` = 10, 20, 40, 60, 120 days |

### `volatility`: GARCH(1,1) and EWMA (topic 1)

| Function | What it does |
|---|---|
| `Garch(omega, alpha_g, beta_g, nu)` | model with $\sigma_t^2 = \omega + a r_{t-1}^2 + b\sigma_{t-1}^2$ (`alpha_g` is $a$, `beta_g` is $b$), Student $t$ innovations with `nu` degrees of freedom or normal if `None`; `.simulate(n, rng)`, `.simulate_paths(sigma2_next, horizon, n_paths, rng)` |
| `garch_filter(r, omega, alpha_g, beta_g)` | conditional variances by a linear filter; the last entry is the forecast |
| `fit_garch(r, dist)` | maximum likelihood with `dist = "normal"` or `"t"` |
| `ewma_variance(r, lam)` | RiskMetrics-style EWMA |

### `evt`: extreme value theory (topics 1 and 8)

| Function | What it does |
|---|---|
| `gpd_fit(excesses)` | maximum likelihood $(\xi, \beta)$ of a generalized Pareto distribution |
| `pot_var_es(losses, alpha, threshold_quantile, u)` | peaks-over-threshold VaR and ES; returns (VaR, ES, $\xi$, $\beta$, $N_u$) |
| `hill(losses, k)` | Hill estimate of $\xi$ from the $k$ largest losses |
| `gpd_quantile_true(alpha, xi, beta, u, p_exceed)` | exact VaR for a GPD tail above $u$ |

## Validation

### `backtest`: VaR and ES backtests, PLA (topic 2)

| Function | What it does |
|---|---|
| `kupiec_pof(x, n, p)` | likelihood ratio and $\chi^2_1$ p-value for $x$ exceptions in $n$ days |
| `christoffersen(hits, p)` | independence and conditional coverage; dict with `LR_ind`, `p_ind`, `LR_cc`, `p_cc` |
| `traffic_light(x)` | zone and multiplier for $x$ exceptions in 250 days (MAR32); `TRAFFIC_LIGHT_MULTIPLIER` holds the table |
| `zone_probabilities(p, n)` | exact P(green), P(amber), P(red) when exceptions are Binomial($n$, $p$) |
| `desk_ineligible(exc_99, exc_975)` | MAR32.19 desk eligibility rule |
| `acerbi_szekely_z2(returns, var, es, alpha)` | the $Z_2$ statistic; returns are gains positive, VaR and ES positive |
| `pla_metrics(hpl, rtpl)` | Spearman correlation, KS distance and PLA zone |

### `model_risk`: estimation risk and model uncertainty (topic 3)

| Function | What it does |
|---|---|
| `plugin_normal_exceedance(n, alpha)` | exact exceedance probability of the plug-in normal VaR |
| `predictive_normal_var(losses, alpha)` | VaR with exact coverage for i.i.d. normal losses |
| `hs_expected_exceedance(n, k)` | $(n+1-k)/(n+1)$ for the $k$-th smallest of $n$ losses |
| `bootstrap_es_interval(losses, alpha, level, n_boot, rng)` | percentile bootstrap interval for the empirical ES |
| `worst_case_tilt(v, eta)` | relative-entropy worst-case reweighting; returns ($\theta$, weights, worst-case mean) |
| `weighted_var_es(losses, weights, alpha)` | VaR and ES of a weighted sample |
| `worst_case_es(losses, alpha, eta)` | ES under the worst-case tilt of the loss |

### `pd_validation`: PD model validation (topic 5)

| Function | What it does |
|---|---|
| `auc(scores_default, scores_nondefault)`, `delong_se(...)` | Mann-Whitney AUC and its DeLong standard error (higher score = riskier) |
| `binomial_pvalue(d, n, pd)`, `jeffreys_pvalue(d, n, pd)` | one-sided calibration tests under independence |
| `vasicek_pvalue(d, n, pd, rho)` | correlation-adjusted p-value under the one-factor binomial mixture |
| `hosmer_lemeshow(d, n, pd)` | statistic and $\chi^2_G$ p-value over grades |
| `simulate_defaults(pd, n, rho, n_years, rng)` | default counts per grade and year |
| `psi(expected_counts, actual_counts)` | population stability index |

## Credit and counterparty risk

### `credit`: portfolio credit risk (topic 4)

| Function | What it does |
|---|---|
| `cond_pd(p, rho, y)` | default probability given the factor $y$ (large $y$ = bad state) |
| `vasicek_cdf(x, p, rho)`, `vasicek_quantile(q, p, rho)` | infinitely granular default-rate distribution |
| `irb_capital(pd, lgd, m)` | Basel IRB capital $K$ per unit EAD; `irb_correlation`, `irb_maturity_b` give $R$ and $b$ |
| `simulate_losses(p, c, rho, n_sims, rng, copula, nu)` | portfolio losses, Gaussian or Student $t$ copula |
| `homogeneous_loss_pmf(n, p, rho)` | exact distribution of defaults among $n$ equal names |
| `granularity_adjustment(q, p, rho, n)` | second-order correction $\Delta$ (derived here) |
| `is_shift(p, c, rho, x)`, `is_tail_probability(p, c, rho, x, n_sims, rng)` | Glasserman-Li importance sampling; returns (estimate, standard error, shift) |
| `factor_grid(n_grid, lim)` | quadrature grid over the standard normal factor |

### `ccr`: counterparty credit risk (topic 6)

| Function | What it does |
|---|---|
| `HullWhite(a, sigma, r0)` | Hull-White model on a flat curve: `p0`, `B`, `V` (the variance function $\Gamma$ of the slides), `bond(t, T, x)`, exact `simulate(times, n_paths, rng)` returning $x$ and the discount factor $D$ |
| `par_rate(hw, pay_times, delta)` | par rate of a swap that starts today |
| `swap_values(hw, times, x, pay_times, delta, K)` | payer swap values on a path grid, unit notional |
| `exposure_metrics(times, V, q)` | EE, PFE, EPE, effective EE, EEPE and EAD $= 1.4\,$EEPE |
| `payer_swaption(hw, expiry, pay_times, delta, K)` | Jamshidian closed form |
| `cva_independent(times, D, V, lgd, lam)`, `cva_wrong_way(times, D, V, lgd, lam, b, scale)` | CVA with independent default, and Hull-White (2012) wrong-way risk |

## Simulation methods, operational risk, stress

### `montecarlo` (topic 7)

`bs_call`, `call_loss_exact`, `loss_threshold` (exact 1% loss level of the test call), `nested_tail_probability(m, n, c, ...)` (outer `m`, inner `n`), `gaussian_tail_mc`, `gaussian_tail_is`, `basket_call_payoffs`, `basket_estimates(n, reps, method, ...)` with `method = "mc"` or `"sobol"`.

### `oprisk` (topic 8)

`simulate_annual_losses(lam, sampler, n_years, rng)`, `lognormal_sampler(mu, s)`, `sla_var(alpha, lam, severity_ppf)`, `discretize(severity_cdf, h, n)`, `compound_poisson_fft(lam, fx)`, `compound_poisson_panjer(lam, fx)`, `quantile_from_pmf(pmf, h, alpha)`.

### `stress` (topic 9)

| Function | What it does |
|---|---|
| `reverse_stress_point(w, Sigma, loss)` | most likely scenario $x^*$, Mahalanobis distance $d$ and $P(\text{loss} \ge \ell)$ |
| `conditional_scenario_mc(w, Sigma, loss, n, band, rng)` | Monte Carlo check of $x^*$ |
| `simulate_regimes(n_days, p_cs, p_sc, cov_calm, cov_stress, rng)` | two-state Markov-switching Gaussian returns |
| `portfolio_es(returns, w, alpha)`, `indirect_stress_es(r_current, r_stress, w, reduced, alpha)` | FRTB indirect stressed ES against the full-set reference |

## Plotting

`plotting.use_style()` loads `journal.mplstyle`; `plotting.style_3d(ax, elev, azim)` gives 3D axes light panes and print-size tick labels; `plotting.save(fig, stem, outdir)` writes a 300 dpi PNG and, if the environment variable `FIGCHECK` points to a folder with `check_figure.py`, audits the figure first.
