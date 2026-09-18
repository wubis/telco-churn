# Telco churn: Bayesian hierarchical logistic regression

This project estimates associations between customer characteristics and churn in the public IBM Telco Customer Churn sample. It compares a Bayesian logistic model with contract-specific, partially pooled intercepts against ordinary logistic regression with contract fixed effects. The Metropolis-Hastings sampler was custom built for understanding; the portfolio validation adds a held-out comparison and stronger chain diagnostics.

## Question and scope

The outcome is the dataset's `Churn` flag. Predictors include tenure, monthly charges, demographics, internet service, payment method, online security, and tech support. The intended prediction point is a **current customer snapshot**, when these fields are already known. The data are observational and have no longitudinal prediction timestamp, so coefficient associations are not causal effects and the held-out split cannot establish future-period performance.

## Reproduce the validation

Use Python 3.11 or newer. From the repository root:

```bash
python -m pip install -r requirements.txt
python analysis/validate.py
```

The script reads `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`, removes the 11 records with blank `TotalCharges`, and uses a stratified 80/20 split (`random_state=2026`). Continuous predictors are standardized using **training-set means and standard deviations only**. It fits both models on training customers and scores the same untouched test customers. Four Bayesian chains use the same model priors as the course analysis. The script saves metrics, customer-level predictions, calibration deciles, replicated-outcome checks, and per-parameter chain diagnostics under `results/`.

The Bayesian sampler interweaves non-centered Metropolis updates with centered conditional updates for the contract hierarchy. Its fixed-effect proposal is preconditioned using the frequentist model's **training-set** information matrix. This changes computation, not the statistical model. The diagnostic CSV reports rank-normalized split R-hat and approximate chain-aware bulk/tail ESS; inspect those values before using posterior intervals.

## Results

On 1,407 held-out customers, the Bayesian model reached **0.839 ROC-AUC, 0.654 average precision, 0.138 Brier score, and 0.424 log loss**. Ordinary logistic regression reached effectively the same scores. A constant training-prevalence prediction had 0.195 Brier score and 0.579 log loss. See [`results/heldout_metrics.csv`](results/heldout_metrics.csv) for exact values.

All parameters in the primary Bayesian fit had rank-normalized split R-hat below 1.009; the lowest approximate bulk ESS was 366 for a contract intercept. Simulated test-set churn rates covered the observed overall and contract-specific rates, though some calibration deciles deviate from the diagonal. Changing the contract-scale prior from Half-Normal(2.5) to Half-Normal(1.0) moved contract-intercept means by less than 0.01 but changed the posterior mean of the scale itself from 1.262 to 0.871. The [`results/mcmc_diagnostics.csv`](results/mcmc_diagnostics.csv), [`results/posterior_predictive_checks.csv`](results/posterior_predictive_checks.csv), and [`results/prior_sensitivity.csv`](results/prior_sensitivity.csv) contain the detailed audit.

The Bayesian model is presented as an interpretable uncertainty model, not a demonstrated accuracy improvement. Its contract-intercept means were within 0.02 log-odds units of the frequentist estimates, so this data set does not demonstrate a meaningful practical benefit from partial pooling.

## Files

| File | Purpose |
| --- | --- |
| `analysis/validate.py` | Reproducible held-out evaluation, sampler, diagnostics, and posterior predictive checks |
| `results/` | Outputs from the validation run |
| `telco_hierarchical_logistic_mh_notebook.ipynb` | Original full-sample course analysis |
| `frequentist_model.ipynb` | Original logistic baseline notebook |
| `telco_eda.ipynb` | Exploratory analysis |
| `instructions_and_requirements/neurips_format.tex` | Report source |
| `jwang_final.pdf` | Updated report |

The original notebook's in-sample metrics and lightweight ESS/R-hat are historical course outputs. Use the validation script and `results/` for claims about predictive performance and sampler diagnostics.

The updated report can be rebuilt from the project root with `pdflatex -jobname=jwang_final instructions_and_requirements/neurips_format.tex` (run twice to resolve references). A LaTeX installation with the standard packages listed in the source is required.

## Limitations

- This is one random held-out split of a static, fictional-company sample. It does not test temporal drift or generalization to another telecom company.
- The model specification was developed during the original full-sample course analysis. This later split is retrospective validation, so it is less independent than a test set reserved before any EDA or model selection.
- There are only three contract groups, so the hyperprior and any practical benefit from partial pooling deserve caution.
- The 0.5 classification threshold is illustrative. A retention decision would need intervention costs, benefits, and a separately chosen threshold.
- Prediction from signup would require a different feature set and an explicit future churn horizon.
