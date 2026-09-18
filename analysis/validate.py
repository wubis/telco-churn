"""Reproducible held-out validation of the telco churn models.

Run from the repository root: python analysis/validate.py
Requires numpy, pandas, scipy, scikit-learn, and numba.
"""

from pathlib import Path
import json
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "telco-mpl-cache"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm, rankdata
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    log_loss, roc_auc_score,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
SEEDS = (11, 22, 33, 44)
CONTINUOUS = ("tenure", "MonthlyCharges")


def prepare():
    df = pd.read_csv(ROOT / "data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    y = (df["Churn"] == "Yes").to_numpy(np.int64)
    x = pd.DataFrame(index=df.index)
    for col in CONTINUOUS + ("SeniorCitizen",):
        x[col] = df[col].astype(float)
    for col in ("Partner", "Dependents", "PaperlessBilling", "OnlineSecurity", "TechSupport"):
        x[col] = (df[col] == "Yes").astype(float)
    for col in ("InternetService", "PaymentMethod"):
        x = pd.concat([x, pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=float)], axis=1)
    groups = pd.Categorical(df["Contract"], categories=["Month-to-month", "One year", "Two year"])
    if np.any(groups.codes < 0):
        raise ValueError("Unexpected contract category")
    return df, x, y, groups.codes.astype(np.int64)


def standardize(x, train_idx):
    x = x.copy()
    means = x.iloc[train_idx][list(CONTINUOUS)].mean()
    sds = x.iloc[train_idx][list(CONTINUOUS)].std()
    x.loc[:, list(CONTINUOUS)] = (x[list(CONTINUOUS)] - means) / sds
    return np.ascontiguousarray(x.to_numpy(float)), means, sds


@njit
def log_posterior(x, y, groups, beta, z, mu, eta, tau_scale):
    tau = np.exp(eta)
    lp = -0.5 * np.sum((beta / 2.5) ** 2) - 0.5 * np.sum(z ** 2)
    lp += -0.5 * (mu / 5.0) ** 2 - 0.5 * (tau / tau_scale) ** 2 + eta
    for i in range(x.shape[0]):
        lin = mu + tau * z[groups[i]]
        for j in range(x.shape[1]):
            lin += x[i, j] * beta[j]
        lp += y[i] * lin - np.logaddexp(0.0, lin)
    return lp


@njit
def centered_tau_log_density(alpha, mu, eta, tau_scale):
    """Conditional log density of log(tau), holding group intercepts fixed."""
    tau = np.exp(eta)
    return -(len(alpha) - 1) * eta - 0.5 * np.sum(((alpha - mu) / tau) ** 2) - 0.5 * (tau / tau_scale) ** 2


@njit
def mh_chain(x, y, groups, beta_chol, n_iter, burn, thin, seed, tau_scale):
    np.random.seed(seed)
    p = x.shape[1]
    beta = np.random.normal(0, 0.05, p)
    z = np.random.normal(0, 0.4, 3)
    mu = np.log(y.mean() / (1 - y.mean())) + np.random.normal(0, 0.15)
    eta = np.log(0.8) + np.random.normal(0, 0.15)
    lp = log_posterior(x, y, groups, beta, z, mu, eta, tau_scale)
    steps = np.array([0.70, 0.16, 0.22, 0.22, 0.25])
    window = np.zeros(5)
    total = np.zeros(5)
    draws = np.empty(((n_iter - burn + thin - 1) // thin, p + 5))
    row = 0
    for it in range(n_iter):
        proposed_beta = beta + steps[0] * (beta_chol @ np.random.normal(0, 1, p))
        proposed_lp = log_posterior(x, y, groups, proposed_beta, z, mu, eta, tau_scale)
        if np.log(np.random.random()) < proposed_lp - lp:
            beta, lp = proposed_beta, proposed_lp
            window[0] += 1; total[0] += 1
        proposed_z = z + np.random.normal(0, steps[1], 3)
        proposed_lp = log_posterior(x, y, groups, beta, proposed_z, mu, eta, tau_scale)
        if np.log(np.random.random()) < proposed_lp - lp:
            z, lp = proposed_z, proposed_lp
            window[1] += 1; total[1] += 1
        proposed_mu = mu + np.random.normal(0, steps[2])
        proposed_lp = log_posterior(x, y, groups, beta, z, proposed_mu, eta, tau_scale)
        if np.log(np.random.random()) < proposed_lp - lp:
            mu, lp = proposed_mu, proposed_lp
            window[2] += 1; total[2] += 1
        proposed_eta = eta + np.random.normal(0, steps[3])
        proposed_lp = log_posterior(x, y, groups, beta, z, mu, proposed_eta, tau_scale)
        if np.log(np.random.random()) < proposed_lp - lp:
            eta, lp = proposed_eta, proposed_lp
            window[3] += 1; total[3] += 1
        # Interweave centered updates while holding alpha fixed. This targets the
        # hyperparameters where a purely non-centered random walk mixes slowly.
        alpha = mu + np.exp(eta) * z
        tau = np.exp(eta)
        variance_mu = 1.0 / (3.0 / (tau * tau) + 1.0 / 25.0)
        mu = variance_mu * alpha.sum() / (tau * tau) + np.sqrt(variance_mu) * np.random.normal()
        z = (alpha - mu) / tau
        proposed_eta = eta + np.random.normal(0, steps[4])
        lp_centered = centered_tau_log_density(alpha, mu, eta, tau_scale)
        proposed_lp_centered = centered_tau_log_density(alpha, mu, proposed_eta, tau_scale)
        if np.log(np.random.random()) < proposed_lp_centered - lp_centered:
            eta = proposed_eta
            z = (alpha - mu) / np.exp(eta)
            window[4] += 1; total[4] += 1
        lp = log_posterior(x, y, groups, beta, z, mu, eta, tau_scale)
        if it < burn and (it + 1) % 250 == 0:
            steps *= np.exp(0.75 * (window / 250 - 0.30))
            steps = np.clip(steps, 0.005, 3.0)
            window[:] = 0
        if it >= burn and (it - burn) % thin == 0:
            draws[row, :p] = beta
            draws[row, p:p+3] = mu + np.exp(eta) * z
            draws[row, p+3] = mu
            draws[row, p+4] = np.exp(eta)
            row += 1
    return draws, total / n_iter


def split_chains(a):
    half = a.shape[1] // 2
    return np.concatenate((a[:, :half], a[:, -half:]), axis=0)


def basic_rhat(a):
    m, n = a.shape
    w = a.var(axis=1, ddof=1).mean()
    if w == 0:
        return np.nan
    b = n * a.mean(axis=1).var(ddof=1)
    return np.sqrt(((n - 1) * w / n + b / n) / w)


def rank_rhat(a):
    a = split_chains(a)
    def transformed(v):
        ranks = rankdata(v.ravel(), method="average").reshape(v.shape)
        return norm.ppf((ranks - 0.375) / (ranks.size + 0.25))
    return max(basic_rhat(transformed(a)), basic_rhat(transformed(np.abs(a - np.median(a)))))


def ess_approx(a):
    """Multi-chain, split-chain initial-positive-sequence ESS approximation."""
    a = split_chains(a)
    m, n = a.shape
    w = a.var(axis=1, ddof=1).mean()
    b = n * a.mean(axis=1).var(ddof=1)
    var_plus = (n - 1) * w / n + b / n
    if var_plus <= 0:
        return np.nan
    centered = a - a.mean(axis=1, keepdims=True)
    size = 1 << (2 * n - 1).bit_length()
    fft = np.fft.rfft(centered, n=size, axis=1)
    acov = np.fft.irfft(fft * np.conj(fft), n=size, axis=1)[:, :n] / n
    rho = 1 - (w - acov.mean(axis=0)) / var_plus
    pair_sums = []
    for t in range(0, n - 1, 2):
        pair = rho[t] + rho[t + 1]
        if pair < 0:
            break
        pair_sums.append(pair)
    for i in range(1, len(pair_sums)):
        pair_sums[i] = min(pair_sums[i], pair_sums[i-1])
    tau = max(-1 + 2 * sum(pair_sums), 1 / np.log10(m * n))
    return min(m * n / tau, m * n * np.log10(m * n))


def diagnostics(chains, names):
    rows = []
    for j, name in enumerate(names):
        a = chains[:, :, j]
        ranks = rankdata(a.ravel()).reshape(a.shape)
        rank_normal = norm.ppf((ranks - 0.375) / (ranks.size + 0.25))
        q05, q95 = np.quantile(a, [0.05, 0.95])
        rows.append(dict(parameter=name, mean=a.mean(), sd=a.std(ddof=1),
                         q2_5=np.quantile(a, .025), q97_5=np.quantile(a, .975),
                         rhat_rank_split=rank_rhat(a), ess_bulk_approx=ess_approx(rank_normal),
                         ess_tail_approx=min(ess_approx((a <= q05).astype(float)),
                                             ess_approx((a >= q95).astype(float)))))
    return pd.DataFrame(rows)


def fit_frequentist(x_train, y_train, groups_train, x_test, groups_test):
    def design(x, g):
        return np.column_stack((np.ones(len(x)), x, (g == 1).astype(float), (g == 2).astype(float)))
    a, b = design(x_train, groups_train), design(x_test, groups_test)
    def objective(theta):
        lin = a @ theta
        return np.logaddexp(0, lin).sum() - y_train @ lin
    def gradient(theta):
        return a.T @ (expit(a @ theta) - y_train)
    opt = minimize(objective, np.zeros(a.shape[1]), jac=gradient, method="BFGS", options={"maxiter": 1000})
    if not opt.success and np.linalg.norm(gradient(opt.x)) > 1e-3:
        raise RuntimeError(f"Frequentist optimizer failed: {opt.message}")
    fitted = expit(a @ opt.x)
    fisher = a.T @ ((fitted * (1 - fitted))[:, None] * a)
    beta_cov = np.linalg.inv(fisher)[1:x_train.shape[1]+1, 1:x_train.shape[1]+1]
    beta_chol = np.linalg.cholesky(beta_cov + 1e-10 * np.eye(x_train.shape[1]))
    return expit(b @ opt.x), opt.x, np.ascontiguousarray(beta_chol)


def bayes_probabilities(draws, x, groups, limit=1200):
    rng = np.random.default_rng(2026)
    selected = draws[rng.choice(len(draws), min(limit, len(draws)), replace=False)]
    p = x.shape[1]
    return expit(selected[:, p:p+3][:, groups] + selected[:, :p] @ x.T)


def score(name, y, p):
    return dict(model=name, n=len(y), prevalence=y.mean(), accuracy=accuracy_score(y, p >= .5),
                roc_auc=roc_auc_score(y, p), average_precision=average_precision_score(y, p),
                brier=brier_score_loss(y, p), log_loss=log_loss(y, p))


def main():
    OUT.mkdir(exist_ok=True)
    df, x_df, y, groups = prepare()
    train, test = train_test_split(np.arange(len(y)), test_size=.20, random_state=2026, stratify=y)
    x, means, sds = standardize(x_df, train)
    freq_p, freq_coef, beta_chol = fit_frequentist(x[train], y[train], groups[train], x[test], groups[test])
    names = list(x_df.columns) + ["alpha[Month-to-month]", "alpha[One year]", "alpha[Two year]", "mu_alpha", "tau_alpha"]
    settings = dict(n_iter=30000, burn=10000, thin=2, tau_scale=2.5)
    chains = []
    rates = []
    for seed in SEEDS:
        print(f"Fitting chain {seed}", flush=True)
        d, r = mh_chain(x[train], y[train], groups[train], beta_chol, seed=seed, **settings)
        chains.append(d); rates.append(r)
    chains = np.stack(chains)
    diag = diagnostics(chains, names)
    diag.to_csv(OUT / "mcmc_diagnostics.csv", index=False)
    selected = ["tenure", "MonthlyCharges", "Dependents", "OnlineSecurity", "TechSupport",
                "PaperlessBilling", "InternetService_Fiber optic", "PaymentMethod_Electronic check"]
    fixed_plot = diag.set_index("parameter").loc[selected].iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ypos = np.arange(len(fixed_plot))
    ax.errorbar(fixed_plot["mean"], ypos,
                xerr=np.vstack((fixed_plot["mean"]-fixed_plot["q2_5"], fixed_plot["q97_5"]-fixed_plot["mean"])),
                fmt="o", color="#246a73", capsize=2)
    ax.axvline(0, color="0.6", linestyle="--")
    ax.set(yticks=ypos, yticklabels=fixed_plot.index, xlabel="Log-odds coefficient",
           title="Training posterior: selected fixed effects")
    fig.tight_layout()
    fig.savefig(ROOT / "figures/posterior_fixed_effects_validation.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 3.4), sharex=True)
    for axis, name in zip(axes, ("MonthlyCharges", "mu_alpha")):
        j = names.index(name)
        for c, seed in enumerate(SEEDS):
            axis.plot(chains[c, ::20, j], linewidth=.55, alpha=.75, label=f"Chain {seed}")
        axis.set_ylabel(name)
    axes[0].legend(ncol=4, fontsize=7, frameon=False)
    axes[-1].set_xlabel("Retained draw / 20")
    fig.tight_layout()
    fig.savefig(ROOT / "figures/mcmc_trace_validation.png", dpi=170)
    plt.close(fig)
    pd.DataFrame(rates, columns=["beta", "z_alpha", "mu_alpha", "tau_alpha", "centered_tau"], index=SEEDS).to_csv(OUT / "acceptance_rates.csv", index_label="seed")
    draws = chains.reshape(-1, chains.shape[-1])
    p_features = x.shape[1]
    contract_rows = []
    for g, label in enumerate(("Month-to-month", "One year", "Two year")):
        alpha = draws[:, p_features + g]
        contract_rows.append(dict(contract=label, bayes_mean=alpha.mean(),
                                  bayes_q2_5=np.quantile(alpha, .025),
                                  bayes_q97_5=np.quantile(alpha, .975),
                                  frequentist_intercept=freq_coef[0] + (freq_coef[-2] if g == 1 else freq_coef[-1] if g == 2 else 0)))
    pd.DataFrame(contract_rows).to_csv(OUT / "contract_intercept_comparison.csv", index=False)
    bayes_draw_probs = bayes_probabilities(draws, x[test], groups[test])
    bayes_p = bayes_draw_probs.mean(axis=0)
    base_p = np.full(len(test), y[train].mean())
    scores = pd.DataFrame([score("Prevalence baseline", y[test], base_p),
                           score("Frequentist logistic", y[test], freq_p),
                           score("Bayesian hierarchical logistic", y[test], bayes_p)])
    scores.to_csv(OUT / "heldout_metrics.csv", index=False)
    pd.DataFrame(dict(customer_id=df.iloc[test]["customerID"].to_numpy(), churn=y[test],
                      contract=df.iloc[test]["Contract"].to_numpy(),
                      bayes_probability=bayes_p, frequentist_probability=freq_p,
                      baseline_probability=base_p)).to_csv(OUT / "heldout_predictions.csv", index=False)
    bins = pd.qcut(bayes_p, 10, duplicates="drop")
    calibration = pd.DataFrame(dict(bin=bins.astype(str), observed=y[test], predicted=bayes_p)).groupby("bin", sort=True).agg(n=("observed", "size"), observed_rate=("observed", "mean"), predicted_rate=("predicted", "mean"))
    calibration.to_csv(OUT / "calibration_deciles.csv")
    fig, ax = plt.subplots(figsize=(4.7, 4.3))
    ax.plot([0, 1], [0, 1], color="0.65", linestyle="--", label="Perfect calibration")
    ax.scatter(calibration["predicted_rate"], calibration["observed_rate"], color="#246a73", s=40, label="Bayesian model")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted churn probability", ylabel="Observed churn rate", title="Held-out calibration by prediction decile")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "figures/heldout_calibration.pdf")
    plt.close(fig)
    rng = np.random.default_rng(777)
    # Each posterior draw generates a genuinely replicated binary outcome vector.
    replicated = rng.binomial(1, bayes_draw_probs)
    checks = []
    for code, label in [(-1, "Overall"), (0, "Month-to-month"), (1, "One year"), (2, "Two year")]:
        mask = np.ones(len(test), dtype=bool) if code == -1 else groups[test] == code
        replicate_rate = replicated[:, mask].mean(axis=1)
        checks.append(dict(group=label, n=int(mask.sum()), observed_rate=y[test][mask].mean(),
                           predicted_rate=bayes_p[mask].mean(),
                           replicate_q2_5=np.quantile(replicate_rate, .025),
                           replicate_q97_5=np.quantile(replicate_rate, .975)))
    pd.DataFrame(checks).to_csv(OUT / "posterior_predictive_checks.csv", index=False)
    sensitivity_rows = []
    for scale in (2.5, 1.0):
        if scale == 2.5:
            sensitivity_chains = chains
        else:
            sensitivity_chains = []
            for seed in SEEDS:
                print(f"Prior sensitivity chain {seed}", flush=True)
                d, _ = mh_chain(x[train], y[train], groups[train], beta_chol,
                                settings["n_iter"], settings["burn"], settings["thin"], seed + 100, scale)
                sensitivity_chains.append(d)
            sensitivity_chains = np.stack(sensitivity_chains)
        selected_names = names[p_features:]
        selected_diag = diagnostics(sensitivity_chains[:, :, p_features:], selected_names)
        for row in selected_diag.to_dict("records"):
            row["tau_prior_scale"] = scale
            sensitivity_rows.append(row)
    pd.DataFrame(sensitivity_rows).to_csv(OUT / "prior_sensitivity.csv", index=False)
    config = dict(split_seed=2026, test_fraction=.2, train_n=len(train), test_n=len(test),
                  sampler=settings, sensitivity_tau_scale=1.0, chain_seeds=SEEDS, continuous_means=means.to_dict(),
                  continuous_sds=sds.to_dict(), feature_names=list(x_df.columns),
                  interpretation="Current customer snapshot only; observational associations are not causal.")
    (OUT / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(scores.to_string(index=False))
    print(diag[["parameter", "rhat_rank_split", "ess_bulk_approx", "ess_tail_approx"]].to_string(index=False))


if __name__ == "__main__":
    main()
