import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import powerlaw
import statsmodels.api as sm

from collections import Counter
from numba import njit
from copia.rarefaction_extrapolation import rarefaction_extrapolation
from copia.coverage import estimate_coverage

from model import Simulator


def make_popsizes(start, end, factor):
    n = int(np.ceil(np.log(end / start) / np.log(factor)))
    return np.round(start * np.logspace(0, n, n + 1, base=factor)).astype(int)

def plot_ccdf(counts, marker='o', color='C0', label=None, ax=None, figsize=(8, 6)):
    counts = np.asarray(counts).ravel()    
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    powerlaw.plot_ccdf(counts, marker=marker, color=color, ax=ax)
    ax.set_xlabel('Count value')
    ax.set_ylabel('CCDF: P(X ≥ x)')
    ax.set(xscale="log", yscale="log")    
    return ax

def pielou_evenness(counts):
    counts = np.asarray(counts)
    counts = counts[counts > 0]
    if counts.size <= 1:
        return 0.0
    p = counts / counts.sum()
    H = -(p * np.log(p)).sum()
    return H / np.log(counts.size)

def hill_numbers(x, q_values):
    x = x[x > 0]
    p = x[x > 0] / x.sum()

    def sub(q):
        return ((x > 0).sum() if q == 0 else np.exp(-np.sum(p * np.log(p)))
                if q == 1 else np.exp(1 / (1 - q) * np.log(np.sum(p**q))))

    return np.array([sub(q) for q in q_values])

def area_under_curve(x, y):    
    return np.trapz(y, x)

def evenness_auc(counts, qs=np.arange(0, 3, 0.25)):
    counts = counts[counts > 0]
    if counts.size <= 1:
        return 0.0

    D = hill_numbers(counts, qs)
    E = (D - 1) / (D[0] - 1) 
    return np.trapz(E, qs) / (qs[-1] - qs[0])

def stratified_sampling(populations, pop_sizes, bias=0.0, target_sample_size=1000):
    pop_sizes = np.array(pop_sizes, dtype=np.float64)
    pop_sizes = pop_sizes ** (1 - bias)  # np.exp(-bias)
    sampling_probabilities = pop_sizes / pop_sizes.sum()

    samples = [[] for _ in range(len(pop_sizes))]
    populations = [population.tolist() for population in populations]

    sample_size = 0
    while sample_size < target_sample_size:
        if all(len(pop) == 0 for pop in populations):
            break
        idx = np.random.choice(len(pop_sizes), p=sampling_probabilities)
        population = populations[idx]

        if not population:
            sampling_probabilities[idx] = 0
            sampling_probabilities /= sampling_probabilities.sum()
            continue

        individual_idx = np.random.randint(0, len(population))
        individual = population.pop(individual_idx)

        samples[idx].append(individual)
        sample_size += 1

    return samples


def productivity_paradox_sampling(populations, pop_sizes, beta=0.7, target_sample_size=1000, eps=1.0, per_capita=False, rng=None):
    rng = np.random.default_rng() if rng is None else rng

    K = len(pop_sizes)
    samples = [[] for _ in range(K)]
    pops = [p.tolist() for p in populations]
    samples_per_pop = np.zeros(K, dtype=float)
    N = np.asarray(pop_sizes, dtype=float)

    sample_size = 0
    while sample_size < target_sample_size:
        if all(len(pop) == 0 for pop in pops):
            break

        if per_capita:
            prod_raw = (samples_per_pop / np.maximum(N, 1)) + eps
        else:
            prod_raw = samples_per_pop + eps

        weights = prod_raw ** beta

        for i, pop in enumerate(pops):
            if len(pop) == 0:
                weights[i] = 0.0

        s = weights.sum()
        if s == 0:
            break
        probs = weights / s

        i = rng.choice(K, p=probs)
        if not pops[i]:
            continue

        j = rng.integers(len(pops[i]))
        individual = pops[i].pop(j)
        samples[i].append(individual)
        samples_per_pop[i] += 1
        sample_size += 1

    return samples


def stromers_riddle_sampling(populations, pop_sizes, gamma=1.0, target_sample_size=1000, rng=None):
    """
    Rarity-based sampling WITHIN each population, without replacement.
    Populations are chosen UNIFORMLY among the non-empty ones each draw.
    """
    if gamma < 0:
        raise ValueError("gamma must be > 0")
    if rng is None:
        rng = np.random.default_rng()

    counters = [Counter(pop) for pop in populations]
    totals = [sum(c.values()) for c in counters]
    n_pops = len(counters)
    samples = [[] for _ in range(n_pops)]

    def variant_weights(counter):
        if not counter:
            return [], np.array([])
        variants, counts = zip(*counter.items())
        counts = np.asarray(counts, dtype=float)
        if gamma == 0:
            weights = counts
        else:
            weights = counts ** (1.0 - gamma)
        s = weights.sum()
        if s == 0:
            weights = np.ones_like(weights)
            s = weights.sum()
        return list(variants), weights / s

    draws = 0
    while draws < target_sample_size:
        if all(t == 0 for t in totals):
            break

        available = [i for i, t in enumerate(totals) if t > 0]
        pop_idx = rng.choice(available)

        variants, w = variant_weights(counters[pop_idx])
        if not variants:
            totals[pop_idx] = 0
            continue

        v_idx = rng.choice(len(variants), p=w)
        v = variants[v_idx]

        samples[pop_idx].append(v)

        counters[pop_idx][v] -= 1
        totals[pop_idx] -= 1
        if counters[pop_idx][v] <= 0:
            del counters[pop_idx][v]

        draws += 1

    return samples


def count_traits(samples):
    return np.array([len(set(sample)) for sample in samples])

def rarefaction_curve(ds, max_steps=None):
    if max_steps is None:
        max_steps=ds.n
    rarefaction = rarefaction_extrapolation(ds, max_steps=max_steps)
    coverage = estimate_coverage(ds, max_steps=max_steps)
    return rarefaction, coverage

def compute_coverage(ds):
    correction = 1
    if ds.f2 > 0:
        correction = (
            ((ds.n - 1) * ds.f1) / ((ds.n - 1) * ds.f1 + 2 * ds.f2))
    return 1 - (ds.f1 / ds.n) * correction

def create_analysis_dataframe(sample_sizes, betas, pop_sizes, sample_size_estimates, 
                             coverage_estimates, true_numbers):
    data = []
    for i, sample_size in enumerate(sample_sizes):
        for j, beta in enumerate(betas):
            for k in range(len(pop_sizes)):
                sample_size_estimate = sample_size_estimates[i][j][k].astype(int)
                coverage_estimate = coverage_estimates[i][j][k].astype(int)
                true_number = int(true_numbers[k])
                data.append({
                    "sample": sample_size_estimate,
                    "coverage": coverage_estimate,
                    "true": true_number,
                    "popsize": pop_sizes[k],
                    "sample_size": sample_size,
                    "beta": beta
                })
    return pd.DataFrame(data)

def fit_poisson_model(y, X, model_name="model", verbose=False):
    try:
        model = sm.GLM(y, X, family=sm.families.Poisson())
        fit = model.fit(disp=verbose)
        return fit
    except Exception as e:
        if verbose:
            print(f"Error fitting {model_name}: {e}")
        return None

def run_poisson_models(data, groupby_cols=["sample_size", "beta"], 
                      response_cols=["true", "sample", "coverage"], 
                      predictor_col="popsize", verbose=False):
    runs = {}
    for group_key, group in data.groupby(groupby_cols):
        X = sm.add_constant(np.log(group[predictor_col]))
        
        model_fits = []
        for response_col in response_cols:
            fit = fit_poisson_model(
                group[response_col], X, 
                model_name=f"{response_col} model for {group_key}", 
                verbose=verbose
            )
            model_fits.append(fit)
        
        runs[group_key] = tuple(model_fits)
    
    return runs

def extract_model_estimates(runs, sample_sizes, betas, 
                           response_names=["true", "sample_based", "coverage_based"], 
                           param_index=1):
    estimates = []
    for sample_size in sample_sizes:
        for beta in betas:
            model_fits = runs.get((sample_size, beta), (None,) * len(response_names))
            
            coefficients = {}
            for i, name in enumerate(response_names):
                fit = model_fits[i]
                coef = fit.params[param_index] if fit is not None else np.nan
                coefficients[name] = coef
            
            coefficients["beta"] = beta
            coefficients["sample_size"] = sample_size
            
            estimates.append(coefficients)
    
    return pd.DataFrame(estimates)


def analyze_estimation_data(sample_sizes, betas, pop_sizes, sample_size_estimates, 
                          coverage_estimates, true_numbers, verbose=False):
    data_df = create_analysis_dataframe(
        sample_sizes, betas, pop_sizes, 
        sample_size_estimates, coverage_estimates, true_numbers)
    
    model_runs = run_poisson_models(
        data_df, 
        groupby_cols=["sample_size", "beta"],
        response_cols=["true", "sample", "coverage"],
        predictor_col="popsize",
        verbose=verbose)
    
    estimates_df = extract_model_estimates(
        model_runs, sample_sizes, betas,
        response_names=["true", "sample_based", "coverage_based"])
    
    return data_df, model_runs, estimates_df

def crp_sample(N, theta, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    tables = []
    for i in range(N):  
        if rng.random() < theta / (theta + i):
            tables.append(1)
        else:
            probs = np.array(tables, dtype=float) / i
            idx = rng.choice(len(tables), p=probs)
            tables[idx] += 1
    return np.asarray(tables, dtype=int)

@njit
def crp_sample_numba(N, theta):
    counts = np.zeros(N, dtype=np.int64)
    K = 0 
    for i in range(N):
        if np.random.random() < theta / (theta + i):
            counts[K] = 1
            K += 1
        else:
            u = np.random.random() * i
            cum = 0.0
            for j in range(K):
                cum += counts[j]
                if u < cum:
                    counts[j] += 1
                    break
    return counts[:K]

def interpolate_richness_at_coverage(rarefaction, coverage_curve, target_coverage):
    rare = np.asarray(rarefaction, dtype=float)
    cov  = np.asarray(coverage_curve, dtype=float)

    c_min, c_max = cov[0], cov[-1]
    if not (c_min <= target_coverage <= c_max):
        return None

    j = int(np.argmax(cov >= target_coverage))  
    if j == 0 or cov[j] == target_coverage:
        return float(rare[j])

    i = j - 1
    c_lo, c_hi = cov[i], cov[j]
    r_lo, r_hi = rare[i], rare[j]

    w = (target_coverage - c_lo) / (c_hi - c_lo)
    w = min(max(w, 0.0), 1.0)
    estimate = r_lo + w * (r_hi - r_lo)
    low, high = sorted((r_lo, r_hi))
    estimate = min(max(estimate, low), high)
    return max(float(estimate), 1)

def make_population(N, theta=1, mu=None):
    if mu is not None:
        theta = 2 * N * mu
    counts = crp_sample_numba(N, theta)
    population = np.repeat(np.arange(len(counts)), counts)
    return population

def make_population_wf(N, mu):
    model = Simulator(n_agents=N, mu=mu, beta=0, disable_pbar=True)
    model.fit()
    return model.sample(return_counts=False)

def richness_at_coverage_argmin(rarefaction, coverage_curve, target_coverage):
    if np.max(coverage_curve) < target_coverage:
        return None
    idx = np.argmin(np.abs(coverage_curve - target_coverage))
    return rarefaction[idx]

def plot_relative_richness(estimates, sample_sizes, pop_sizes, betas, S_true,
                           title=None, save_path=None, figsize=(8, 2.3), max_reference=False,
                           ncols=None, nrows=1):
    if ncols is None:
        nrows, ncols = 1, len(sample_sizes)
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize, constrained_layout=True, sharey=True)
    axes = axes.flatten()

    def reference_point(x):
        return max(x) if max_reference else x[-1]
    
    for i, sample_size in enumerate(sample_sizes):
        for j in range(len(betas)):
            axes[i].plot(pop_sizes, estimates[i, j] / reference_point(estimates[i, j]))
        
        axes[i].set(xscale="log")
        axes[i].text(0.05, 0.9, f"n={sample_size}", 
                    horizontalalignment='left',
                    verticalalignment='center', 
                    transform=axes[i].transAxes, 
                    fontsize=10)
        axes[i].plot(pop_sizes, S_true / reference_point(S_true), '--', color="grey", zorder=20)
    
    axes[0].set_ylabel("Relative richness", size=10)
    
    fig.supxlabel("Population size (log)", size=10)
    fig.suptitle(title, x=1, y=1, ha="right")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig, axes