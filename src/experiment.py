import tqdm
import numpy as np
from copia.data import to_copia_dataset

from simulation import stratified_sampling, productivity_paradox_sampling, stromers_riddle_sampling
from simulation import compute_coverage, rarefaction_curve, richness_at_coverage_argmin
from simulation import make_population_wf, make_popsizes

import warnings

def hill_number_q2(counts):
    p = counts / np.sum(counts)
    return 1.0 / np.sum(p**2)

def single_pass(pops, Ns, beta=0, sample_size=1000, sampling_method="stratified"):

    if sampling_method == "stratified":
        samples = stratified_sampling(pops, Ns, bias=beta, target_sample_size=sample_size)
    elif sampling_method == "productivity":
        samples = productivity_paradox_sampling(pops, Ns, beta=beta, target_sample_size=sample_size)
    elif sampling_method == "stromer":
        samples = stromers_riddle_sampling(pops, Ns, gamma=beta, target_sample_size=sample_size)

    datasets = [to_copia_dataset(
         sample, input_type="observations", data_type="abundance") for sample in samples]
    C_max = min(filter(lambda C: C > 0, (compute_coverage(ds) for ds in datasets)))
    n_min = min(filter(lambda n: n > 0, (ds.n for ds in datasets)))
    results = np.full((len(Ns), 4), np.nan) # n populations * 3 estimators + 1 (sample sizes) 
    for i, ds in enumerate(datasets):
        results[i, 0] = ds.S_obs
        rarefaction, coverage_curve = rarefaction_curve(ds)
        if len(rarefaction) > 0:
            if compute_coverage(ds) > 0:
                results[i, 1] = richness_at_coverage_argmin(rarefaction, coverage_curve, C_max)
            results[i, 2] = rarefaction[max(n_min - 1, 0)]
        results[i, 3] = ds.n

    return results, hill_number_q2(results[:, -1])

def run_model(populations, pop_sizes, beta=0, sample_size=1000, n_sampling_iterations=5, 
              sampling_method="stratified", disable=True):
    all_estimates = np.zeros((n_sampling_iterations, len(pop_sizes), 4))
    all_stats = []
    for i in tqdm.trange(n_sampling_iterations, desc="Sampling iterations", disable=False):
        try:
            results, stats = single_pass(
                populations, pop_sizes, beta, sample_size, sampling_method=sampling_method)
            all_estimates[i] = results
            all_stats.append(stats)
        except ValueError as e:
            print(e)
            all_estimates[i] = np.nan
            all_stats.append(np.nan)
    normalized_estimates = np.nanmean(all_estimates, axis=0)
    averaged_stats = np.nanmean(all_stats)
    return normalized_estimates, averaged_stats

def run_experiment(populations_list, pop_sizes, sample_sizes, betas, n_sampling_iterations=5, sampling_method="stratified"):
    
    def process_population(populations, pop_sizes, sample_sizes, betas, n_sampling_iterations, sampling_method):
        samples_per_pop = np.zeros((len(sample_sizes), len(betas), len(pop_sizes)))
        sample_size_estimates = np.zeros((len(sample_sizes), len(betas), len(pop_sizes)))
        coverage_estimates = np.zeros((len(sample_sizes), len(betas), len(pop_sizes)))
        rarefaction_estimates = np.zeros((len(sample_sizes), len(betas), len(pop_sizes)))
        statistics = np.zeros((len(sample_sizes), len(betas)))
        
        with warnings.catch_warnings(action="ignore"):
            for i, sample_size in enumerate(sample_sizes):
                for j, beta in enumerate(betas):
                    estimates, stats = run_model(
                        populations, pop_sizes, beta, sample_size,
                        n_sampling_iterations, sampling_method=sampling_method)
                    sample_size_estimates[i, j] = estimates[:, 0]
                    coverage_estimates[i, j] = estimates[:, 1]
                    rarefaction_estimates[i, j] = estimates[:, 2]
                    samples_per_pop[i, j] = estimates[:, 3]
                    statistics[i, j] = stats
        
        return sample_size_estimates, coverage_estimates, rarefaction_estimates, samples_per_pop, statistics
    
    results = Parallel(n_jobs=-1)(
        delayed(process_population)(populations, pop_sizes, sample_sizes, betas, n_sampling_iterations, sampling_method)
        for populations in tqdm.tqdm(populations_list, desc="Population sets")
    )
    
    sample_size_estimates = np.array([r[0] for r in results])
    coverage_estimates = np.array([r[1] for r in results])
    rarefaction_estimates = np.array([r[2] for r in results])
    samples_per_pop = np.array([r[3] for r in results])
    statistics = np.array([r[4] for r in results])
    
    return {
        'sample_size': sample_size_estimates,
        'rarefaction': rarefaction_estimates,
        'coverage': coverage_estimates,
        'sample_per_pop': samples_per_pop,
        'statistics': statistics,
    }


if __name__ == "__main__":
    from joblib import Parallel, delayed
    import pickle

    sample_sizes = 500, 1000, 5000#, 10000, 20000
    n_sampling_iterations = 100
    theta, mu = None, 0.0005
    pop_sizes = make_popsizes(1000, 100_000, 2).tolist()

    n_cores = -1

    def make_populations(pop_sizes):
        populations = [make_population_wf(N, mu) for N in pop_sizes]
        return populations, np.array([len(set(pop)) for pop in populations])

    results = Parallel(n_jobs=n_cores)(
        delayed(make_populations)(pop_sizes) for _ in tqdm.trange(100))

    populations_list = [r[0] for r in results]
    S_true = np.vstack([r[1] for r in results]).mean(axis=0)

    with open('populations_list.pkl', 'wb') as f:
        pickle.dump(populations_list, f)

    np.save('S_true.npy', S_true)

    print("=" * 80)
    print("Wright Fisher simulation done")

    with open('populations_list.pkl', 'rb') as f:
        populations_list = pickle.load(f)

    S_true = np.load("S_true.npy")

    print("=" * 80)
    print()
    print("Starting stratified sampling...")

    # Stratified sampling
    betas = np.linspace(-1, 1, num=5)
    exp_pop = run_experiment(
        populations_list, pop_sizes, sample_sizes, betas,
        n_sampling_iterations, sampling_method="stratified"
    )
    with open('exp_stratified-new.pkl', 'wb') as f:
        pickle.dump(exp_pop, f)
    print("Stratified sampling done...")
    print("=" * 80)

    # Productivity sampling
    print("Starting productivity sampling...")
    betas = [0, 0.5, 1]
    exp_prod = run_experiment(
        populations_list, pop_sizes, sample_sizes, betas,
        n_sampling_iterations, sampling_method="productivity"
    )
    with open('exp_productivity-new.pkl', 'wb') as f:
        pickle.dump(exp_prod, f)
    print("Productivity sampling done...")
    print("=" * 80)

    # Stromer sampling
    print("Starting Stromer sampling...")
    betas = [0, 1, 3]
    exp_strom = run_experiment(
        populations_list, pop_sizes, sample_sizes, betas,
        n_sampling_iterations, sampling_method="stromer"
    )
    with open('exp_stromer-new.pkl', 'wb') as f:
        pickle.dump(exp_strom, f)
    print("Stromer sampling done...")
    