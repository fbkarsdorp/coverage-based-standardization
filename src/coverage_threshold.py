import pickle
import numpy as np
import tqdm
import argparse

from simulation import make_population_wf


if __name__ == "__main__":
    from joblib import Parallel, delayed
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", type=float, default=2)
    args = parser.parse_args()

    n_cores = -1

    pop_sizes = 10**4, 10**4
    mu_values = 10**-3, args.factor*10**-3

    def make_populations(pop_sizes, μ_values):
        populations = [make_population_wf(N, μ) for N, μ in zip(pop_sizes, μ_values)]
        return populations, np.array([len(set(pop)) for pop in populations])
    
    results = Parallel(n_jobs=n_cores)(
        delayed(make_populations)(pop_sizes, mu_values) for _ in tqdm.trange(100)
    )

    populations_list = [r[0] for r in results]

    with open(f'threshold_populations1-{args.factor}.pkl', 'wb') as f:
        pickle.dump(populations_list, f)