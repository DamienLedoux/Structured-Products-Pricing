import numpy as np

def simulate_gbm_paths(S0, T, mu, sigma, n_steps, n_paths, q=0.0, seed=None):

    if n_paths < 1:
        raise ValueError(f"n_paths doit être >= 1, reçu : {n_paths}")
    if n_steps < 1:
        raise ValueError(f"n_steps doit être >= 1, reçu : {n_steps}")

    dt = T / n_steps

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(size=(n_paths,n_steps))

    log_increments = (mu-q-(sigma**2)/2)*dt + sigma*np.sqrt(dt)*Z # (n_paths, n_steps)
    
    zero_col = np.zeros((n_paths, 1))
    log_increments = np.concatenate([zero_col, log_increments], axis=1)  # (n_paths, n_steps+1)

    log_paths = np.log(S0) + np.cumsum(log_increments, axis=1)          # (n_paths, n_steps+1)

    S = np.exp(log_paths)
    return S

