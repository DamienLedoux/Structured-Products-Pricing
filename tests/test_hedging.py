import numpy as np
import pytest
import black_scholes as bs
import paths as ps
import hedging as hg

SEED = 42
N_PATHS = 200_000
N_PATHS_SMALL = 5 # for when not a lot of paths are needed

# base parameters shared by all hedging tests
R = 0.05
T = 1.0
K = 100
SIGMA = 0.2
S0 = 100


def test_pnl_shape():
    S = ps.simulate_gbm_paths(S0=S0, T=T, mu=R, sigma=SIGMA, n_steps=12, n_paths=100, seed=SEED)
    pnl = hg.delta_hedge_pnl(S, K, T, R, sigma_impl=SIGMA)
    assert pnl.shape == (100,)


def test_flat_path_one_step():
    S = np.full((3, 2), float(S0))      # 3 flat paths, n_steps = 1

    C0 = bs.bs_call(S0, K, T, R, SIGMA)
    delta0 = bs.bs_delta_call(S0, K, T, R, SIGMA)
    expected = (C0 - delta0 * S0) * np.exp(R * T) + delta0 * S0

    pnl = hg.delta_hedge_pnl(S, K, T, R, SIGMA)
    assert pnl == pytest.approx(expected)       # all 3 paths equal expected


def test_invalid_q():
    S = ps.simulate_gbm_paths(S0=S0, T=T, mu=R, sigma=SIGMA, n_steps=12, n_paths=100, seed=SEED)
    with pytest.raises(ValueError, match="q must be 0"):
        hg.delta_hedge_pnl(S, K, T, R, sigma_impl=SIGMA, c=0.0, q=0.01)


def test_invalid_n_cols():
    S = np.full((3, 1), float(S0))
    with pytest.raises(ValueError, match="must have >= 2 columns"):
        hg.delta_hedge_pnl(S, K, T, R, sigma_impl=SIGMA, c=0.0, q=0.0)


def _simulate_replication(n_steps=252, n_paths=N_PATHS // 4):
    """Simulate paths once and return the discounted P&L of the delta hedge."""
    S = ps.simulate_gbm_paths(S0=S0, T=T, mu=R, sigma=SIGMA, n_steps=n_steps, n_paths=n_paths, seed=SEED)
    pnl = hg.delta_hedge_pnl(S, K, T, R, sigma_impl=SIGMA)
    return pnl * np.exp(-R * T)         # discounted P&L


def test_replication_mean():
    pnl_disc = _simulate_replication()

    # self-financing + correct premium => E[discounted P&L] = 0
    se = pnl_disc.std(ddof=1) / np.sqrt(len(pnl_disc))
    assert abs(pnl_disc.mean()) < 3 * se, "mean of discounted P&L not 0"


def test_replication_std():
    n_steps = 252
    pnl_disc = _simulate_replication(n_steps=n_steps)

    # asymptotic std of the discrete-rebalancing hedging error:
    # sqrt(pi/4) * vega * sigma * sqrt(dt)  (asymptotic result, checked numerically)
    vega = bs.bs_vega(S0, K, T, R, SIGMA)
    dt = T / n_steps
    std_theory = np.sqrt(np.pi / 4) * vega * SIGMA * np.sqrt(dt)

    std_obs = pnl_disc.std(ddof=1)
    assert std_obs == pytest.approx(std_theory, rel=0.10), \
        f"std of discounted P&L = {std_obs:.3f}, theory = {std_theory:.3f}"

def test_transaction_cost():
    n_steps = 5
    dt = T / n_steps
    c = 0.01
    n_paths = N_PATHS_SMALL
    S = ps.simulate_gbm_paths(S0=S0, T=T, mu=R, sigma=SIGMA, n_steps=n_steps, n_paths=n_paths, seed=SEED)

    pnl_0 = hg.delta_hedge_pnl(S, K, T, R, sigma_impl=SIGMA)
    pnl_c = hg.delta_hedge_pnl(S, K, T, R, sigma_impl=SIGMA, c=c)

    deltas = np.zeros((n_paths, n_steps)) 
    for k in range (n_steps):
        delta_k = bs.bs_delta_call(S=S[:, k], K=K, T=T-k*dt, r=R, sigma=SIGMA, q=0.0)
        deltas[:, k] = delta_k

    deltas_diff = np.diff(deltas, axis=1)
    trade_idx = np.arange(1, n_steps)
    cost = ((c*np.abs(deltas_diff)) * S[:, 1 : n_steps] * np.exp(R*(T-trade_idx*dt))).sum(axis=1)

    assert pnl_0 - pnl_c == pytest.approx(cost), "pnl_0 - pnl_c was not close to its theoretical value"
    assert np.all(pnl_c < pnl_0), "pnl_c (w transac. cost) was not less than pnl_0 (w/o transac cost)"