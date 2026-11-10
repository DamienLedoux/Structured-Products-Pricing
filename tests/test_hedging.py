import numpy as np
import pytest
import black_scholes as bs
import paths as ps
import hedging as hg

SEED = 42
N_PATHS = 200_000


def test_pnl_shape():
    S = ps.simulate_gbm_paths(S0=100, T=1.0, mu=0.05, sigma=0.2, n_steps=12, n_paths=100, seed=SEED)
    pnl = hg.delta_hedge_pnl(S, K=100, T=1.0, r=0.05, sigma_impl=0.2)
    assert pnl.shape == (100,)

def test_flat_path_one_step():
    S = np.full((3, 2), 100.0)          # 3 flat paths, n_steps = 1
    K, T, r, sigma = 100, 1.0, 0.05, 0.2

    C0 = bs.bs_call(100, K, T, r, sigma)
    delta0 = bs.bs_delta_call(100, K, T, r, sigma)
    expected = (C0 - delta0 * 100) * np.exp(r * T) + delta0 * 100

    pnl = hg.delta_hedge_pnl(S, K, T, r, sigma)
    assert pnl == pytest.approx(expected)          # fills the 3 paths

def test_invalid_q():
    S = ps.simulate_gbm_paths(S0=100, T=1.0, mu=0.05, sigma=0.2, n_steps=12, n_paths=100, seed=SEED)
    with pytest.raises(ValueError, match="q must be 0"):
        hg.delta_hedge_pnl(S, K=100, T=1.0, r=0.05, sigma_impl=0.2, c=0.0, q=0.01)

def test_invalid_n_cols():
    S = np.full((3, 1), 100.0)
    with pytest.raises(ValueError, match="must have >= 2 columns"):
        hg.delta_hedge_pnl(S, K=100, T=1.0, r=0.05, sigma_impl=0.2, c=0.0, q=0.0)

def _simulate_replication(n_steps=252, n_paths=N_PATHS // 4):
    """Simulate paths once and return the discounted P&L of the delta hedge."""
    r=0.05
    T=1.0
    K=100
    sigma=0.2 
    S0=100

    S = ps.simulate_gbm_paths(S0=S0, T=T, mu=r, sigma=sigma, n_steps=n_steps, n_paths=n_paths, seed=SEED)
    pnl = hg.delta_hedge_pnl(S, K, T, r, sigma_impl=sigma)
    pnl_disc = pnl * np.exp(-r * T) # discounted pnl
    return pnl_disc


def test_replication_mean():
    pnl_disc = _simulate_replication()

    # self-financing + correct premium => E[discounted P&L] = 0
    se = pnl_disc.std(ddof=1) / np.sqrt(len(pnl_disc))
    assert abs(pnl_disc.mean()) < 3 * se, "mean of discounted P&L not 0"

def test_replication_std():
    r, T, K, sigma, S0 = 0.05, 1.0, 100, 0.2, 100
    n_steps = 252
    pnl_disc = _simulate_replication(n_steps=n_steps)

    # asymptotic std of the discrete-rebalancing hedging error
    # (Bertsimas, Kogan & Lo): sqrt(pi/4) * vega * sigma * sqrt(dt)
    vega = bs.bs_vega(S0, K, T, r, sigma)
    dt = T / n_steps
    std_theory = np.sqrt(np.pi / 4) * vega * sigma * np.sqrt(dt)

    std_obs = pnl_disc.std(ddof=1)
    assert std_obs == pytest.approx(std_theory, rel=0.10), (f"std of discounted P&L = {std_obs:.3f}, theory = {std_theory:.3f}")