import numpy as np
import pytest
import black_scholes as bs
import paths as ps
import hedging as hg

SEED = 42
N_PATHS = 200_000
N_PATHS_SMALL = 5 # for when not a lot of paths are needed
N_STEPS_DAILY = 252 # number of trading days in a year
SOME_N_STEPS = [25, 50, 100, 200, 400] # several number of steps for last test (or when needed)

# base parameters shared by all hedging tests
# mu = R everywhere (risk-neutral drift): the discounted hedging portfolio is then a
# martingale, so the mean of the discounted P&L only depends on what is being tested
# (e.g. the gap between real and implied vol), not on the stock drift.
R = 0.05
T = 1.0
K = 100
SIGMA = 0.2
S0 = 100


def test_pnl_shape():
    # mu = R: see top of file
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
    # mu = R: see top of file
    S = ps.simulate_gbm_paths(S0=S0, T=T, mu=R, sigma=SIGMA, n_steps=12, n_paths=100, seed=SEED)
    with pytest.raises(ValueError, match="q must be 0"):
        hg.delta_hedge_pnl(S, K, T, R, sigma_impl=SIGMA, c=0.0, q=0.01)


def test_invalid_n_cols():
    S = np.full((3, 1), float(S0))
    with pytest.raises(ValueError, match="must have >= 2 columns"):
        hg.delta_hedge_pnl(S, K, T, R, sigma_impl=SIGMA, c=0.0, q=0.0)

# Utility function
def _simulate_replication(n_steps=252, n_paths=N_PATHS // 4):
    """Simulate paths once and return the discounted P&L of the delta hedge."""
    # mu = R: see top of file
    S = ps.simulate_gbm_paths(S0=S0, T=T, mu=R, sigma=SIGMA, n_steps=n_steps, n_paths=n_paths, seed=SEED)
    pnl = hg.delta_hedge_pnl(S, K, T, R, sigma_impl=SIGMA)
    return pnl * np.exp(-R * T)         # discounted P&L


def test_replication_mean():
    pnl_disc = _simulate_replication()

    # self-financing + correct premium => E[discounted P&L] = 0
    se = pnl_disc.std(ddof=1) / np.sqrt(len(pnl_disc))
    assert abs(pnl_disc.mean()) < 3 * se, "mean of discounted P&L not 0"


def test_replication_std():
    n_steps = N_STEPS_DAILY
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
    # mu = R: see top of file
    S = ps.simulate_gbm_paths(S0=S0, T=T, mu=R, sigma=SIGMA, n_steps=n_steps, n_paths=n_paths, seed=SEED)

    pnl_0 = hg.delta_hedge_pnl(S, K, T, R, sigma_impl=SIGMA)
    pnl_c = hg.delta_hedge_pnl(S, K, T, R, sigma_impl=SIGMA, c=c)

    deltas = np.zeros((n_paths, n_steps))
    for k in range(n_steps):
        delta_k = bs.bs_delta_call(S=S[:, k], K=K, T=T-k*dt, r=R, sigma=SIGMA, q=0.0)
        deltas[:, k] = delta_k

    deltas_diff = np.diff(deltas, axis=1)
    trade_idx = np.arange(1, n_steps)
    cost = ((c*np.abs(deltas_diff)) * S[:, 1 : n_steps] * np.exp(R*(T-trade_idx*dt))).sum(axis=1)

    assert pnl_0 - pnl_c == pytest.approx(cost), "pnl_0 - pnl_c was not close to its theoretical value"
    assert np.all(pnl_c < pnl_0), "pnl_c (w transac. cost) was not less than pnl_0 (w/o transac cost)"

@pytest.mark.parametrize("sigma_real, sign", [(0.15, +1), (0.25, -1)])
def test_volatility(sigma_real, sign):
    # Seller hedges with sigma_impl = SIGMA but the stock really moves with sigma_real.
    # sigma_real < SIGMA: option sold too expensive -> seller wins  (sign = +1)
    # sigma_real > SIGMA: option sold too cheap     -> seller loses (sign = -1)
    n_steps = N_STEPS_DAILY
    n_paths = N_PATHS // 4

    # mu = R: see top of file
    S = ps.simulate_gbm_paths(S0=S0, T=T, mu=R, sigma=sigma_real, n_steps=n_steps, n_paths=n_paths, seed=SEED)

    # premium and delta are computed with the implied vol (SIGMA), not the real one
    pnl = hg.delta_hedge_pnl(S, K, T, R, sigma_impl=SIGMA)

    # discount the P&L and compute se
    pnl_disc = pnl * np.exp(-R * T)
    se = pnl_disc.std(ddof=1) / np.sqrt(len(pnl_disc))

    # sign * mean > 3 se  <=>  mean > 3 se if sign = +1,  mean < -3 se if sign = -1
    assert pnl_disc.mean() * sign > 3*se, f"mean = {pnl_disc.mean():.3f}"

def test_slope():
    several_n_steps = SOME_N_STEPS
    stds = np.zeros(len(several_n_steps))

    for i, n_steps in enumerate(several_n_steps) :
        pnl_disc = _simulate_replication(n_steps=n_steps, n_paths=10_000)
        stds[i] = pnl_disc.std(ddof=1)
    
    slope, _ = np.polyfit(np.log(several_n_steps), np.log(stds), 1)

    assert slope == pytest.approx(-0.5, abs=0.05), f"slope measured is not close to theory (-0.5). slope measured : {slope:.3f}"
