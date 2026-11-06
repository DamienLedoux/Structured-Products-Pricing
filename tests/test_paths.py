import numpy as np
import pytest
import paths as ps

SEED = 42
BASE = dict(S0=100, mu=0.05, q=0.0)
N_PATHS = 200_000

PARAMS = [
    # S0,  T,    mu,    sigma, n_steps, q
    (100, 1.0,  0.05,  0.20,   1,   0.00),   # reference, only one step
    (100, 1.0,  0.05,  0.20,  12,   0.02),   # q != 0
    (100, 0.5,  0.10,  0.25,  52,   0.01),   # high mu
    (100, 2.0, -0.03,  0.30,  24,   0.03),   # negative mu
    (100, 3.0,  0.04,  0.40,   7,   0.04),   # high sigma and T : Ito very visible
    ( 50, 0.25, 0.00,  0.60, 100,   0.00),   # mu = 0, very high vol
]

# for shape tests
SHAPES = [
    # (n_steps, n_paths)
    (7, 50),     # base case
    (1, 50),     # only one step -> (50, 2)
    (7, 1),      # only one path -> (1, 8)
    (252, 10),   # a lot of steps
]

# for positivity tests
POSITIVE = [
    # (sigma, T)
    (0.2, 1.0),   # base case
    (0.8, 5.0),   # high vol and long T : the most demanding
    (0.6, 0.25),  # high vol, short T
]

# params for the reproducibility test (single set, nothing to parametrize)
REPRO = dict(T=1, sigma=0.2, n_steps=7, n_paths=100)

# test_shape: checks the array has shape (n_paths, n_steps + 1)
@pytest.mark.parametrize("n_steps, n_paths", SHAPES)
def test_shape(n_steps, n_paths):
    S = ps.simulate_gbm_paths(**BASE, T=1, sigma=0.2, n_steps=n_steps, n_paths=n_paths, seed=SEED)
    assert S.shape == (n_paths, n_steps + 1)

# test_initial_value: checks column 0 equals S0 (up to rounding)
@pytest.mark.parametrize("n_steps, n_paths", SHAPES)
def test_initial_value(n_steps, n_paths):
    S = ps.simulate_gbm_paths(**BASE, T=1, sigma=0.2, n_steps=n_steps, n_paths=n_paths, seed=SEED)
    # column 0 = exp(log(S0)) : equal to S0 up to floating point rounding
    assert np.allclose(S[:, 0], BASE["S0"])

# test_reprod: checks same seed gives same paths, different seed gives different ones
def test_reprod():
    S_1 = ps.simulate_gbm_paths(**BASE, **REPRO, seed=SEED)
    S_2 = ps.simulate_gbm_paths(**BASE, **REPRO, seed=SEED)
    S_3 = ps.simulate_gbm_paths(**BASE, **REPRO, seed=SEED + 1)
    assert np.array_equal(S_1, S_2)
    assert not np.array_equal(S_1, S_3)

# test_positive: checks all simulated prices are strictly positive
@pytest.mark.parametrize("sigma, T", POSITIVE)
def test_positive(sigma, T):
    S = ps.simulate_gbm_paths(**BASE, sigma=sigma, T=T, n_steps=52, n_paths=200, seed=SEED)
    assert np.all(S > 0)

# test_invalid_n_paths: checks ValueError is raised for n_paths < 1
@pytest.mark.parametrize("bad_n_paths", [0, -1, -100])
def test_invalid_n_paths(bad_n_paths):
    with pytest.raises(ValueError, match="n_paths"):
        ps.simulate_gbm_paths(**BASE, T=1, sigma=0.2, n_steps=7, n_paths=bad_n_paths, seed=SEED)

# test_invalid_n_steps: checks ValueError is raised for n_steps < 1
@pytest.mark.parametrize("bad_n_steps", [0, -1, -5])
def test_invalid_n_steps(bad_n_steps):
    with pytest.raises(ValueError, match="n_steps"):
        ps.simulate_gbm_paths(**BASE, T=1, sigma=0.2, n_steps=bad_n_steps, n_paths=100, seed=SEED)

# test_ST_mean: checks E[S_T] matches S0 * exp((mu - q) * T) within 3 standard errors
@pytest.mark.parametrize("S0, T, mu, sigma, n_steps, q", PARAMS)
def test_ST_mean(S0, T, mu, sigma, n_steps, q) :

    theoretical_ST_mean = S0 * np.exp((mu-q)*T)

    S = ps.simulate_gbm_paths(S0=S0, T=T, mu=mu, sigma=sigma, n_steps=n_steps, n_paths=N_PATHS, q=q, seed=SEED)
    empirical_ST_mean = S[:, -1].mean()

    std = S[:, -1].std(ddof=1)
    se = std / np.sqrt(N_PATHS)

    assert abs(empirical_ST_mean - theoretical_ST_mean) < 3*se

# test_logreturn_mean: checks E[ln(S_T/S0)] matches (mu - q - sigma^2/2) * T within 3 standard errors
@pytest.mark.parametrize("S0, T, mu, sigma, n_steps, q", PARAMS)
def test_logreturn_mean(S0, T, mu, sigma, n_steps, q) : # logreturn = x

    S = ps.simulate_gbm_paths(S0=S0, T=T, mu=mu, sigma=sigma, n_steps=n_steps, n_paths=N_PATHS, q=q, seed=SEED)
    x = np.log(S[:, -1]/S0)
    
    theoretical_x_mean = (mu-q-sigma**2/2)*T

    se = x.std(ddof=1)/np.sqrt(N_PATHS)

    assert abs(x.mean() - theoretical_x_mean) < 3*se

# test_logreturn_var: checks Var[ln(S_T/S0)] matches sigma^2 * T within 2%
@pytest.mark.parametrize("S0, T, mu, sigma, n_steps, q", PARAMS)
def test_logreturn_var(S0, T, mu, sigma, n_steps, q) : # logreturn = x

    S = ps.simulate_gbm_paths(S0=S0, T=T, mu=mu, sigma=sigma, n_steps=n_steps, n_paths=N_PATHS, q=q, seed=SEED)
    x = np.log(S[:, -1]/S0)
    
    assert x.var(ddof=1) == pytest.approx((sigma**2)*T, rel=0.02)

# test_law_at_each_date: checks mean of S and variance of ln(S/S0) are correct at intermediate dates, not only at T
@pytest.mark.parametrize("S0, T, mu, sigma, n_steps, q", PARAMS)
def test_law_at_each_date(S0, T, mu, sigma, n_steps, q) :
    S = ps.simulate_gbm_paths(S0=S0, T=T, mu=mu, sigma=sigma, n_steps=n_steps, n_paths=N_PATHS, q=q, seed=SEED)
    ks = sorted({1, max(1, n_steps // 2), n_steps})
    for k in ks :
        t_k = k*(T/n_steps)
        se = S[:, k].std(ddof=1)/np.sqrt(N_PATHS)

        assert abs(S[:, k].mean() - S0*np.exp((mu - q) * t_k)) < 3*se, f"mean, k={k}"
        assert np.log(S[:, k]/S0).var(ddof=1) == pytest.approx((sigma**2)*t_k, rel=0.02), f"var, k={k}"