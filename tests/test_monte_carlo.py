import numpy as np
import monte_carlo as mc
import black_scholes as bs
import pytest

N_PATHS = 1_000_000
SEED = 42

PARAMS = [
    (100, 100, 1.0, 0.05, 0.20, 0.00),   # ATM, sans dividende
    (100, 100, 1.0, 0.05, 0.20, 0.02),   # ATM, avec dividende
    (100,  80, 0.5, 0.03, 0.25, 0.01),   # call ITM, maturité courte
    (100, 130, 2.0, 0.04, 0.30, 0.03),   # call OTM, maturité moyenne
    (100, 100, 5.0, 0.02, 0.15, 0.04),   # longue maturité, vol faible
    (50,   60, 0.25, 0.00, 0.60, 0.00),  # taux nul, vol très élevée
]

@pytest.mark.parametrize("S, K, T, r, sigma, q", PARAMS)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_mc_vs_bs(S, K, T, r, sigma, q, option_type):
    price, standard_error, _ = mc.mc_european(S, K, T, r, sigma, q , option_type=option_type, n_paths=N_PATHS, seed=SEED) 

    if option_type == "call" :
       bs_price = bs.bs_call(S, K, T, r, sigma, q)
    elif option_type == "put" :
       bs_price = bs.bs_put(S, K, T, r, sigma, q)
    
    assert abs(price - bs_price) < 3 * standard_error