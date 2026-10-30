import numpy as np
import monte_carlo as mc
import black_scholes as bs
import pytest

N_PATHS = 1_000_000
SEED = 42
ATM = dict(S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.0) # when no need for all the test variables, lighten the load

PARAMS = [
    (100, 100, 1.0, 0.05, 0.20, 0.00),   # ATM, sans dividende
    (100, 100, 1.0, 0.05, 0.20, 0.02),   # ATM, avec dividende
    (100,  80, 0.5, 0.03, 0.25, 0.01),   # call ITM, maturité courte
    (100, 130, 2.0, 0.04, 0.30, 0.03),   # call OTM, maturité moyenne
    (100, 100, 5.0, 0.02, 0.15, 0.04),   # longue maturité, vol faible
    (50,   60, 0.25, 0.00, 0.60, 0.00),  # taux nul, vol très élevée
]

# Check if the approximated price via MC is right (against Black-Scholes model)
@pytest.mark.parametrize("S, K, T, r, sigma, q", PARAMS)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_mc_vs_bs(S, K, T, r, sigma, q, option_type):
    price, standard_error, _ = mc.mc_european(S, K, T, r, sigma, q , option_type=option_type, n_paths=N_PATHS, seed=SEED) 

    if option_type == "call" :
       bs_price = bs.bs_call(S, K, T, r, sigma, q)
    elif option_type == "put" :
       bs_price = bs.bs_put(S, K, T, r, sigma, q)
    
    assert abs(price - bs_price) < 3 * standard_error

#Verify the reproductibilty of the system (ie check against seed error)
@pytest.mark.parametrize("S, K, T, r, sigma, q", PARAMS)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_mc_reprod (S, K, T, r, sigma, q, option_type):
   price_1, standard_error_1, (c_low_1, c_high_1) = mc.mc_european(S, K, T, r, sigma, q , option_type=option_type, n_paths=N_PATHS, seed=SEED)
   price_2, standard_error_2, (c_low_2, c_high_2) = mc.mc_european(S, K, T, r, sigma, q , option_type=option_type, n_paths=N_PATHS, seed=SEED)
   assert price_1 == price_2 
   assert standard_error_1 == standard_error_2 
   assert (c_low_1, c_high_1) == (c_low_2, c_high_2)

# Verify that in MC, SE is two times less when number of variables is 2*2 = 4 times more
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_mc_se (option_type):
   _, se_large, _ = mc.mc_european(**ATM, option_type=option_type, n_paths=N_PATHS//4, seed=SEED)
   _, se_small, _ = mc.mc_european(**ATM, option_type=option_type, n_paths=N_PATHS, seed=SEED)
   
   assert se_large / 2 == pytest.approx(se_small, rel=0.05)

def test_value_error_option_type(): # Check if the option_type error works
   with pytest.raises(ValueError, match = "option_type"):
      mc.mc_european(**ATM, option_type='straddle', n_paths=N_PATHS, seed=SEED)

def test_value_error_n_paths(): # Check if the n_paths error works
   with pytest.raises(ValueError, match = "n_paths"):
      mc.mc_european(**ATM, option_type='call', n_paths=1, seed=SEED)
   