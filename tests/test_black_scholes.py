import pytest
from black_scholes import bs_call, bs_put
import numpy as np

def test_call_reference_value():
    assert bs_call(100, 100, 1, 0.05, 0.2) == pytest.approx(10.4506, abs=1e-4)

def test_put_reference_value():
    assert bs_put(100, 100, 1, 0.05, 0.2) == pytest.approx(5.5735, abs = 1e-4)


@pytest.mark.parametrize("S, K, T, r, sigma, q",[
    (100, 100, 1.0, 0.05, 0.20, 0.00),   # ATM, sans dividende
    (100, 100, 1.0, 0.05, 0.20, 0.02),   # ATM, avec dividende
    (100,  80, 0.5, 0.03, 0.25, 0.01),   # call ITM, maturité courte
    (100, 130, 2.0, 0.04, 0.30, 0.03),   # call OTM, maturité moyenne
    (100, 100, 5.0, 0.02, 0.15, 0.04),   # longue maturité, vol faible
    (50,   60, 0.25, 0.00, 0.60, 0.00),  # taux nul, vol très élevée
])
def test_put_call_parity(S, K, T, r, sigma, q):
    call = bs_call(S, K, T, r, sigma, q)
    put = bs_put(S, K, T, r, sigma, q)
    assert call - put == pytest.approx(S*np.exp(-q*T) - K*np.exp(-r*T))