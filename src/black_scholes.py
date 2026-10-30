import numpy as np
from scipy.stats import norm

# Paramètres :
# S = spot
# K = Strike
# T = maturity (year)
# r = zero risk rate
# sigma = volatility
# q = dividends

# norm.cdf = F(x) = fonction de réparttion loi normale
# norm.pdf = phi = densité loi normale

def _d1_d2(S, K, T, r, sigma, q):

    d1 = (np.log(S/K) + (r - q + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    return d1, d2

def bs_call(S, K, T, r, sigma, q=0.0) :
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)

    call_price = S * np.exp(-q*T) * norm.cdf(d1) - K*np.exp(-r*T) * norm.cdf(d2)

    return(call_price)

def bs_put (S, K, T, r, sigma, q=0.0) :
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)

    put_price = K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

    return (put_price)

def bs_delta_call(S, K, T, r, sigma, q=0.0) :
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    delta_call = np.exp(-q*T)*norm.cdf(d1)
    return (delta_call)

def bs_delta_put(S, K, T, r, sigma, q=0.0) :
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    delta_put = -np.exp(-q*T) * norm.cdf(-d1)
    return (delta_put)

def bs_gamma (S, K, T, r, sigma, q=0.0) :
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    gamma = np.exp(-q*T) * norm.pdf(d1) / (S*sigma*np.sqrt(T))
    return (gamma)

def bs_vega (S, K, T, r, sigma, q=0.0) :
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    vega = S*np.exp(-q*T)*norm.pdf(d1)*np.sqrt(T)
    return(vega)

def bs_rho_call (S, K, T, r, sigma, q=0.0) :
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    rho_call = K*T*np.exp(-r*T)*norm.cdf(d2)
    return(rho_call)

def bs_rho_put (S, K, T, r, sigma, q=0.0) :
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    rho_put = -K*T*np.exp(-r*T)*norm.cdf(-d2)
    return (rho_put)

def bs_theta_call (S, K, T, r, sigma, q=0.0) :
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    theta_call = (-S*np.exp(-q*T)*norm.pdf(d1))*sigma/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2) + q*S*np.exp(-q*T)*norm.cdf(d1)
    return (theta_call)

def bs_theta_put (S, K, T, r, sigma, q=0.0) :
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    theta_put = (-S*np.exp(-q*T)*norm.pdf(d1))*sigma/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2) - q*S*np.exp(-q*T)*norm.cdf(-d1)
    return (theta_put)

if __name__ == "__main__":
    print(bs_put(S=100, K=100, T=1, r=0.05, sigma=0.2))