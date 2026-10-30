import numpy as np
from scipy.stats import norm

def mc_european(S, K, T, r, sigma, q=0.0, option_type="call", n_paths=100_000, seed=None) :

    if n_paths < 2 :
        raise ValueError ("n_paths doit être >= 2")

    # Step 0 : seed
    rng = np.random.default_rng(seed)

    # Step 1 : N variables --> N(0,1)
    z = rng.standard_normal(size=n_paths)

    # Step 2 : final Price of each variables
    S_T = S*np.exp((r-q-sigma**2/2)*T + sigma*np.sqrt(T)*z)

    # Step 3 : le payoff dépend du type
    if option_type == "call":
        payoff = np.maximum(S_T - K, 0)
    elif option_type == "put":
        payoff = np.maximum(K - S_T, 0)
    else:
        raise ValueError(f"option_type doit être 'call' ou 'put', reçu : {option_type!r}")

    # Step 4 : actualize
    actualized_payoffs = payoff * np.exp(-r*T)

    # Step 5 : today price
    price = actualized_payoffs.mean()

    # Step 6 : precision
    std = actualized_payoffs.std(ddof=1)
    standard_error = std / np.sqrt(n_paths)
    z_alpha = norm.ppf(0.975)
    ci_low, ci_high = price - z_alpha*standard_error, price + z_alpha*standard_error

    # Step 7 : return
    return price, standard_error, (ci_low, ci_high)




