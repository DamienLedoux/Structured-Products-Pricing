import numpy as np
import black_scholes as bs


def delta_hedge_pnl(S, K, T, r, sigma_impl, c=0.0, q=0.0) :
    """Final P&L of a short call hedged in delta at each grid date.

    Parameters
    ----------
    S : ndarray, shape (n_paths, n_steps + 1)
        Simulated prices (real-world dynamics), e.g. from simulate_gbm_paths.
    K, T, r : strike, maturity (years), continuous risk-free rate.
    sigma_impl : volatility used for the premium and the delta.
    c : proportional transaction cost (fraction of traded notional).
    q : dividend yield, must be 0 for now.

    Returns
    -------
    pnl : ndarray, shape (n_paths,)
        Seller's P&L at maturity, NOT discounted. Multiply by exp(-r*T)
        to compare with the premium. No cost on the final liquidation.
    """

    if q!=0 :
        raise ValueError (f"Dividends are not yet taken into account by the model. q must be 0. q received :{q}")
    
    n_paths, n_cols = S.shape
    if n_cols < 2 :
        raise ValueError(f"S must have >= 2 columns. number of columns of S :{n_cols}")

    n_steps = n_cols - 1
    dt = T / n_steps
    S0 = S[0,0]

    C0 = bs.bs_call(S0, K, T, r, sigma_impl, q=q)
    delta0 = bs.bs_delta_call(S0, K, T, r, sigma_impl, q=q)
    cash0 = C0 - delta0 * S0

    cash = np.full(n_paths, cash0)
    delta = np.full(n_paths, delta0)

    for k in range (1, n_steps):
        # capitalize cash
        cash = cash * np.exp(r*dt)

        # actualize delta
        delta_new = bs.bs_delta_call(S[:, k], K, T-k*dt, r, sigma_impl, q=q)
        d = delta_new - delta

        # hedging costs (delta hedge + processing fee)
        cash = cash - d*S[:, k] - c*np.abs(d) * S[:, k]
        delta = delta_new

    # last actualization (from step n-1 to n = n_cols)
    S_T = S[:, -1]
    cash = cash * np.exp(r * dt)

    # pnl = cash + value of our hedge - payoff that we owe
    pnl = cash + delta * S_T - np.maximum(S_T - K, 0)
    
    return pnl
