"""Monte Carlo Barrier Options Project."""

######################################################################
# IMPORTS
######################################################################
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from warnings import warn

######################################################################
# CLOSED FORMULA IN THE BLACK-SCHOLES MODEL
######################################################################
def delta_plus(
    s: float,
    T: float,
    r: float,
    sigma: float
):
    """Compute delta_plus."""
    return (np.log(s) + (r + (sigma**2)/2) * T) / (sigma * (T**0.5))


def delta_minus(
    s: float,
    T: float,
    r: float,
    sigma: float
):
    """Compute delta_minus."""
    return (np.log(s) + (r - (sigma**2)/2) * T) / (sigma * (T**0.5))


def call(
    S: float, 
    K: float,
    T: float,
    r: float,
    sigma: float
):
    """Black-Scholes closed formula for a call option."""
    return S * norm.cdf(delta_plus(s=S/K, T=T, r=r, sigma=sigma)) - np.exp(-r*T) * K * norm.cdf(delta_minus(s=S/K, T=T, r=r, sigma=sigma))


def put(
    S: float, 
    K: float,
    T: float,
    r: float,
    sigma: float
):
    """Black-Scholes closed formula for a put option."""
    return np.exp(-r*T) * K * norm.cdf(-delta_minus(s=S/K, T=T, r=r, sigma=sigma)) - S * norm.cdf(-delta_plus(s=S/K, T=T, r=r, sigma=sigma))


# KNOCK-OUT OPTIONS

def up_out_call(
    S: float, 
    K: float,
    B: float,
    T: float,
    r: float,
    sigma: float
):
    """Black-Scholes closed formula for an up-and-out call option."""
    # If the price is breaking the barrier already, the value is zero
    if S >= B:
        return 0
    # If the up-and-out barrier is lower than the strike, the price is zero (the payoff can only be zero)
    if B <= K:
        return 0
    else:
        return call(S=S, K=K, T=T, r=r, sigma=sigma) - S * norm.cdf(delta_plus(s=S / B, T=T, r=r, sigma=sigma)) - (
        B * (B / S)**(2*r/(sigma**2))) * (norm.cdf(delta_plus(s=B**2/(K*S), T=T, r=r, sigma=sigma)) - norm.cdf(delta_plus(s=B / S, T=T, r=r, sigma=sigma))) - np.exp(-r*T) * K * (
            - norm.cdf(delta_minus(s=S / B, T=T, r=r, sigma=sigma)) - 
        ((S / B) ** (1-2*r/(sigma**2))) * (norm.cdf(delta_minus(s=B**2/(K*S), T=T, r=r, sigma=sigma)) - norm.cdf(delta_minus(s=B / S, T=T, r=r, sigma=sigma)))
        )


def up_out_put(
    S: float, 
    K: float,
    B: float,
    T: float,
    r: float,
    sigma: float
):
    """Black-Scholes closed formula for an up-and-out put option."""
    # If the price is breaking the barrier already, the value is zero
    if S >= B:
        return 0
    if B <= K:
        return S * (norm.cdf(delta_plus(s=S / B, T=T, r=r, sigma=sigma)) - 1 - (B / S) ** (1+2*r/(sigma**2)) * (norm.cdf(delta_plus(s=B/S, T=T, r=r, sigma=sigma)) - 1)) - K * np.exp(-r*T) * (
        norm.cdf(delta_minus(s=S / B, T=T, r=r, sigma=sigma)) - 1 - (S / B) ** (1-2*r/(sigma**2)) * (norm.cdf(delta_minus(s=B/S, T=T, r=r, sigma=sigma)) - 1)
        )
    else:
        return S * (norm.cdf(delta_plus(s=S / K, T=T, r=r, sigma=sigma)) - 1 - (B / S) ** (1+2*r/(sigma**2)) * (norm.cdf(delta_plus(s=B**2/(K*S), T=T, r=r, sigma=sigma)) - 1)) - K * np.exp(-r*T) * (
        norm.cdf(delta_minus(s=S / K, T=T, r=r, sigma=sigma)) - 1 - (S / B) ** (1-2*r/(sigma**2)) * (norm.cdf(delta_minus(s=B**2/(K*S), T=T, r=r, sigma=sigma)) - 1)
        )


def down_out_call(
    S: float, 
    K: float,
    B: float,
    T: float,
    r: float,
    sigma: float
):
    """Black-Scholes closed formula for an down-and-out call option."""
    # If the price is breaking the barrier already, the value is zero
    if S <= B:
        return 0
    if B <= K:
        return call(S=S, K=K, T=T, r=r, sigma=sigma) - S * (B/S)**(2*r/sigma**2) * call(S=B/S, K=K/B, T=T, r=r, sigma=sigma)
    else:
        return S * norm.cdf(delta_plus(s=S/B, T=T, r=r, sigma=sigma
        )) - K * np.exp(-r*T) * norm.cdf(delta_minus(s=S/B, T=T, r=r, sigma=sigma
        )) - B * (B/S)**(2*r/(sigma**2)) * norm.cdf(delta_plus(s=B/S, T=T, r=r, sigma=sigma
        )) + K * np.exp(-r*T) * (S/B)**(1-2*r/(sigma**2)) * norm.cdf(delta_minus(s=B/S, T=T, r=r, sigma=sigma))


def down_out_put(
    S: float, 
    K: float,
    B: float,
    T: float,
    r: float,
    sigma: float
):
    """Black-Scholes closed formula for an down-and-out put option."""
    # If the price is breaking the barrier already, the value is zero
    if S <= B:
        return 0
    if B >= K:
        return 0
    else:
        return put(S=S, K=K, T=T, r=r, sigma=sigma) + S * norm.cdf(-delta_plus(s=S/B, T=T, r=r, sigma=sigma
        )) - B * (B/S)**(2*r/(sigma**2)) * (norm.cdf(delta_plus(s=B**2/(K*S), T=T, r=r, sigma=sigma)) - norm.cdf(delta_plus(s=B/S, T=T, r=r, sigma=sigma))
        ) - K * np.exp(-r*T) * norm.cdf(-delta_minus(s=S/B, T=T, r=r, sigma=sigma
        )) + K * np.exp(-r*T) * (S/B)**(1-2*r/(sigma**2)) * (norm.cdf(delta_minus(s=B**2/(K*S), T=T, r=r, sigma=sigma)) - norm.cdf(delta_minus(s=B/S, T=T, r=r, sigma=sigma))
        )
    

# KNOCK-IN OPTIONS

def down_in_call(
    S: float, 
    K: float,
    B: float,
    T: float,
    r: float,
    sigma: float
):
    """
    Black-Scholes closed formula for an down-and-in call option.
    c_di = c - c_do
    """
    return call(S=S, K=K, T=T, r=r, sigma=sigma) - down_out_call(S=S, K=K, B=B, T=T, r=r, sigma=sigma)


def down_in_put(
    S: float, 
    K: float,
    B: float,
    T: float,
    r: float,
    sigma: float
):
    """
    Black-Scholes closed formula for an down-and-in put option.
    p_di = p - p_do
    """
    return put(S=S, K=K, T=T, r=r, sigma=sigma) - down_out_put(S=S, K=K, B=B, T=T, r=r, sigma=sigma)


def up_in_call(
    S: float, 
    K: float,
    B: float,
    T: float,
    r: float,
    sigma: float
):
    """
    Black-Scholes closed formula for an up-and-in call option.
    c_ui = c - c_uo
    """
    return call(S=S, K=K, T=T, r=r, sigma=sigma) - up_out_call(S=S, K=K, B=B, T=T, r=r, sigma=sigma)


def up_in_put(
    S: float, 
    K: float,
    B: float,
    T: float,
    r: float,
    sigma: float
):
    """
    Black-Scholes closed formula for an up-and-in call option.
    p_ui = p - p_uo
    """
    return put(S=S, K=K, T=T, r=r, sigma=sigma) - up_out_put(S=S, K=K, B=B, T=T, r=r, sigma=sigma)


# PLOTS
MAPPING_FUNCTIONS = {
    "call": {
        "up_in": up_in_call,
        "up_out": up_out_call,
        "down_in": down_in_call,
        "down_out": down_out_call,
    },
    "put": {
        "up_in": up_in_put,
        "up_out": up_out_put,
        "down_in": down_in_put,
        "down_out": down_out_put,
    }
}

def plot_price_surface(
    K: float,
    B: float,
    r: float,
    sigma: float,
    type_option: str,
    type_barrier: str,
    elev: float = 10,
    azim: float = 10,
    ):
    """Plot the price surface for the barrier option as a function of T and S."""
    # Get the appropriate function to compute the price
    f = MAPPING_FUNCTIONS[type_option][type_barrier]
    # List of S
    l_S = np.linspace(50, 90, 50)
    # List of T
    l_T = np.linspace(1, 120, 50) / 365

    # Meshgrid for the surface
    S, T = np.meshgrid(l_S, l_T)

    # Compute the option price
    price = np.vectorize(f)(S=S, K=K, B=B, T=T, r=r, sigma=sigma)

    # Plot
    fig = plt.figure(figsize=(6, 8), facecolor="white")
    ax = fig.add_subplot(projection='3d')
    # ax = plt.axes(projection='3d')
    ax.plot_surface(S, T, price, edgecolors='r', lw=0.6, alpha=0)
    str_title = type_barrier + " " + type_option + " price surface (K=" + str(K) + ", B=" + str(B) + ")"
    # ax.set_title(str_title)
    # Change the X-ticks labels
    labels = ax.get_yticks().tolist()
    labels = [int(float(l) * 365) for l in labels]
    ax.set_yticklabels(labels)
    # Set axis labels
    ax.set(xlabel="S ($)", ylabel="T (days)", zlabel="Price ($)")
    # Change the azimuth and elevation angles ("camera position")
    ax.view_init(elev=elev, azim=azim)
    return price


def plot_price_vs_volatility(
    S: float,
    K: float,
    B: float,
    T: float,
    r: float,
    type_option: str,
    type_barrier: str
    ):
    """Plot the option price as a function of volatility"""
    # Get the appropriate function to compute the price
    f = MAPPING_FUNCTIONS[type_option][type_barrier]
    # List of sigma
    l_sigma = np.linspace(0.001, 1, 100)

    # Compute the option price
    price = np.vectorize(f)(S=S, K=K, B=B, T=T, r=r, sigma=l_sigma)

    high = S - K*np.exp(-r*T)
    low = S - B
    # Plot
    fig, ax = plt.subplots()
    ax.plot(l_sigma, price, c="r")
    ax.axhline(high, c="b")
    ax.axhline(low, c="purple")
    ax.grid()
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Price ($)')
    str_title = type_barrier + " " + type_option + " price VS volatility (K=" + str(K) + ", B=" + str(B) + ")"
    ax.set_title(str_title)


######################################################################
# MONTE CARLO PRICING
######################################################################

# Discrete Euler Scheme ----------------------------------------------
def indicatrice_outside_domaine(X, D : tuple):
    borne_min = min(D)
    borne_max = max(D)
    
    return np.where((borne_min < X) & (X < borne_max), False, True)


def b_black_scholes(X, b):
    return b * X

def sigma_black_scholes(X, sigma):
    return sigma * X


def discrete_barrier(
        x : float, 
        T : float, 
        N : int, 
        M : int, 
        D : tuple, 
        r: float, 
        sigma: float
    ):
    """
    Compute the terminal value of the process XT and an array indicating if it went outside the domain.
    This version uses the b and sigma functions from the Black-Scholes model.
    """
    X = np.ones(M) * x
    dt = T/N
    
    borne_min = min(D)
    borne_max = max(D)
    
    if borne_min <= x <= borne_max:
        went_outside = np.zeros(M).astype('bool')
    else:
        went_outside = np.ones(M).astype('bool')
        warn("Warning : The starting position of the process is outside the domain.")
        
    for i in range(N):
        X = X + b_black_scholes(X=X, b=r) * dt + sigma_black_scholes(X=X, sigma=sigma) * np.random.normal(size=M) * np.sqrt(dt)
        
        went_outside = indicatrice_outside_domaine(X, D) | went_outside   # This is the logical OR
        
    return X, went_outside


def call_price_mc(r, T, XT, K, went_outside):
    premium = np.maximum((1-went_outside).astype('int') * (XT - K), 0)
    price = np.mean(premium)
    actualized_price = np.exp(-r*T) * price
    
    return actualized_price


def put_price_mc(r, T, XT, K, went_outside):
    premium = np.maximum((1-went_outside).astype('int') * (K - XT), 0)
    price = np.mean(premium)
    actualized_price = np.exp(-r*T) * price
    
    return actualized_price


def mc_pricer_knock_out_discrete(
    x: float,
    K: float,
    B: float,
    T: float,
    N: int,
    M: int,
    r: float,
    sigma: float,
    type_option: str,
    type_barrier: str
    ):
    """Monte-Carlo pricing of a knock-out barrier call using the discrete Euler Scheme."""
    # Create the domain
    if type_barrier == "down_out":
        D = (B, np.inf)
    else:
        D = (-np.inf, B)
    # Get the terminal prices XT and the went_outside array
    XT, went_outside = discrete_barrier(x, T, N, M, D, r=r, sigma=sigma)
    # Return the call price
    if type_option == "call":
        return call_price_mc(r=r, T=T, XT=XT, K=K, went_outside=went_outside)
    else:
        return put_price_mc(r=r, T=T, XT=XT, K=K, went_outside=went_outside)

# Continuous Euler Scheme ----------------------------------------------
def p(z1, z2, dt, D, sigma):
    """Return the probability that the process stays in D during dt knowing that it starts at z1 and ends at z2."""
    # D = [a, b]
    a = min(D)
    b = max(D)
    
    if np.isneginf(a):
        num = (b - z1)*(b - z2)
        denom = sigma_black_scholes(X=z1, sigma=sigma) * sigma_black_scholes(X=z1, sigma=sigma) * dt
        p = 1 - np.exp(-2 * num/denom)
        p = np.where(b > z1, 1, 0) * np.where(b > z2, 1, 0) * p
    
    elif np.isposinf(b):
        num = (a - z1)*(a - z2)
        denom = sigma_black_scholes(X=z1, sigma=sigma) * sigma_black_scholes(X=z1, sigma=sigma) * dt
        p = 1 - np.exp(-2 * num/denom)
        
        p = np.where(z1 > a, 1, 0) * np.where(z2 > a, 1, 0) * p
    
    else:
        raise ValueError("D must have one of its boundaries set as +/- infinity.")
    
    return p


def continuous_barrier(
        x : float, 
        T : float, 
        N : int, 
        M : int, 
        D : tuple, 
        r : int, 
        sigma : int
    ):

    X = np.ones(M) * x
    dt = T/N
    probas = np.ones(M).astype('float32')
    
    borne_min = min(D)
    borne_max = max(D)
    
    if borne_min <= x <= borne_max:
        went_outside = np.zeros(M).astype('bool')
    else:
        went_outside = np.ones(M).astype('bool')
        warn("Warning : The starting position of the process is outside the domain.")
        
    for i in range(N):
        past_X = X
        X = X + b_black_scholes(X=X, b=r) * dt + sigma_black_scholes(X=X, sigma=sigma) * np.random.normal(size=M) * np.sqrt(dt)
        
        bernoulli_p = 1 - p(past_X, X, dt, D, sigma)
        probas *= p(past_X, X, dt, D, sigma)
        #went_outside_between = np.random.binomial(1, bernoulli_p, size=M)
        
        went_outside = indicatrice_outside_domaine(X, D) | went_outside   # For the particular discrete values
        #went_outside = went_outside | went_outside_between                # Between those values
    
    return X, probas


def mc_pricer_knock_out_continuous(
    x: float,
    K: float,
    B: float,
    T: float,
    N: int,
    M: int,
    r: float,
    sigma: float,
    type_option: str,
    type_barrier: str
    ):
    """Monte-Carlo pricing of a knock-out barrier call using the continous Euler Scheme."""
    # Create the domain
    if type_barrier == "down_out":
        D = (B, np.inf)
    else:
        D = (-np.inf, B)
    # Get the terminal prices XT and the went_outside array
    XT, probas = continuous_barrier(x, T, N, M, D, r=r, sigma=sigma)
    
    # Return the call price
    if type_option == "call":
        #return call_price_mc(r=r, T=T, XT=XT, K=K, went_outside=went_outside)
        
        premium = probas * np.maximum(XT - K, 0)
        price = np.mean(premium)
        actualized_price = np.exp(-r*T) * price
        
        return actualized_price
    else:
        #return put_price_mc(r=r, T=T, XT=XT, K=K, went_outside=went_outside)
        
        premium = probas * np.maximum(K - XT, 0)
        price = np.mean(premium)
        actualized_price = np.exp(-r*T) * price

        return actualized_price
        
        
    


######################################################################
# CONPARISON CLOSED FORMULAS VS MONTE CARLO
######################################################################
def compare_prices_out(
    S: float,
    K: float,
    T: float,
    N: int,
    M: int,
    r: float,
    sigma: float,
    B: float,
    type_option: str,
    type_barrier: str
):
    """Compare the prices of knock-out options given by the closed formula, discrete and continuous Monte Carlo."""
    # Create the domain
    if type_barrier == "down_out":
        D = (B, np.inf)
    else:
        D = (-np.inf, B)
    # Compute the Monte Carlo prices
    price_mc_discrete = mc_pricer_knock_out_discrete(
        x=S,
        K=K,
        B=B,
        T=T,
        N=N,
        M=M,
        r=r,
        sigma=sigma,
        type_option=type_option,
        type_barrier=type_barrier
    )
    price_mc_continuous = mc_pricer_knock_out_continuous(
        x=S,
        K=K,
        B=B,
        T=T,
        N=N,
        M=M,
        r=r,
        sigma=sigma,
        type_option=type_option,
        type_barrier=type_barrier
    )
    # Get the appropriate closed formula function
    f = MAPPING_FUNCTIONS[type_option][type_barrier]
    # Compute the closed formula price
    price_closed = f(
        S=S, 
        K=K,
        B=B,
        T=T,
        r=r,
        sigma=sigma
    )
    return price_closed, price_mc_discrete, price_mc_continuous


def convergence_out(
    S: float,
    K: float,
    T: float,
    M: int,
    r: float,
    sigma: float,
    B: float,
    type_option: str,
    type_barrier: str,
    nb_points: int
):
    """
    Illustrate the convergence of the numerical scheme with regards to N (number of time steps).
    Plot the strong and weak errors for the scheme.
    """
    # List of N values
    l_N = np.logspace(1, 3, nb_points).astype(int)
    # List of strong and weak errors for each m values
    l_error_d = np.empty(nb_points)
    l_error_c = np.empty(nb_points)
    for k in range(nb_points):
        N = l_N[k]
        # Get the different prices
        price_closed, price_mc_discrete, price_mc_continuous = compare_prices_out(
            S=S,
            K=K,
            T=T,
            N=N,
            M=M,
            r=r,
            sigma=sigma,
            B=B,
            type_option=type_option,
            type_barrier=type_barrier
        )
        # Compute the errors
        l_error_d[k] = np.abs(price_closed-price_mc_discrete)
        l_error_c[k] = np.abs(price_closed-price_mc_continuous)
       
    print(l_error_c)
    # Plot the errors
    fig, axs = plt.subplots(2, figsize=(12, 8))
    axs[0].plot(l_N, l_error_d)
    axs[1].plot(l_N, l_error_c)
    # Compute lines with slope 1 or 1/2 to illustrate the convergence
    slope_d = 0.5
    slope_c = 1
    # The slope lists are calibrated so that they are close to the plot (0.9 parameter)
    l_slope_d = ((1 / l_N) ** slope_d) * l_error_d[0] / ((1 / l_N) ** slope_d)[0] * 0.9
    l_slope_c = ((1 / l_N) ** slope_c) * l_error_c[0] / ((1 / l_N) ** slope_c)[0] * 0.9
    # Plot the lines for the slope
    axs[0].plot(l_N, l_slope_d, linestyle="--", color="red")
    axs[1].plot(l_N, l_slope_c, linestyle="--", color="red")
    # Set the plot titles
    axs[0].set_title("Discrete Euler error VS number of time steps")
    axs[1].set_title("Continuous Euler error VS number of time steps")
    # Set the x and y-axes scales
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    # Grid for the plot
    axs[0].grid()
    axs[1].grid()
    # Y-axis title
    axs[0].set_ylabel("Weak error ($) (log)")
    axs[1].set_ylabel("Weak error ($) (log)")
    # X-axis title
    axs[0].set_xlabel("Number of time steps (N) (log)")
    axs[1].set_xlabel("Number of time steps (N) (log)")
    # Legend
    axs[0].legend(["Discrete error", "Slope " + str(-slope_d)])
    axs[1].legend(["Continuous error", "Slope " + str(-slope_c)])
    # Tight layout
    fig.tight_layout()
