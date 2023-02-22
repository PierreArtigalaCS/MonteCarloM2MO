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


def call_price_mc(r, T, XT, K, went_outside, knock: str = "out"):
    # If it is a knock-out option, went_outside = 1 <=> premium = 0
    if knock == "out":
        actualized_premium = np.exp(-r*T) * np.maximum((1-went_outside).astype('int') * (XT - K), 0)
    else:
        actualized_premium = np.exp(-r*T) * np.maximum(went_outside.astype('int') * (XT - K), 0)
    price = np.mean(actualized_premium)
    # Compute the standard deviation of the price
    std = actualized_premium.std()
    return price, std


def put_price_mc(r, T, XT, K, went_outside, knock: str = "out"):
    # If it is a knock-out option, went_outside = 1 <=> premium = 0
    if knock == "out":
        actualized_premium = np.exp(-r*T) * np.maximum((1-went_outside).astype('int') * (K - XT), 0)
    else:
        actualized_premium = np.exp(-r*T) * np.maximum(went_outside.astype('int') * (K - XT), 0)
    price = np.mean(actualized_premium)
    # Compute the standard deviation of the price
    std = actualized_premium.std()
    return price, std
    

def mc_pricer_discrete(
    x: float,
    K: float,
    B: float,
    T: float,
    N: int,
    M: int,
    r: float,
    sigma: float,
    type_option: str,
    direction: str,
    knock: str
    ):
    """Monte-Carlo pricing of a barrier option using the discrete Euler Scheme."""
    # Create the domain
    if direction == "down":
        D = (B, np.inf)
    else:
        D = (-np.inf, B)
    # Get the terminal prices XT and the went_outside array
    XT, went_outside = discrete_barrier(x, T, N, M, D, r=r, sigma=sigma)
    # Return the call price
    if type_option == "call":
        return call_price_mc(r=r, T=T, XT=XT, K=K, went_outside=went_outside, knock=knock)
    else:
        return put_price_mc(r=r, T=T, XT=XT, K=K, went_outside=went_outside, knock=knock)

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
        probas *= p(past_X, X, dt, D, sigma)
    return X, probas
        
        
def mc_pricer_continuous(
    x: float,
    K: float,
    B: float,
    T: float,
    N: int,
    M: int,
    r: float,
    sigma: float,
    type_option: str,
    direction: str,
    knock: str
    ):
    """Monte-Carlo pricing of a barrier option using the continous Euler Scheme."""
    # Create the domain
    if direction == "down":
        D = (B, np.inf)
    else:
        D = (-np.inf, B)
    # Get the terminal prices XT and the went_outside array
    XT, probas = continuous_barrier(x, T, N, M, D, r=r, sigma=sigma)
    if knock == "in":
        probas = 1-probas
    # Return the call price
    if type_option == "call":
        premium = probas * np.maximum(XT - K, 0)
        actualized_premium = np.exp(-r*T) * premium
        price = np.mean(actualized_premium)
    else:
        premium = probas * np.maximum(K - XT, 0)
        actualized_premium = np.exp(-r*T) * premium
        price = np.mean(actualized_premium)

    # Compute the standard deviation of the price
    std = actualized_premium.std()
    return price, std



######################################################################
# COMPARISON CLOSED FORMULAS VS MONTE CARLO
######################################################################
def compare_prices(
    S: float,
    K: float,
    T: float,
    N: int,
    M: int,
    r: float,
    sigma: float,
    B: float,
    type_option: str,
    direction: str,
    knock: str
):
    """Compare the prices of knock-out options given by the closed formula, discrete and continuous Monte Carlo."""
    # Create the domain
    if direction == "down":
        D = (B, np.inf)
    else:
        D = (-np.inf, B)
    # Compute the Monte Carlo prices
    price_mc_discrete, std_d = mc_pricer_discrete(
        x=S,
        K=K,
        B=B,
        T=T,
        N=N,
        M=M,
        r=r,
        sigma=sigma,
        type_option=type_option,
        direction=direction,
        knock=knock
    )
    price_mc_continuous, std_c = mc_pricer_continuous(
        x=S,
        K=K,
        B=B,
        T=T,
        N=N,
        M=M,
        r=r,
        sigma=sigma,
        type_option=type_option,
        direction=direction,
        knock=knock
    )
    # Get the appropriate closed formula function
    f = MAPPING_FUNCTIONS[type_option][direction + "_" + knock]
    # Compute the closed formula price
    price_closed = f(
        S=S, 
        K=K,
        B=B,
        T=T,
        r=r,
        sigma=sigma
    )
    return {"price": [price_closed, price_mc_discrete, price_mc_continuous], "std": [std_d, std_c]}


def convergence(
    S: float,
    K: float,
    T: float,
    M: int,
    r: float,
    sigma: float,
    B: float,
    type_option: str,
    direction: str,
    knock: str,
    nb_points: int
):
    """
    Illustrate the convergence of the numerical scheme with regards to N (number of time steps).
    Plot the weak error against N.
    """
    # List of N values
    l_N = np.logspace(1, 3, nb_points).astype(int)
    # List of strong and weak errors for each m values
    l_error_d = np.empty(nb_points)
    l_error_c = np.empty(nb_points)
    for k in range(nb_points):
        N = l_N[k]
        # Get the different prices
        result = compare_prices(
            S=S,
            K=K,
            T=T,
            N=N,
            M=M,
            r=r,
            sigma=sigma,
            B=B,
            type_option=type_option,
            direction=direction,
            knock=knock
        )
        price_closed, price_mc_discrete, price_mc_continuous = result["price"]
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
    # Title
    fig.suptitle("Convergence for " + direction + "_" + knock + " " + type_option + " (M=" + str(M) + ", S=" + str(S) + ", K=" + str(K) + ", B=" + str(B) + ")")
    # Tight layout
    fig.tight_layout()


def convergence_price(
    S: float,
    K: float,
    T: float,
    M: int,
    r: float,
    sigma: float,
    B: float,
    type_option: str,
    direction: str,
    knock: str,
    nb_points: int,
    alpha: float
):
    """
    Illustrate the convergence of the numerical scheme with regards to N (number of time steps).
    Plot the price and its confidence interval (level alpha) against N.
    """
    # Compute the normal quantile z_{alpha/2}
    z = norm.ppf(1-alpha/2)
    # List of N values
    l_N = np.linspace(10, 1000, nb_points).astype(int)
    # List of strong and weak errors for each m values
    l_price_closed = np.empty(nb_points)
    l_price_d = np.empty(nb_points)
    l_price_c = np.empty(nb_points)
    high_d = np.empty(nb_points)
    high_c = np.empty(nb_points)
    low_d = np.empty(nb_points)
    low_c = np.empty(nb_points)
    for k in range(nb_points):
        N = l_N[k]
        # Get the different prices
        result = compare_prices(
            S=S,
            K=K,
            T=T,
            N=N,
            M=M,
            r=r,
            sigma=sigma,
            B=B,
            type_option=type_option,
            direction=direction,
            knock=knock
        )
        price_closed, price_mc_d, price_mc_c = result["price"]
        std_d, std_c = result["std"]
        # Compute the errors
        l_price_closed[k] = price_closed
        l_price_d[k] = price_mc_d
        l_price_c[k] = price_mc_c
        # Compute the high and low borders of the confidence interval
        high_d[k] = price_mc_d + z * std_d / (M**0.5)
        high_c[k] = price_mc_c + z * std_c / (M**0.5)
        low_d[k] = price_mc_d - z * std_d / (M**0.5)
        low_c[k] = price_mc_c - z * std_c / (M**0.5)
       
    # Plot the errors
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(l_N, l_price_closed, c="k", ls="--")
    ax.plot(l_N, l_price_d, c="b")
    ax.plot(l_N, l_price_c, c="r")
    # Plot the confidence intervals
    ax.fill_between(l_N, low_d, high_d, alpha=0.1, color="b")
    ax.fill_between(l_N, low_c, high_c, alpha=0.1, color="r")
    # Grid for the plot
    ax.grid()
    # Y-axis title
    ax.set_ylabel("Price ($)")
    # X-axis title
    ax.set_xlabel("Number of time steps (N)")
    # Legend
    str_ci = str(int((1-alpha)*100)) + "% CI"
    ax.legend(["Closed formula", "Discrete Euler", "Continuous Euler", str_ci+" discrete", str_ci+" continuous"])
    # Title
    fig.suptitle("Price convergence for " + direction + "_" + knock + " " + type_option + " (M=" + str(M) + ", S=" + str(S) + ", K=" + str(K) + ", B=" + str(B) + ")")
    # Tight layout
    fig.tight_layout()
    # Save the figure
    file_name = "cv-price-" + direction + "-" + knock + "-" + type_option + ".png"
    fig.savefig("Figures/"+file_name)