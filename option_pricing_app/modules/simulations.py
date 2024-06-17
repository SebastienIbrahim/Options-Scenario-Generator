import numpy as np
from modules.calculations import call_price, put_price
from scipy.optimize import brentq
from scipy.stats import norm


# Function to generate price paths using Monte Carlo simulation
def generate_scenarios(S0, r, sigma, T, num_steps, num_simulations):
    dt = T / num_steps
    price_paths = np.zeros((num_steps + 1, num_simulations))
    price_paths[0] = S0
    for t in range(1, num_steps + 1):
        z = np.random.standard_normal(num_simulations)
        price_paths[t] = price_paths[t - 1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        )
    return price_paths


# Function to calculate call payoffs
def calculate_call_payoffs(price_paths, K):
    return np.maximum(price_paths[-1] - K, 0)


# Function to calculate put payoffs
def calculate_put_payoffs(price_paths, K):
    return np.maximum(K - price_paths[-1], 0)


# Function to simulate option value
def simulate_option_value(option, condition):
    S = option["S"] * (1 + condition["price_change"])
    sigma = option["sigma"] + condition["volatility_change"]
    if option["type"] == "call":
        return call_price(S, option["K"], option["T"], option["r"], sigma)
    elif option["type"] == "put":
        return put_price(S, option["K"], option["T"], option["r"], sigma)


# Function to simulate different market scenarios
def simulate_scenario(option_portfolio, market_conditions):
    simulated_values = []
    for condition in market_conditions:
        portfolio_value = 0
        for option in option_portfolio:
            value = simulate_option_value(option, condition)
            portfolio_value += value
        simulated_values.append(portfolio_value)
    return simulated_values


def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
