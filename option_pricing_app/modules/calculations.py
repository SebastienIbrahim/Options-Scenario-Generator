import numpy as np
from scipy.stats import norm


# Function to calculate d1
def d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


# Function to calculate d2
def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


# Function to calculate call option price
def call_price(S, K, T, r, sigma):
    return S * norm.cdf(d1(S, K, T, r, sigma)) - K * np.exp(-r * T) * norm.cdf(
        d2(S, K, T, r, sigma)
    )


# Function to calculate put option price
def put_price(S, K, T, r, sigma):
    return K * np.exp(-r * T) * norm.cdf(-d2(S, K, T, r, sigma)) - S * norm.cdf(
        -d1(S, K, T, r, sigma)
    )


# Function to calculate call delta
def call_delta(S, K, T, r, sigma):
    return norm.cdf(d1(S, K, T, r, sigma))


# Function to calculate put delta
def put_delta(S, K, T, r, sigma):
    return call_delta(S, K, T, r, sigma) - 1


# Function to calculate gamma
def gamma(S, K, T, r, sigma):
    return norm.pdf(d1(S, K, T, r, sigma)) / (S * sigma * np.sqrt(T))


# Function to calculate vega
def vega(S, K, T, r, sigma):
    return (
        S * norm.pdf(d1(S, K, T, r, sigma)) * np.sqrt(T) / 100
    )  # Divide by 100 to get vega in percentage


# Function to calculate call theta
def call_theta(S, K, T, r, sigma):
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    term1 = -S * norm.pdf(d1_val) * sigma / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * norm.cdf(d2_val)
    return (term1 - term2) / 365  # Divide by 365 to get theta per day


# Function to calculate put theta
def put_theta(S, K, T, r, sigma):
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    term1 = -S * norm.pdf(d1_val) * sigma / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * norm.cdf(-d2_val)
    return (term1 + term2) / 365  # Divide by 365 to get theta per day


# Function to calculate call rho
def call_rho(S, K, T, r, sigma):
    d2_val = d2(S, K, T, r, sigma)
    return (
        K * T * np.exp(-r * T) * norm.cdf(d2_val) / 100
    )  # Divide by 100 to get rho in percentage


# Function to calculate put rho
def put_rho(S, K, T, r, sigma):
    d2_val = d2(S, K, T, r, sigma)
    return (
        -K * T * np.exp(-r * T) * norm.cdf(-d2_val) / 100
    )  # Divide by 100 to get rho in percentage
