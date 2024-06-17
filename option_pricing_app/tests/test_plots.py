import unittest
import numpy as np
import matplotlib.pyplot as plt
from modules.plots import (
    plot_volatility_skew,
    plot_price_paths,
    plot_payoff_distribution,
)


class TestPlots(unittest.TestCase):

    def setUp(self):
        self.option_data = {
            "strike": np.arange(80, 120, 5),
            "volatility": np.random.uniform(0.15, 0.35, size=8),
        }
        self.price_paths = np.random.normal(100, 20, (252, 10))
        self.call_payoffs = np.maximum(self.price_paths[-1] - 100, 0)
        self.put_payoffs = np.maximum(100 - self.price_paths[-1], 0)

    def test_plot_price_paths(self):
        fig, ax = plt.subplots()
        plot_price_paths(self.price_paths)
        plt.close(fig)  # Close the plot to avoid displaying it during tests

    def test_plot_payoff_distribution(self):
        fig, ax = plt.subplots()
        plot_payoff_distribution(self.call_payoffs, "call")
        plot_payoff_distribution(self.put_payoffs, "put")
        plt.close(fig)  # Close the plot to avoid displaying it during tests


if __name__ == "__main__":
    unittest.main()
