import unittest
from modules.calculations import (
    call_price,
    put_price,
    call_delta,
    put_delta,
    gamma,
    vega,
    call_theta,
    put_theta,
    call_rho,
    put_rho,
)


class TestCalculations(unittest.TestCase):

    def setUp(self):
        self.S = 100
        self.K = 100
        self.T = 1
        self.r = 0.05
        self.sigma = 0.2

    def test_call_price(self):
        price = call_price(self.S, self.K, self.T, self.r, self.sigma)
        self.assertAlmostEqual(price, 10.4506, places=4)

    def test_put_price(self):
        price = put_price(self.S, self.K, self.T, self.r, self.sigma)
        self.assertAlmostEqual(price, 5.5735, places=4)

    def test_call_delta(self):
        delta = call_delta(self.S, self.K, self.T, self.r, self.sigma)
        self.assertAlmostEqual(delta, 0.6368, places=4)

    def test_put_delta(self):
        delta = put_delta(self.S, self.K, self.T, self.r, self.sigma)
        self.assertAlmostEqual(delta, -0.3632, places=4)

    def test_gamma(self):
        gamma_value = gamma(self.S, self.K, self.T, self.r, self.sigma)
        self.assertAlmostEqual(gamma_value, 0.01876, places=5)

    def test_vega(self):
        vega_value = vega(self.S, self.K, self.T, self.r, self.sigma)
        self.assertAlmostEqual(vega_value, 0.3752, places=4)

    def test_call_theta(self):
        theta = call_theta(self.S, self.K, self.T, self.r, self.sigma)
        self.assertAlmostEqual(theta, -0.0176, places=4)

    def test_put_theta(self):
        theta = put_theta(self.S, self.K, self.T, self.r, self.sigma)
        self.assertAlmostEqual(theta, -0.0045, places=4)

    def test_call_rho(self):
        rho = call_rho(self.S, self.K, self.T, self.r, self.sigma)
        self.assertAlmostEqual(rho, 0.5323, places=4)

    def test_put_rho(self):
        rho = put_rho(self.S, self.K, self.T, self.r, self.sigma)
        self.assertAlmostEqual(rho, -0.4189, places=4)


if __name__ == "__main__":
    unittest.main()
