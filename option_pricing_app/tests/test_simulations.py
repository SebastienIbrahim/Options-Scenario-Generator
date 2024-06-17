import unittest
import numpy as np
from modules.simulations import (
    generate_scenarios,
    calculate_call_payoffs,
    calculate_put_payoffs,
    simulate_option_value,
)


class TestSimulations(unittest.TestCase):

    def setUp(self):
        self.test_cases = [
            {
                "S0": 100,
                "r": 0.05,
                "sigma": 0.2,
                "T": 1,
                "num_steps": 252,
                "num_simulations": 10000,
                "K": 100,
            },
            {
                "S0": 50,
                "r": 0.03,
                "sigma": 0.1,
                "T": 0.5,
                "num_steps": 126,
                "num_simulations": 5000,
                "K": 55,
            },
            {
                "S0": 150,
                "r": 0.07,
                "sigma": 0.3,
                "T": 2,
                "num_steps": 504,
                "num_simulations": 20000,
                "K": 140,
            },
        ]

    def test_generate_scenarios(self):
        for case in self.test_cases:
            with self.subTest(case=case):
                S0, r, sigma, T, num_steps, num_simulations, K = (
                    case.values()
                )  # Unpack K as well
                price_paths = generate_scenarios(
                    S0, r, sigma, T, num_steps, num_simulations
                )

                # Check shape
                self.assertEqual(price_paths.shape, (num_steps + 1, num_simulations))

                # Check positive prices
                self.assertTrue(np.all(price_paths > 0))

                # Check if initial price is S0
                self.assertTrue(np.all(price_paths[0] == S0))

                # Check mean and std deviation
                mean_final_price = np.mean(price_paths[-1])
                std_final_price = np.std(price_paths[-1])
                expected_mean = S0 * np.exp(r * T)
                expected_std = S0 * np.exp(r * T) * sigma * np.sqrt(T)
                self.assertAlmostEqual(
                    mean_final_price, expected_mean, delta=expected_mean * 0.1
                )
                self.assertAlmostEqual(
                    std_final_price, expected_std, delta=expected_std * 0.1
                )

    def test_calculate_call_payoffs(self):
        for case in self.test_cases:
            with self.subTest(case=case):
                S0, r, sigma, T, num_steps, num_simulations, K = case.values()
                price_paths = generate_scenarios(
                    S0, r, sigma, T, num_steps, num_simulations
                )
                call_payoffs = calculate_call_payoffs(price_paths, K)

                # Check length
                self.assertEqual(len(call_payoffs), num_simulations)

                # Check non-negative payoffs
                self.assertTrue(np.all(call_payoffs >= 0))

                # Check payoffs calculation
                expected_payoffs = np.maximum(price_paths[-1] - K, 0)
                np.testing.assert_array_almost_equal(call_payoffs, expected_payoffs)

    def test_calculate_put_payoffs(self):
        for case in self.test_cases:
            with self.subTest(case=case):
                S0, r, sigma, T, num_steps, num_simulations, K = case.values()
                price_paths = generate_scenarios(
                    S0, r, sigma, T, num_steps, num_simulations
                )
                put_payoffs = calculate_put_payoffs(price_paths, K)

                # Check length
                self.assertEqual(len(put_payoffs), num_simulations)

                # Check non-negative payoffs
                self.assertTrue(np.all(put_payoffs >= 0))

                # Check payoffs calculation
                expected_payoffs = np.maximum(K - price_paths[-1], 0)
                np.testing.assert_array_almost_equal(put_payoffs, expected_payoffs)

    def test_simulate_option_value(self):
        option_call = {
            "type": "call",
            "S": 100,
            "K": 100,
            "T": 1,
            "r": 0.05,
            "sigma": 0.2,
        }
        option_put = {
            "type": "put",
            "S": 100,
            "K": 100,
            "T": 1,
            "r": 0.05,
            "sigma": 0.2,
        }
        condition = {"price_change": 0.1, "volatility_change": 0.05}

        call_value = simulate_option_value(option_call, condition)
        put_value = simulate_option_value(option_put, condition)

        self.assertGreater(call_value, 0)
        self.assertGreater(put_value, 0)

        # Check if the call value increases with the price change
        new_condition = {"price_change": 0.2, "volatility_change": 0.05}
        new_call_value = simulate_option_value(option_call, new_condition)
        self.assertGreater(new_call_value, call_value)

        # Check if the put value decreases with the price change
        self.assertLess(simulate_option_value(option_put, new_condition), put_value)


if __name__ == "__main__":
    unittest.main()
