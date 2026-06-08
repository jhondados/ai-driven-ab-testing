"""Bayesian A/B testing engine."""
import numpy as np
from scipy.stats import beta as beta_dist
from typing import Tuple

class BayesianABTest:
    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.prior_alpha = prior_alpha
        self.prior_beta  = prior_beta
        self.control_alpha = prior_alpha
        self.control_beta  = prior_beta
        self.variant_alpha = prior_alpha
        self.variant_beta  = prior_beta

    def update(self, group: str, conversions: int, trials: int):
        if group == "control":
            self.control_alpha += conversions
            self.control_beta  += (trials - conversions)
        else:
            self.variant_alpha += conversions
            self.variant_beta  += (trials - conversions)

    def probability_variant_beats_control(self, n_samples: int = 100_000) -> float:
        control_samples = beta_dist.rvs(self.control_alpha, self.control_beta, size=n_samples)
        variant_samples = beta_dist.rvs(self.variant_alpha, self.variant_beta, size=n_samples)
        return float(np.mean(variant_samples > control_samples))

    def expected_loss(self, n_samples: int = 100_000) -> Tuple[float, float]:
        control_samples = beta_dist.rvs(self.control_alpha, self.control_beta, size=n_samples)
        variant_samples = beta_dist.rvs(self.variant_alpha, self.variant_beta, size=n_samples)
        loss_control = np.mean(np.maximum(variant_samples - control_samples, 0))
        loss_variant = np.mean(np.maximum(control_samples - variant_samples, 0))
        return float(loss_control), float(loss_variant)

    def make_decision(self, threshold: float = 0.95) -> dict:
        prob = self.probability_variant_beats_control()
        loss_c, loss_v = self.expected_loss()
        winner = "variant" if prob >= threshold else "control" if (1 - prob) >= threshold else "no_decision"
        return {"probability_variant_wins": prob, "expected_loss_if_control": loss_c,
                "expected_loss_if_variant": loss_v, "decision": winner, "confident": winner != "no_decision"}
