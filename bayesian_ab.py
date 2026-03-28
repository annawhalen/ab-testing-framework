"""
bayesian_ab.py
--------------
Bayesian A/B testing for proportion metrics.
Uses Beta-Binomial conjugate model — no MCMC required, runs instantly.
Outputs probability that treatment beats control and expected loss.
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass


@dataclass
class BayesianResult:
    prob_treatment_wins: float
    expected_loss_control: float
    expected_loss_treatment: float
    control_posterior_mean: float
    treatment_posterior_mean: float
    relative_lift_pct: float
    credible_interval_95: tuple
    recommendation: str

    def __str__(self):
        return (
            f"\n{'='*55}\n"
            f"  Bayesian A/B Test Result\n"
            f"{'='*55}\n"
            f"  P(Treatment > Control):   {self.prob_treatment_wins:.2%}\n"
            f"  Control Posterior Mean:   {self.control_posterior_mean:.4f}\n"
            f"  Treatment Posterior Mean: {self.treatment_posterior_mean:.4f}\n"
            f"  Relative Lift:            {self.relative_lift_pct:+.2f}%\n"
            f"  95% Credible Interval:    ({self.credible_interval_95[0]:.4f}, {self.credible_interval_95[1]:.4f})\n"
            f"  Expected Loss (Control):  {self.expected_loss_control:.5f}\n"
            f"  Expected Loss (Treatment):{self.expected_loss_treatment:.5f}\n"
            f"  → {self.recommendation}\n"
        )


def bayesian_proportion_test(
    control_conversions: int,
    control_n: int,
    treatment_conversions: int,
    treatment_n: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    n_samples: int = 100_000,
    decision_threshold: float = 0.95,
) -> BayesianResult:
    """
    Bayesian A/B test using Beta-Binomial conjugate model.

    The posterior for each variant is:
        Beta(alpha + conversions, beta + non_conversions)

    Parameters
    ----------
    control_conversions : int
    control_n : int
    treatment_conversions : int
    treatment_n : int
    prior_alpha : float
        Alpha parameter of Beta prior. Default 1 (uniform/uninformative).
    prior_beta : float
        Beta parameter of Beta prior. Default 1 (uniform/uninformative).
    n_samples : int
        Monte Carlo samples for probability estimation. Default 100,000.
    decision_threshold : float
        Probability threshold to recommend shipping. Default 0.95.

    Returns
    -------
    BayesianResult
    """
    # Posterior parameters
    control_alpha = prior_alpha + control_conversions
    control_beta = prior_beta + (control_n - control_conversions)
    treatment_alpha = prior_alpha + treatment_conversions
    treatment_beta = prior_beta + (treatment_n - treatment_conversions)

    # Sample from posteriors
    np.random.seed(42)
    control_samples = np.random.beta(control_alpha, control_beta, n_samples)
    treatment_samples = np.random.beta(treatment_alpha, treatment_beta, n_samples)

    # Core metrics
    prob_treatment_wins = (treatment_samples > control_samples).mean()

    # Expected loss: cost of choosing the wrong variant
    loss_control = np.maximum(treatment_samples - control_samples, 0).mean()
    loss_treatment = np.maximum(control_samples - treatment_samples, 0).mean()

    # Posterior means
    control_mean = control_alpha / (control_alpha + control_beta)
    treatment_mean = treatment_alpha / (treatment_alpha + treatment_beta)

    # 95% credible interval on the difference
    diff_samples = treatment_samples - control_samples
    ci = (np.percentile(diff_samples, 2.5), np.percentile(diff_samples, 97.5))

    # Recommendation
    if prob_treatment_wins >= decision_threshold:
        recommendation = f"Ship treatment — {prob_treatment_wins:.1%} probability of improvement."
    elif prob_treatment_wins <= (1 - decision_threshold):
        recommendation = f"Do not ship — treatment underperforms with {1-prob_treatment_wins:.1%} probability."
    else:
        recommendation = f"Continue testing — insufficient certainty ({prob_treatment_wins:.1%} P(win))."

    return BayesianResult(
        prob_treatment_wins=float(prob_treatment_wins),
        expected_loss_control=float(loss_control),
        expected_loss_treatment=float(loss_treatment),
        control_posterior_mean=control_mean,
        treatment_posterior_mean=treatment_mean,
        relative_lift_pct=(treatment_mean - control_mean) / control_mean * 100,
        credible_interval_95=ci,
        recommendation=recommendation,
    )


if __name__ == "__main__":
    result = bayesian_proportion_test(
        control_conversions=412, control_n=8500,
        treatment_conversions=489, treatment_n=8500,
    )
    print(result)
