"""
power_analysis.py
-----------------
Sample size and statistical power calculations for A/B tests.
Supports proportion tests (conversion rate) and mean tests (revenue, time-on-site).
"""

import numpy as np
from scipy import stats


def sample_size_for_proportion(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_tailed: bool = True,
) -> int:
    """
    Calculate required sample size per variant for a proportion-based A/B test.

    Parameters
    ----------
    baseline_rate : float
        Current conversion rate (e.g., 0.05 for 5%)
    mde : float
        Minimum detectable effect — absolute change (e.g., 0.01 = 1pp lift)
    alpha : float
        Significance level (Type I error rate). Default 0.05.
    power : float
        Statistical power (1 - Type II error rate). Default 0.80.
    two_tailed : bool
        Whether to use a two-tailed test. Default True.

    Returns
    -------
    int
        Required sample size per variant.

    Example
    -------
    >>> sample_size_for_proportion(baseline_rate=0.05, mde=0.01)
    3842
    """
    treatment_rate = baseline_rate + mde
    z_alpha = stats.norm.ppf(1 - alpha / (2 if two_tailed else 1))
    z_power = stats.norm.ppf(power)

    p1 = baseline_rate
    p2 = treatment_rate
    pooled = (p1 + p2) / 2

    numerator = (z_alpha * np.sqrt(2 * pooled * (1 - pooled)) +
                 z_power * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    denominator = (p2 - p1) ** 2

    return int(np.ceil(numerator / denominator))


def sample_size_for_mean(
    baseline_mean: float,
    baseline_std: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_tailed: bool = True,
) -> int:
    """
    Calculate required sample size per variant for a mean-based A/B test
    (e.g., average order value, session duration).
    """
    z_alpha = stats.norm.ppf(1 - alpha / (2 if two_tailed else 1))
    z_power = stats.norm.ppf(power)

    n = (2 * baseline_std ** 2 * (z_alpha + z_power) ** 2) / (mde ** 2)
    return int(np.ceil(n))


def achieved_power(
    baseline_rate: float,
    treatment_rate: float,
    n_per_variant: int,
    alpha: float = 0.05,
    two_tailed: bool = True,
) -> float:
    """
    Calculate the statistical power achieved at a given sample size.
    Useful for post-hoc analysis or checking power mid-experiment.
    """
    z_alpha = stats.norm.ppf(1 - alpha / (2 if two_tailed else 1))
    pooled = (baseline_rate + treatment_rate) / 2
    effect = abs(treatment_rate - baseline_rate)

    se = np.sqrt(2 * pooled * (1 - pooled) / n_per_variant)
    z_power = effect / se - z_alpha

    return float(stats.norm.cdf(z_power))


def runtime_estimate(
    daily_traffic: int,
    sample_size_per_variant: int,
    n_variants: int = 2,
) -> dict:
    """
    Estimate how long an experiment needs to run to hit target sample size.
    """
    traffic_per_variant = daily_traffic / n_variants
    days = np.ceil(sample_size_per_variant / traffic_per_variant)
    return {
        "required_sample_per_variant": sample_size_per_variant,
        "daily_traffic_per_variant": int(traffic_per_variant),
        "estimated_days": int(days),
        "estimated_weeks": round(days / 7, 1),
    }


if __name__ == "__main__":
    n = sample_size_for_proportion(baseline_rate=0.05, mde=0.01, alpha=0.05, power=0.80)
    runtime = runtime_estimate(daily_traffic=10000, sample_size_per_variant=n)

    print(f"Required sample size per variant: {n:,}")
    print(f"Estimated runtime: {runtime['estimated_days']} days ({runtime['estimated_weeks']} weeks)")
    print(f"Achieved power check: {achieved_power(0.05, 0.06, n):.2%}")
