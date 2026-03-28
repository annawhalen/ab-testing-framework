"""
frequentist_tests.py
--------------------
Frequentist A/B test implementations: z-test, t-test, chi-square.
Each function returns a clean result dict with all key statistics.
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass


@dataclass
class TestResult:
    test_type: str
    control_mean: float
    treatment_mean: float
    absolute_lift: float
    relative_lift_pct: float
    p_value: float
    statistic: float
    confidence_interval: tuple
    alpha: float
    significant: bool
    recommendation: str

    def __str__(self):
        sig = "✅ SIGNIFICANT" if self.significant else "❌ NOT SIGNIFICANT"
        return (
            f"\n{'='*55}\n"
            f"  {self.test_type} Result — {sig}\n"
            f"{'='*55}\n"
            f"  Control:        {self.control_mean:.4f}\n"
            f"  Treatment:      {self.treatment_mean:.4f}\n"
            f"  Absolute Lift:  {self.absolute_lift:+.4f}\n"
            f"  Relative Lift:  {self.relative_lift_pct:+.2f}%\n"
            f"  p-value:        {self.p_value:.4f}  (alpha={self.alpha})\n"
            f"  Test Statistic: {self.statistic:.4f}\n"
            f"  95% CI:         ({self.confidence_interval[0]:.4f}, {self.confidence_interval[1]:.4f})\n"
            f"  → {self.recommendation}\n"
        )


def proportion_ztest(
    control_conversions: int,
    control_n: int,
    treatment_conversions: int,
    treatment_n: int,
    alpha: float = 0.05,
    two_tailed: bool = True,
) -> TestResult:
    """
    Two-proportion z-test for conversion rate experiments.
    """
    p_control = control_conversions / control_n
    p_treatment = treatment_conversions / treatment_n
    p_pooled = (control_conversions + treatment_conversions) / (control_n + treatment_n)

    se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / control_n + 1 / treatment_n))
    z_stat = (p_treatment - p_control) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat))) if two_tailed else (1 - stats.norm.cdf(z_stat))

    z_crit = stats.norm.ppf(1 - alpha / (2 if two_tailed else 1))
    se_diff = np.sqrt(p_control * (1 - p_control) / control_n +
                      p_treatment * (1 - p_treatment) / treatment_n)
    diff = p_treatment - p_control
    ci = (diff - z_crit * se_diff, diff + z_crit * se_diff)

    significant = p_value < alpha
    recommendation = (
        "Ship the treatment — statistically significant improvement detected."
        if significant and diff > 0
        else "Do not ship — treatment is significantly worse."
        if significant and diff < 0
        else "Inconclusive — insufficient evidence to make a decision."
    )

    return TestResult(
        test_type="Two-Proportion Z-Test",
        control_mean=p_control,
        treatment_mean=p_treatment,
        absolute_lift=diff,
        relative_lift_pct=diff / p_control * 100,
        p_value=p_value,
        statistic=z_stat,
        confidence_interval=ci,
        alpha=alpha,
        significant=significant,
        recommendation=recommendation,
    )


def means_ttest(
    control_values: np.ndarray,
    treatment_values: np.ndarray,
    alpha: float = 0.05,
) -> TestResult:
    """
    Welch's t-test for continuous metrics (e.g., revenue, session duration).
    """
    control_values = np.asarray(control_values)
    treatment_values = np.asarray(treatment_values)

    t_stat, p_value = stats.ttest_ind(treatment_values, control_values, equal_var=False)

    mean_c = control_values.mean()
    mean_t = treatment_values.mean()
    diff = mean_t - mean_c

    se = np.sqrt(control_values.var(ddof=1) / len(control_values) +
                 treatment_values.var(ddof=1) / len(treatment_values))
    df = len(control_values) + len(treatment_values) - 2
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci = (diff - t_crit * se, diff + t_crit * se)

    significant = p_value < alpha
    recommendation = (
        "Ship the treatment — statistically significant improvement."
        if significant and diff > 0
        else "Do not ship — treatment is significantly worse."
        if significant and diff < 0
        else "Inconclusive — insufficient evidence to make a decision."
    )

    return TestResult(
        test_type="Welch's T-Test (Means)",
        control_mean=mean_c,
        treatment_mean=mean_t,
        absolute_lift=diff,
        relative_lift_pct=diff / mean_c * 100,
        p_value=p_value,
        statistic=t_stat,
        confidence_interval=ci,
        alpha=alpha,
        significant=significant,
        recommendation=recommendation,
    )


def chi_square_test(
    contingency_table: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    Chi-square test of independence for categorical outcomes.
    """
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    return {
        "test_type": "Chi-Square Test of Independence",
        "chi2_statistic": round(chi2, 4),
        "p_value": round(p_value, 4),
        "degrees_of_freedom": dof,
        "expected_frequencies": expected,
        "alpha": alpha,
        "significant": p_value < alpha,
        "recommendation": (
            "Significant association detected between variant and outcome distribution."
            if p_value < alpha
            else "No significant association between variant and outcome distribution."
        ),
    }


if __name__ == "__main__":
    result = proportion_ztest(
        control_conversions=412, control_n=8500,
        treatment_conversions=489, treatment_n=8500,
    )
    print(result)
