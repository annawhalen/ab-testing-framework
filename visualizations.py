"""
visualizations.py
-----------------
Visualization functions for A/B test results and power analysis.
Produces publication-ready charts for stakeholder communication.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats


COLORS = {
    "control": "#4C72B0",
    "treatment": "#55A868",
    "significant": "#2ecc71",
    "not_significant": "#e74c3c",
    "neutral": "#95a5a6",
}


def plot_confidence_intervals(results: list, metric_name: str = "Conversion Rate", save_path: str = None):
    """
    Plot confidence intervals for multiple A/B test results side by side.

    Parameters
    ----------
    results : list of dicts, each with keys:
        - 'label': str
        - 'control_mean': float
        - 'treatment_mean': float
        - 'ci_lower': float
        - 'ci_upper': float
        - 'significant': bool
    """
    fig, ax = plt.subplots(figsize=(10, max(4, len(results) * 1.2)))

    for i, r in enumerate(results):
        diff = r["treatment_mean"] - r["control_mean"]
        color = COLORS["significant"] if r["significant"] else COLORS["not_significant"]

        ax.barh(i, diff, color=color, alpha=0.7, height=0.5)
        ax.errorbar(
            x=diff, y=i,
            xerr=[[diff - r["ci_lower"]], [r["ci_upper"] - diff]],
            fmt="none", color="black", capsize=5, linewidth=2
        )
        ax.text(max(r["ci_upper"], diff) + 0.001, i,
                f"  {'✓' if r['significant'] else '✗'} {diff:+.4f}",
                va="center", fontsize=10)

    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels([r["label"] for r in results])
    ax.set_xlabel(f"Absolute Lift in {metric_name}")
    ax.set_title(f"A/B Test Results — {metric_name}", fontweight="bold", pad=15)

    sig_patch = mpatches.Patch(color=COLORS["significant"], alpha=0.7, label="Significant")
    ns_patch = mpatches.Patch(color=COLORS["not_significant"], alpha=0.7, label="Not Significant")
    ax.legend(handles=[sig_patch, ns_patch], loc="lower right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_power_curve(
    baseline_rate: float,
    mde_range: np.ndarray = None,
    n_per_variant: int = 5000,
    alpha: float = 0.05,
    save_path: str = None,
):
    """
    Plot statistical power as a function of minimum detectable effect.
    """
    if mde_range is None:
        mde_range = np.linspace(0.001, 0.05, 100)

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    powers = []

    for mde in mde_range:
        treatment_rate = baseline_rate + mde
        pooled = (baseline_rate + treatment_rate) / 2
        se = np.sqrt(2 * pooled * (1 - pooled) / n_per_variant)
        z_power = abs(treatment_rate - baseline_rate) / se - z_alpha
        powers.append(stats.norm.cdf(z_power))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(mde_range * 100, [p * 100 for p in powers], color=COLORS["control"], linewidth=2.5)
    ax.axhline(80, color=COLORS["not_significant"], linestyle="--", linewidth=1.5, label="80% Power threshold")
    ax.axhline(90, color=COLORS["significant"], linestyle="--", linewidth=1.5, label="90% Power threshold")
    ax.fill_between(mde_range * 100, [p * 100 for p in powers], alpha=0.1, color=COLORS["control"])

    ax.set_xlabel("Minimum Detectable Effect (percentage points)")
    ax.set_ylabel("Statistical Power (%)")
    ax.set_title(
        f"Power Curve  |  n={n_per_variant:,}/variant  |  baseline={baseline_rate:.1%}  |  α={alpha}",
        fontweight="bold"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_bayesian_posteriors(
    control_alpha: float, control_beta: float,
    treatment_alpha: float, treatment_beta: float,
    save_path: str = None,
):
    """
    Plot posterior distributions for Bayesian A/B test.
    """
    x = np.linspace(0, 0.2, 1000)
    control_dist = stats.beta(control_alpha, control_beta)
    treatment_dist = stats.beta(treatment_alpha, treatment_beta)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, control_dist.pdf(x), color=COLORS["control"], linewidth=2.5, label="Control")
    ax.plot(x, treatment_dist.pdf(x), color=COLORS["treatment"], linewidth=2.5, label="Treatment")
    ax.fill_between(x, control_dist.pdf(x), alpha=0.15, color=COLORS["control"])
    ax.fill_between(x, treatment_dist.pdf(x), alpha=0.15, color=COLORS["treatment"])

    ax.axvline(control_dist.mean(), color=COLORS["control"], linestyle="--", alpha=0.7)
    ax.axvline(treatment_dist.mean(), color=COLORS["treatment"], linestyle="--", alpha=0.7)

    ax.set_xlabel("Conversion Rate")
    ax.set_ylabel("Posterior Density")
    ax.set_title("Bayesian Posterior Distributions — Control vs Treatment", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_power_curve(baseline_rate=0.05, n_per_variant=8500)
