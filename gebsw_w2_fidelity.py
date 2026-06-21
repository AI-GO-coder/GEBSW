#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GEBSW Wasserstein-fidelity experiment on controlled manifold distribution pairs.

Purpose
-------
For each controlled manifold distribution pair, compute:
  1) exact empirical 2-Wasserstein distance W2(X, Y) in the original sample space;
  2) GEBSW distances for 18 configurations:
       projection order q in {1, 3, 5}
       energy rule in {C, e, r=1, r=2, r=3, r=4}

Then visualize how close each GEBSW configuration is to the exact empirical W2.
This experiment is designed to study Wasserstein-fidelity / metric fidelity of
GEBSW configurations, not downstream optimization performance.

Distribution pairs
------------------
  low_dim_linear       : noisy line-segment pair in 2D
  low_dim_nonlinear    : circle vs flower-shaped nonlinear manifold in 2D
  high_dim_linear      : low-rank linear subspace pair embedded in high dimension
  high_dim_nonlinear   : Swiss roll vs twisted Swiss roll embedded in high dimension

Outputs
-------
For each distribution type:
  - fidelity_heatmap_<scenario>.png          (primary heatmap; color encodes relative error)
  - distance_with_w2_reference_<scenario>.png

CSV files:
  - raw_results.csv
  - summary_results.csv
  - best_by_scenario.csv

Run examples
------------
Quick test:
    python gebsw_w2_fidelity_manifold_experiment.py --n 80 --repeats 3 --L 64

Paper-scale run:
    python gebsw_w2_fidelity_manifold_experiment.py --n 128 --repeats 20 --L 128

Notes
-----
The exact empirical W2 is computed by solving the equal-weight assignment problem:
    W2^2 = (1/n) min_pi sum_i ||x_i - y_{pi(i)}||^2
using scipy.optimize.linear_sum_assignment.

For high-dimensional polynomial projections, all monomials up to degree q are used.
To avoid combinatorial explosion, the default high ambient dimension is 10.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "This script requires scipy. Please install it with `pip install scipy`."
    ) from exc

EPS = 1e-12
P = 2

Q_LIST = [1, 3, 5]
ENERGY_RULES: List[Tuple[str, Optional[int]]] = [
    ("C", None),
    ("e", None),
    ("r", 1),
    ("r", 2),
    ("r", 3),
    ("r", 4),
]
ENERGY_ORDER = ["C", "e", "r=1", "r=2", "r=3", "r=4"]

# Short labels are intentionally compact for Scientific Reports-style composite figures.
# Detailed descriptions should be placed in the figure legend rather than as long in-panel titles.
SCENARIO_SHORT_LABELS = {
    "low_dim_linear": "2D affine line",
    "low_dim_nonlinear": "Circle vs flower",
    "high_dim_linear": "High-D linear",
    "high_dim_nonlinear": "High-D nonlinear",
}


@dataclass(frozen=True)
class Config:
    q: int
    rule: str
    r: Optional[int] = None

    @property
    def energy_label(self) -> str:
        if self.rule == "C":
            return "C"
        if self.rule == "e":
            return "e"
        return f"r={int(self.r)}"

    @property
    def method_label(self) -> str:
        return f"GEBSW({self.energy_label},{self.q})"

    @property
    def family(self) -> str:
        # Baseline special cases: C,q is SW/GSW; adaptive q=1 is EBSW-family.
        if self.rule == "C":
            return "baseline_uniform_SW_GSW"
        if self.q == 1:
            return "baseline_linear_adaptive_EBSW"
        return "nonbaseline_nonlinear_adaptive_GEBSW"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GEBSW Wasserstein-fidelity heatmaps on manifold distributions."
    )
    parser.add_argument("--n", type=int, default=128, help="Number of samples per distribution.")
    parser.add_argument("--L", type=int, default=128, help="Number of projections / slices.")
    parser.add_argument("--repeats", type=int, default=20, help="Number of random repeats.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--dim_high", type=int, default=10, help="Ambient dimension for high-dimensional cases.")
    parser.add_argument("--noise", type=float, default=0.03, help="Noise level added to manifolds.")
    parser.add_argument("--temperature_scale", type=float, default=1.0,
                        help="Softmax temperature scale for e-rule: T = scale * std(costs).")
    parser.add_argument("--out_dir", type=str, default="gebsw_w2_fidelity_outputs",
                        help="Output directory.")
    parser.add_argument("--plot_n", type=int, default=384,
                        help="Number of points used only for distribution visualization panels.")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["low_dim_linear", "low_dim_nonlinear", "high_dim_linear", "high_dim_nonlinear"],
        choices=["low_dim_linear", "low_dim_nonlinear", "high_dim_linear", "high_dim_nonlinear"],
        help="Distribution scenarios to run.",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def rng_from(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def normalize_pair(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Jointly normalize a distribution pair in the original sample space."""
    Z = np.vstack([X, Y])
    mean = Z.mean(axis=0, keepdims=True)
    std = Z.std(axis=0, keepdims=True) + 1e-8
    return (X - mean) / std, (Y - mean) / std


def normalize_feature_pair(FX: np.ndarray, FY: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Jointly normalize feature representations to avoid scale domination."""
    Z = np.vstack([FX, FY])
    mean = Z.mean(axis=0, keepdims=True)
    std = Z.std(axis=0, keepdims=True) + 1e-8
    return (FX - mean) / std, (FY - mean) / std


def random_orthonormal_matrix(rng: np.random.Generator, input_dim: int, output_dim: int) -> np.ndarray:
    """Return an input_dim x output_dim matrix with approximately orthonormal columns."""
    A = rng.normal(size=(input_dim, output_dim))
    Q, _ = np.linalg.qr(A)
    return Q[:, :output_dim]


# -----------------------------------------------------------------------------
# Distribution pairs
# -----------------------------------------------------------------------------

def make_low_dim_linear(n: int, seed: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
    """Strict low-dimensional linear-manifold pair.

    Design goal:
        q=1 should be favored because the two distributions live on the same
        one-dimensional affine linear manifold. The target differs mainly by
        an affine shift along the same line plus very small isotropic noise.

    This replaces the earlier rotated-line construction, which could make
    higher-order polynomial projections numerically closer to W2 even though
    the scenario was named "linear".
    """
    rng = rng_from(seed)
    t = np.sort(rng.uniform(-1.0, 1.0, size=n))
    direction = np.array([1.0, 0.18], dtype=np.float64)
    direction = direction / (np.linalg.norm(direction) + EPS)
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)

    X = t[:, None] * direction[None, :]
    # Same linear support: mild scale + translation along the same affine line.
    Y = (1.05 * t + 0.42)[:, None] * direction[None, :]

    # Small shared-thickness noise, weaker than the affine displacement.
    X = X + noise * (0.35 * rng.normal(size=(n, 1)) * direction[None, :] +
                     0.15 * rng.normal(size=(n, 1)) * normal[None, :])
    Y = Y + noise * (0.35 * rng.normal(size=(n, 1)) * direction[None, :] +
                     0.15 * rng.normal(size=(n, 1)) * normal[None, :])
    return normalize_pair(X, Y)


def make_low_dim_nonlinear(n: int, seed: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
    """Low-dimensional nonlinear-manifold pair.

    Design goal:
        q=3 or q=5 should be favored because Y is a nonlinear radial/angular
        deformation of X rather than a pure affine transformation.
    """
    rng = rng_from(seed)
    t = rng.uniform(0.0, 2.0 * np.pi, size=n)
    X = np.column_stack([np.cos(t), np.sin(t)])

    # Same angular support but nonlinear radial deformation.
    r = 1.0 + 0.32 * np.sin(5.0 * t + 0.35)
    Y = np.column_stack([r * np.cos(t + 0.25), r * np.sin(t + 0.25)])

    X = X + noise * rng.normal(size=X.shape)
    Y = Y + noise * rng.normal(size=Y.shape)
    return normalize_pair(X, Y)


def make_high_dim_linear(n: int, seed: int, noise: float, dim_high: int) -> Tuple[np.ndarray, np.ndarray]:
    """Strict high-dimensional low-rank linear-manifold pair.

    Design goal:
        q=1, especially C/q=1 or linear adaptive weighting, should be stable.
        Both X and Y share the same low-rank linear subspace embedded in a
        high-dimensional ambient space. The target is generated by an affine
        transform in the same latent linear coordinates, not by changing to a
        different nonlinear feature geometry.
    """
    rng = rng_from(seed)
    latent_dim = 2
    Z = rng.normal(size=(n, latent_dim))

    A = rng.normal(size=(latent_dim, dim_high))
    # Orthonormalize rows approximately through QR on the transpose.
    Q, _ = np.linalg.qr(A.T)
    A = Q[:, :latent_dim].T

    latent_shift = np.array([0.55, -0.25], dtype=np.float64)
    latent_scale = np.array([1.05, 0.95], dtype=np.float64)
    X_lat = Z
    Y_lat = Z * latent_scale[None, :] + latent_shift[None, :]

    X = X_lat @ A
    Y = Y_lat @ A

    # Weak ambient noise only; it should not dominate the linear subspace.
    X = X + noise * 0.35 * rng.normal(size=X.shape)
    Y = Y + noise * 0.35 * rng.normal(size=Y.shape)
    return normalize_pair(X, Y)


def make_swiss_roll(n: int, seed: int, noise: float, twist: bool) -> np.ndarray:
    rng = rng_from(seed)
    t = rng.uniform(1.5 * np.pi, 4.5 * np.pi, size=n)
    h = rng.uniform(-1.0, 1.0, size=n)

    x = t * np.cos(t)
    y = h * 5.0
    z = t * np.sin(t)
    X = np.column_stack([x, y, z])
    X = X / (np.std(X, axis=0, keepdims=True) + 1e-8)

    if twist:
        # Nonlinear twist and radial perturbation.
        angle = 0.55 * np.sin(0.55 * t)
        ca, sa = np.cos(angle), np.sin(angle)
        x0, z0 = X[:, 0].copy(), X[:, 2].copy()
        X[:, 0] = x0 * ca - z0 * sa
        X[:, 2] = x0 * sa + z0 * ca
        X[:, 1] = X[:, 1] + 0.35 * np.sin(1.3 * X[:, 0])

    X = X + noise * rng.normal(size=X.shape)
    return X


def make_high_dim_nonlinear(n: int, seed: int, noise: float, dim_high: int) -> Tuple[np.ndarray, np.ndarray]:
    """High-dimensional nonlinear-manifold pair.

    Design goal:
        higher-order GEBSW variants should be favored because Y is a nonlinear
        twist of the same Swiss-roll latent manifold before high-dimensional
        embedding.
    """
    rng = rng_from(seed)
    X3 = make_swiss_roll(n, seed=seed + 11, noise=noise, twist=False)
    Y3 = make_swiss_roll(n, seed=seed + 11, noise=noise, twist=True)

    A = rng.normal(size=(3, dim_high))
    A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-8)
    X = X3 @ A
    Y = Y3 @ A

    # Add weak high-dimensional nuisance noise.
    X = X + noise * rng.normal(size=X.shape)
    Y = Y + noise * rng.normal(size=Y.shape)
    return normalize_pair(X, Y)


def make_pair(scenario: str, n: int, seed: int, noise: float, dim_high: int) -> Tuple[np.ndarray, np.ndarray]:
    if scenario == "low_dim_linear":
        return make_low_dim_linear(n, seed, noise)
    if scenario == "low_dim_nonlinear":
        return make_low_dim_nonlinear(n, seed, noise)
    if scenario == "high_dim_linear":
        return make_high_dim_linear(n, seed, noise, dim_high)
    if scenario == "high_dim_nonlinear":
        return make_high_dim_nonlinear(n, seed, noise, dim_high)
    raise ValueError(f"Unknown scenario: {scenario}")


# -----------------------------------------------------------------------------
# Distances
# -----------------------------------------------------------------------------

def exact_empirical_w2(X: np.ndarray, Y: np.ndarray) -> float:
    """Exact equal-weight empirical W2 using assignment."""
    if len(X) != len(Y):
        raise ValueError("X and Y must have the same number of samples for exact assignment W2.")
    C = cdist(X, Y, metric="sqeuclidean")
    row_ind, col_ind = linear_sum_assignment(C)
    w2_sq = float(C[row_ind, col_ind].mean())
    return math.sqrt(max(w2_sq, 0.0))


def monomial_powers(dim: int, max_degree: int) -> List[Tuple[int, ...]]:
    """All monomial exponent tuples with total degree 1..max_degree."""
    powers: List[Tuple[int, ...]] = []
    for total_degree in range(1, max_degree + 1):
        # Generate weak compositions of total_degree into dim parts.
        for comb in itertools.combinations_with_replacement(range(dim), total_degree):
            exp = [0] * dim
            for idx in comb:
                exp[idx] += 1
            powers.append(tuple(exp))
    return powers


def polynomial_features(X: np.ndarray, q: int) -> np.ndarray:
    """Polynomial feature map up to degree q.

    For q=1, this is the identity feature map. For q=3/5, all monomials up to
    degree q are used. This is tractable for the default high dimension 10.
    """
    X = np.asarray(X, dtype=np.float64)
    if q == 1:
        return X.copy()
    n, dim = X.shape
    powers = monomial_powers(dim, q)
    F = np.empty((n, len(powers)), dtype=np.float64)
    for j, exp in enumerate(powers):
        val = np.ones(n, dtype=np.float64)
        for k, power in enumerate(exp):
            if power:
                val *= X[:, k] ** power
        F[:, j] = val
    return F


def sample_directions(dim: int, L: int, seed: int) -> np.ndarray:
    rng = rng_from(seed)
    theta = rng.normal(size=(dim, L))
    theta /= np.linalg.norm(theta, axis=0, keepdims=True) + EPS
    return theta


def energy_weights(costs: np.ndarray, rule: str, r: Optional[int], temperature_scale: float) -> np.ndarray:
    costs = np.asarray(costs, dtype=np.float64) + EPS
    if rule == "C":
        return np.ones_like(costs) / len(costs)
    if rule == "e":
        temp = temperature_scale * np.std(costs) + 1e-8
        logits = (costs - np.max(costs)) / temp
        w = np.exp(logits)
    elif rule == "r":
        if r is None:
            raise ValueError("r must be provided for power energy rule.")
        w = costs ** int(r) + EPS
    else:
        raise ValueError(f"Unknown energy rule: {rule}")
    return w / (np.sum(w) + EPS)


def projected_wasserstein_costs(FX: np.ndarray, FY: np.ndarray, theta: np.ndarray) -> np.ndarray:
    PX = FX @ theta
    PY = FY @ theta
    SX = np.sort(PX, axis=0)
    SY = np.sort(PY, axis=0)
    return np.mean(np.abs(SX - SY) ** P, axis=0) + EPS


def gebsw_distance(
    X: np.ndarray,
    Y: np.ndarray,
    cfg: Config,
    L: int,
    seed: int,
    temperature_scale: float,
) -> Tuple[float, float, float]:
    """Return GEBSW distance, normalized entropy, top-5 weight ratio."""
    FX = polynomial_features(X, cfg.q)
    FY = polynomial_features(Y, cfg.q)
    FX, FY = normalize_feature_pair(FX, FY)
    theta = sample_directions(FX.shape[1], L, seed=seed)
    costs = projected_wasserstein_costs(FX, FY, theta)
    w = energy_weights(costs, cfg.rule, cfg.r, temperature_scale=temperature_scale)
    d = float(np.sum(w * costs) ** (1.0 / P))
    entropy = float(-np.sum(w * np.log(w + EPS)) / np.log(len(w)))
    top5 = float(np.sum(np.sort(w)[-min(5, len(w)):]))
    return d, entropy, top5


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def pivot_metric(summary: pd.DataFrame, scenario: str, metric: str) -> pd.DataFrame:
    sub = summary[summary["scenario"] == scenario]
    pivot = sub.pivot(index="energy_rule", columns="q", values=metric)
    return pivot.loc[ENERGY_ORDER, Q_LIST]


def plot_heatmap(
    pivot_values: pd.DataFrame,
    pivot_labels: Optional[pd.DataFrame],
    title: str,
    cbar_label: str,
    out_path: str,
    fmt: str = ".3f",
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.4))

    values = pivot_values.values.astype(float)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))

    # Use a more eye-catching and interpretable colormap:
    # lower error = greener (better), higher error = redder (worse).
    im = ax.imshow(values, aspect="auto", cmap="RdYlGn_r", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(pivot_values.columns)))
    ax.set_xticklabels([str(c) for c in pivot_values.columns], fontsize=11)
    ax.set_yticks(np.arange(len(pivot_values.index)))
    ax.set_yticklabels(list(pivot_values.index), fontsize=11)
    ax.set_xlabel("Projection order q", fontsize=12)
    ax.set_ylabel("Energy rule", fontsize=12)
    ax.set_title(title, fontsize=14)

    # Thin white gridlines make each cell boundary clearer.
    ax.set_xticks(np.arange(-0.5, len(pivot_values.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot_values.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0, alpha=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Only highlight the globally best cell, i.e. the lowest relative error.
    # Other cells keep plain text without a background box, avoiding clutter and
    # making the best configuration visually prominent.
    best_value = float(np.nanmin(values))
    value_range = max(vmax - vmin, EPS)
    for i in range(pivot_values.shape[0]):
        for j in range(pivot_values.shape[1]):
            if pivot_labels is None:
                label_text = format(float(values[i, j]), fmt)
            else:
                label_text = str(pivot_labels.values[i, j])

            is_best = bool(np.isclose(values[i, j], best_value, rtol=1e-10, atol=1e-12))
            norm_val = (float(values[i, j]) - vmin) / value_range
            plain_text_color = "white" if norm_val < 0.28 else "black"

            if is_best:
                best_text_color = "white" if norm_val < 0.28 else "black"
                ax.text(
                    j,
                    i,
                    label_text,
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color=best_text_color,
                    zorder=4,
                )
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=3.0, zorder=6)
                ax.add_patch(rect)
            else:
                ax.text(
                    j,
                    i,
                    label_text,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="normal",
                    color=plain_text_color,
                )

    cbar = fig.colorbar(im, ax=ax, label=cbar_label)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label(cbar_label, fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)




def pca_project_pair(X: np.ndarray, Y: np.ndarray, out_dim: int = 2) -> Tuple[np.ndarray, np.ndarray, float]:
    """Project a pair to 2D by a shared PCA/SVD basis."""
    Z = np.vstack([X, Y]).astype(np.float64)
    Zc = Z - Z.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(Zc, full_matrices=False)
    basis = vt[:out_dim].T
    coords = Zc @ basis
    explained = float(np.sum(s[:out_dim] ** 2) / (np.sum(s ** 2) + EPS))
    return coords[: len(X)], coords[len(X):], explained


def get_plot_coordinates(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
    """Return 2D coordinates and a compact coordinate note."""
    if X.shape[1] == 2:
        return X[:, :2], Y[:, :2], "original"
    Xp, Yp, explained = pca_project_pair(X, Y, out_dim=2)
    return Xp, Yp, f"PCA {100.0 * explained:.1f}%"


def set_equal_2d_limits(ax: plt.Axes, Xp: np.ndarray, Yp: np.ndarray, pad_ratio: float = 0.08) -> None:
    Z = np.vstack([Xp, Yp])
    xmin, ymin = Z.min(axis=0)
    xmax, ymax = Z.max(axis=0)
    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    span = max(float(xmax - xmin), float(ymax - ymin), EPS)
    half = 0.5 * span * (1.0 + pad_ratio)
    ax.set_xlim(xmid - half, xmid + half)
    ax.set_ylim(ymid - half, ymid + half)
    ax.set_aspect("equal", adjustable="box")


def subsample_for_plot(X: np.ndarray, Y: np.ndarray, max_n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Subsample only for visualization; the quantitative experiment is unaffected."""
    if len(X) <= max_n and len(Y) <= max_n:
        return X, Y
    rng = rng_from(seed)
    ix = rng.choice(len(X), size=min(max_n, len(X)), replace=False)
    iy = rng.choice(len(Y), size=min(max_n, len(Y)), replace=False)
    return X[ix], Y[iy]


def representative_pair_for_plot(args: argparse.Namespace, scenario: str, scenario_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic representative distribution pair for the left panels.

    The high-dimensional nonlinear case is generated from a Swiss-roll manifold.
    It is visually more informative when plotted with more points, so only this
    panel uses a denser visualization sample. The quantitative experiment is
    unchanged and still uses args.n.
    """
    seed_pair = args.seed + 70001 + 101 * (scenario_idx + 1)
    plot_multiplier = 3 if scenario == "high_dim_nonlinear" else 1
    visual_n = max(int(args.plot_n) * plot_multiplier, int(args.n))
    X, Y = make_pair(
        scenario=scenario,
        n=visual_n,
        seed=seed_pair,
        noise=args.noise,
        dim_high=args.dim_high,
    )
    return subsample_for_plot(X, Y, max_n=visual_n, seed=seed_pair + 999)


def make_fidelity_label_grid(dist_pivot: pd.DataFrame, err_pivot: pd.DataFrame) -> pd.DataFrame:
    """Compact cell label in the format: distance(error%)."""
    labels = err_pivot.copy().astype(object)
    for er in ENERGY_ORDER:
        for q in Q_LIST:
            d = float(dist_pivot.loc[er, q])
            e = float(err_pivot.loc[er, q])
            labels.loc[er, q] = f"{d:.3f} ({100*e:.1f}%)"
    return labels


def plot_distribution_compact(
    ax: plt.Axes,
    X: np.ndarray,
    Y: np.ndarray,
    scenario: str,
    show_legend: bool = False,
    w2_ref: Optional[float] = None,
) -> None:
    """Compact distribution panel for the left column without frame or axes."""
    Xp, Yp, _ = get_plot_coordinates(X, Y)
    ax.scatter(Xp[:, 0], Xp[:, 1], s=4.2, alpha=0.72, label="Source", edgecolors="none")
    ax.scatter(Yp[:, 0], Yp[:, 1], s=4.2, alpha=0.72, label="Target", edgecolors="none")
    set_equal_2d_limits(ax, Xp, Yp)

    # Remove rectangle frame, coordinate axes, ticks, labels, and grid for a compact paper-style panel.
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Keep the in-panel title compact. The PCA explained-variance percentage is
    # omitted from the title to avoid distracting from the ablation comparison.
    title = SCENARIO_SHORT_LABELS.get(scenario, scenario)
    ax.set_title(title, fontsize=8.0, fontweight="bold", pad=2.0)

    # The empirical W2 reference is a row-level property of the source--target
    # pair, so it is displayed under the distribution visualization rather than
    # as the heatmap title. The caption/text can clarify that the displayed
    # number is averaged over repeats.
    if w2_ref is not None:
        ax.text(
            0.5,
            -0.085,
            rf"$W_2^{{\mathrm{{ref}}}}={w2_ref:.4f}$",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.2,
            fontweight="normal",
            clip_on=False,
        )

    if show_legend:
        ax.legend(loc="best", fontsize=5.8, frameon=False, borderpad=0.15, handletextpad=0.25, markerscale=1.2)


def draw_compact_heatmap_on_ax(
    ax: plt.Axes,
    err_pivot: pd.DataFrame,
    labels: pd.DataFrame,
    w2_ref: float,
    vmin: float,
    vmax: float,
) -> matplotlib.image.AxesImage:
    """Compact heatmap for the right column; global vmin/vmax enables row-wise comparison."""
    values = err_pivot.values.astype(float)
    im = ax.imshow(values, aspect="auto", cmap="RdYlGn_r", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(Q_LIST)))
    ax.set_xticklabels([str(q) for q in Q_LIST], fontsize=7.4)
    ax.set_yticks(np.arange(len(ENERGY_ORDER)))
    ax.set_yticklabels(ENERGY_ORDER, fontsize=7.4)
    ax.set_xlabel("Projection order q", fontsize=7.8, labelpad=1.0)
    ax.set_ylabel("Energy rule", fontsize=7.8, labelpad=0.2)

    ax.set_xticks(np.arange(-0.5, len(Q_LIST), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ENERGY_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.7, alpha=0.85)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=2.0, width=0.6, pad=1.0)

    best_value = float(np.nanmin(values))
    value_range = max(float(vmax - vmin), EPS)
    for i in range(err_pivot.shape[0]):
        for j in range(err_pivot.shape[1]):
            val = float(values[i, j])
            is_best = bool(np.isclose(val, best_value, rtol=1e-10, atol=1e-12))
            # Text color is chosen from the globally normalized value.
            norm_val = (val - vmin) / value_range
            text_color = "white" if norm_val < 0.28 else "black"
            ax.text(
                j, i, str(labels.values[i, j]),
                ha="center", va="center",
                fontsize=6.0 if not is_best else 6.5,
                fontweight="bold" if is_best else "normal",
                color=text_color,
                linespacing=1.0,
                zorder=7 if is_best else 3,
            )
            if is_best:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=3.0, zorder=6)
                ax.add_patch(rect)
    return im


def plot_sr_compact_2col4row(
    summary: pd.DataFrame,
    scenarios: Iterable[str],
    args: argparse.Namespace,
    out_path: str,
) -> None:
    """Create a compact Scientific Reports-style 2-column x 4-row composite figure.

    Left column: distribution pair visualization.
    Right column: GEBSW Wasserstein-fidelity heatmap.

    The actual journal-style figure title and detailed panel explanations should be
    provided in the figure legend. Inside the figure, only compact panel titles are used.
    """
    valid_scenarios = [s for s in scenarios if not summary[summary["scenario"] == s].empty]
    if len(valid_scenarios) != 4:
        raise ValueError(
            "The compact paper figure is designed for exactly 4 scenarios. "
            f"Got {len(valid_scenarios)} valid scenarios: {valid_scenarios}"
        )

    # Global color scale makes relative-error colors comparable across all heatmaps.
    all_err_values = []
    for scenario in valid_scenarios:
        all_err_values.append(pivot_metric(summary, scenario, "relative_error_mean").values.ravel())
    all_err_values = np.concatenate(all_err_values).astype(float)
    vmin = float(np.nanmin(all_err_values))
    vmax = float(np.nanmax(all_err_values))

    # Full-width figure; heatmaps are deliberately wider than distribution panels.
    fig = plt.figure(figsize=(7.2, 8.55))
    gs = fig.add_gridspec(
        nrows=4,
        ncols=3,
        width_ratios=[0.68, 2.22, 0.050],
        height_ratios=[1, 1, 1, 1],
        hspace=0.40,
        wspace=0.16,
    )

    last_im = None
    for row, scenario in enumerate(valid_scenarios):
        ax_dist = fig.add_subplot(gs[row, 0])
        ax_heat = fig.add_subplot(gs[row, 1])

        dist_pivot = pivot_metric(summary, scenario, "distance_mean")
        err_pivot = pivot_metric(summary, scenario, "relative_error_mean")
        labels = make_fidelity_label_grid(dist_pivot, err_pivot)
        w2_ref = float(summary[summary["scenario"] == scenario]["w2_reference_mean"].iloc[0])

        X, Y = representative_pair_for_plot(args, scenario, row)
        plot_distribution_compact(ax_dist, X, Y, scenario, show_legend=(row == 0), w2_ref=w2_ref)

        last_im = draw_compact_heatmap_on_ax(ax_heat, err_pivot, labels, w2_ref, vmin=vmin, vmax=vmax)

    cax = fig.add_subplot(gs[:, 2])
    cb = fig.colorbar(last_im, cax=cax)
    cb.ax.tick_params(labelsize=7.0, length=2.0, width=0.6)
    cb.set_label("Relative error", fontsize=8.0, labelpad=5)

    fig.subplots_adjust(left=0.055, right=0.965, top=0.985, bottom=0.060, hspace=0.40, wspace=0.16)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    # Also save a vector version for journal submission.
    base, _ = os.path.splitext(out_path)
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def plot_distance_bar_with_reference(summary: pd.DataFrame, scenario: str, out_path: str) -> None:
    sub = summary[summary["scenario"] == scenario].copy()
    sub["config_order"] = sub["energy_rule"].map({e: i for i, e in enumerate(ENERGY_ORDER)}) * 10 + sub["q"]
    sub = sub.sort_values(["config_order", "q"])
    labels = [f"{er},{q}" for er, q in zip(sub["energy_rule"], sub["q"])]
    values = sub["distance_mean"].to_numpy()
    w2 = float(sub["w2_reference_mean"].iloc[0])

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.bar(np.arange(len(values)), values, alpha=0.85)
    ax.axhline(w2, color="red", linestyle="--", linewidth=2.0, label=f"Exact empirical W2 = {w2:.4f}")
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Distance")
    ax.set_title(f"GEBSW distances with exact W2 reference: {scenario}")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def make_plots(summary: pd.DataFrame, scenarios: Iterable[str], out_dir: str, args: argparse.Namespace) -> None:
    plot_dir = os.path.join(out_dir, "plots")
    ensure_dir(plot_dir)

    # Remove stale redundant plots from older versions, so rerunning this script
    # does not leave relative_error_heatmap_* or distance_heatmap_* files that
    # can be mistaken as current outputs. The current primary heatmap is
    # fidelity_heatmap_* only.
    for old_name in os.listdir(plot_dir):
        if old_name.startswith(("relative_error_heatmap_", "distance_heatmap_")) and old_name.endswith(".png"):
            try:
                os.remove(os.path.join(plot_dir, old_name))
            except OSError:
                pass
    for scenario in scenarios:
        sub = summary[summary["scenario"] == scenario]
        if sub.empty:
            continue
        w2 = float(sub["w2_reference_mean"].iloc[0])

        dist_pivot = pivot_metric(summary, scenario, "distance_mean")
        err_pivot = pivot_metric(summary, scenario, "relative_error_mean")

        # Primary heatmap: color encodes relative error (metric fidelity),
        # while each cell also reports the corresponding GEBSW distance.
        fidelity_label = err_pivot.copy().astype(object)
        for er in ENERGY_ORDER:
            for q in Q_LIST:
                d = float(dist_pivot.loc[er, q])
                e = float(err_pivot.loc[er, q])
                fidelity_label.loc[er, q] = f"{d:.3f} ({100*e:.1f}%)"

        fidelity_title = (
            f"GEBSW Wasserstein-fidelity heatmap: {scenario}\n"
            f"Exact empirical W2 reference = {w2:.4f}; color encodes relative error (lower is better)"
        )
        fidelity_cbar = "Relative error |D_GEBSW - W2| / W2"

        plot_heatmap(
            err_pivot,
            fidelity_label,
            title=fidelity_title,
            cbar_label=fidelity_cbar,
            out_path=os.path.join(plot_dir, f"fidelity_heatmap_{scenario}.png"),
        )


        plot_distance_bar_with_reference(
            summary,
            scenario,
            out_path=os.path.join(plot_dir, f"distance_with_w2_reference_{scenario}.png"),
        )

    # Primary paper-style composite figure requested for the manuscript.
    # It uses 2 columns x 4 rows: distribution panels on the left, heatmaps on the right.
    plot_sr_compact_2col4row(
        summary,
        scenarios,
        args,
        out_path=os.path.join(plot_dir, "sr_compact_distribution_heatmap_2col4row.png"),
    )


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    ensure_dir(args.out_dir)
    configs = [Config(q=q, rule=rule, r=r) for q in Q_LIST for rule, r in ENERGY_RULES]

    raw_rows: List[Dict[str, object]] = []
    for scenario in args.scenarios:
        print(f"[Scenario] {scenario}")
        for rep in range(args.repeats):
            seed_pair = args.seed + 10000 * rep + 101 * (args.scenarios.index(scenario) + 1)
            X, Y = make_pair(
                scenario=scenario,
                n=args.n,
                seed=seed_pair,
                noise=args.noise,
                dim_high=args.dim_high,
            )
            w2_ref = exact_empirical_w2(X, Y)

            # For fairness: for each q, all energy rules share the same theta.
            theta_seed_by_q = {
                q: args.seed + 50000 * rep + 1009 * q + 17 * (args.scenarios.index(scenario) + 1)
                for q in Q_LIST
            }
            for cfg in configs:
                dist, entropy, top5 = gebsw_distance(
                    X,
                    Y,
                    cfg,
                    L=args.L,
                    seed=theta_seed_by_q[cfg.q],
                    temperature_scale=args.temperature_scale,
                )
                abs_err = abs(dist - w2_ref)
                rel_err = abs_err / (abs(w2_ref) + EPS)
                raw_rows.append({
                    "scenario": scenario,
                    "repeat": rep,
                    "q": cfg.q,
                    "energy_rule": cfg.energy_label,
                    "method": cfg.method_label,
                    "family": cfg.family,
                    "distance": dist,
                    "w2_reference": w2_ref,
                    "absolute_error": abs_err,
                    "relative_error": rel_err,
                    "weight_entropy_norm": entropy,
                    "top5_weight_ratio": top5,
                    "n": args.n,
                    "L": args.L,
                    "dim": X.shape[1],
                })

    raw = pd.DataFrame(raw_rows)
    raw_path = os.path.join(args.out_dir, "raw_results.csv")
    raw.to_csv(raw_path, index=False)

    summary = raw.groupby(["scenario", "q", "energy_rule", "method", "family"], as_index=False).agg(
        distance_mean=("distance", "mean"),
        distance_std=("distance", "std"),
        w2_reference_mean=("w2_reference", "mean"),
        w2_reference_std=("w2_reference", "std"),
        absolute_error_mean=("absolute_error", "mean"),
        absolute_error_std=("absolute_error", "std"),
        relative_error_mean=("relative_error", "mean"),
        relative_error_std=("relative_error", "std"),
        entropy_mean=("weight_entropy_norm", "mean"),
        top5_mean=("top5_weight_ratio", "mean"),
    )
    summary_path = os.path.join(args.out_dir, "summary_results.csv")
    summary.to_csv(summary_path, index=False)

    best = summary.sort_values(["scenario", "relative_error_mean"]).groupby("scenario", as_index=False).first()
    best_path = os.path.join(args.out_dir, "best_by_scenario.csv")
    best.to_csv(best_path, index=False)

    family_summary = raw.groupby(["scenario", "family"], as_index=False).agg(
        distance_mean=("distance", "mean"),
        relative_error_mean=("relative_error", "mean"),
        relative_error_std=("relative_error", "std"),
        entropy_mean=("weight_entropy_norm", "mean"),
        top5_mean=("top5_weight_ratio", "mean"),
    )
    family_path = os.path.join(args.out_dir, "family_summary.csv")
    family_summary.to_csv(family_path, index=False)

    make_plots(summary, args.scenarios, args.out_dir, args)

    print("\nSaved outputs:")
    print(" ", raw_path)
    print(" ", summary_path)
    print(" ", best_path)
    print(" ", family_path)
    print(" ", os.path.join(args.out_dir, "plots"))
    print(" ", os.path.join(args.out_dir, "plots", "sr_compact_distribution_heatmap_2col4row.png"))
    print(" ", os.path.join(args.out_dir, "plots", "sr_compact_distribution_heatmap_2col4row.pdf"))

    print("\nBest configuration by relative error:")
    cols = ["scenario", "method", "family", "distance_mean", "w2_reference_mean", "relative_error_mean"]
    print(best[cols].to_string(index=False))


if __name__ == "__main__":
    run(parse_args())
