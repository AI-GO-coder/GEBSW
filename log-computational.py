import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
import argparse
import math
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
ENERGY_RULES = ('C', 'e', '4')
Q_LIST = (1, 3, 5)
EPS = 1e-08

def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(',') if x.strip()]

@lru_cache(maxsize=None)
def exact_degree_multi_indices(d: int, q: int) -> Tuple[Tuple[int, ...], ...]:
    result: List[Tuple[int, ...]] = []

    def rec(prefix: List[int], remaining: int, slots_left: int) -> None:
        if slots_left == 1:
            result.append(tuple(prefix + [remaining]))
            return
        for a in range(remaining + 1):
            rec(prefix + [a], remaining - a, slots_left - 1)
    rec([], q, d)
    return tuple(result)

def feature_dim(d: int, q: int) -> int:
    return len(exact_degree_multi_indices(d, q))

def sample_theta_on_sphere(rng: np.random.Generator, num_projections: int, dim: int) -> np.ndarray:
    theta = rng.normal(size=(num_projections, dim)).astype(np.float64)
    norms = np.linalg.norm(theta, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return theta / norms

def generate_shifted_gaussians(rng: np.random.Generator, n: int, d: int, shift: float=0.2) -> Tuple[np.ndarray, np.ndarray]:
    scale = math.sqrt(1.0 / 3.0)
    x = rng.normal(loc=0.0, scale=scale, size=(n, d)).astype(np.float64)
    y = rng.normal(loc=0.0, scale=scale, size=(n, d)).astype(np.float64)
    y[:, 0] += shift
    return (x, y)

def project_single_theta(x: np.ndarray, theta: np.ndarray, q: int) -> np.ndarray:
    if q == 1:
        return x @ theta
    n, d = x.shape
    indices = exact_degree_multi_indices(d, q)
    projected = np.zeros(n, dtype=np.float64)
    for coeff, alpha in zip(theta, indices):
        term = np.full(n, coeff, dtype=np.float64)
        for j, power in enumerate(alpha):
            if power > 0:
                term *= x[:, j] ** power
        projected += term
    return projected

def energy_weights(costs: np.ndarray, energy_rule: str) -> np.ndarray:
    if energy_rule == 'C':
        return np.ones_like(costs, dtype=np.float64)
    if energy_rule == 'e':
        return np.exp(np.clip(costs, 0.0, 700.0))
    if energy_rule == '4':
        return costs ** 4 + EPS
    raise ValueError(f'Unknown energy_rule: {energy_rule}')

def gebsw_forward_algorithmic_numpy(x: np.ndarray, y: np.ndarray, theta: np.ndarray, q: int, energy_rule: str, p: int=2) -> float:
    L = theta.shape[0]
    costs = np.empty(L, dtype=np.float64)
    for l in range(L):
        projected_x = project_single_theta(x, theta[l], q)
        projected_y = project_single_theta(y, theta[l], q)
        sorted_x = np.sort(projected_x)
        sorted_y = np.sort(projected_y)
        costs[l] = np.mean(np.abs(sorted_x - sorted_y) ** p)
    weights = energy_weights(costs, energy_rule)
    value_p = np.sum(weights * costs) / np.sum(weights)
    return float(value_p ** (1.0 / p))

def time_energy_rules_interleaved(x: np.ndarray, y: np.ndarray, theta: np.ndarray, q: int, p: int, repeats: int, warmups: int, rng: np.random.Generator) -> Dict[str, List[Tuple[float, float]]]:
    results: Dict[str, List[Tuple[float, float]]] = {rule: [] for rule in ENERGY_RULES}
    for _ in range(warmups):
        order = list(ENERGY_RULES)
        rng.shuffle(order)
        for energy_rule in order:
            _ = gebsw_forward_algorithmic_numpy(x=x, y=y, theta=theta, q=q, energy_rule=energy_rule, p=p)
    for _ in range(repeats):
        order = list(ENERGY_RULES)
        rng.shuffle(order)
        for energy_rule in order:
            start = time.perf_counter()
            estimate = gebsw_forward_algorithmic_numpy(x=x, y=y, theta=theta, q=q, energy_rule=energy_rule, p=p)
            end = time.perf_counter()
            runtime_ms = (end - start) * 1000.0
            results[energy_rule].append((runtime_ms, estimate))
    return results

def summarize_runtime(raw_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ['scaling', 'var_value', 'N', 'L', 'd', 'q', 'energy_rule']
    summary = raw_df.groupby(group_cols, as_index=False).agg(median_ms=('runtime_ms', 'median'), q25_ms=('runtime_ms', lambda s: float(np.quantile(s, 0.25))), q75_ms=('runtime_ms', lambda s: float(np.quantile(s, 0.75))), median_estimate=('estimate', 'median'), feature_dim=('feature_dim', 'first'))
    summary['iqr_ms'] = summary['q75_ms'] - summary['q25_ms']
    return summary

def format_sample_count_plain(x: float, pos: int) -> str:
    if x <= 0:
        return '0'
    return f'{int(x)}'

def style_maps():
    color_map = {(1, 'C'): '#1f77b4', (1, 'e'): '#4fa3e3', (1, '4'): '#0d3b66', (3, 'C'): '#ff7f0e', (3, 'e'): '#ffb55a', (3, '4'): '#c65d00', (5, 'C'): '#2ca02c', (5, 'e'): '#66c266', (5, '4'): '#0b6e0b'}
    marker_map = {'C': 'o', 'e': 's', '4': '^'}
    linestyle_map = {'C': '-', 'e': '--', '4': '-.'}
    return (color_map, marker_map, linestyle_map)

def plot_runtime_scaling_combined(summary: pd.DataFrame, out_dir: Path, log_y: bool=True) -> None:
    color_map, marker_map, linestyle_map = style_maps()
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 9.4), constrained_layout=False)
    ax = axes[0]
    sub_l = summary[summary['scaling'] == 'L'].copy()
    for q in Q_LIST:
        for energy in ENERGY_RULES:
            cur = sub_l[(sub_l['q'] == q) & (sub_l['energy_rule'] == energy)].sort_values('var_value')
            ax.plot(cur['var_value'].to_numpy(), cur['median_ms'].to_numpy(), color=color_map[q, energy], marker=marker_map[energy], linestyle=linestyle_map[energy], linewidth=1.8, markersize=5.0, label=f'$q={q}$, {energy}')
    ax.set_xlabel('Number of sampled projections $L$')
    ax.set_ylabel('Median CPU runtime (ms)')
    ax.grid(True, linewidth=0.4, alpha=0.35)
    if log_y:
        ax.set_yscale('log')
    ax = axes[1]
    sub_n = summary[summary['scaling'] == 'N'].copy()
    for q in Q_LIST:
        for energy in ENERGY_RULES:
            cur = sub_n[(sub_n['q'] == q) & (sub_n['energy_rule'] == energy)].sort_values('var_value')
            ax.plot(cur['var_value'].to_numpy(), cur['median_ms'].to_numpy(), color=color_map[q, energy], marker=marker_map[energy], linestyle=linestyle_map[energy], linewidth=1.8, markersize=5.0, label=f'$q={q}$, {energy}')
    ax.set_xlabel('Number of samples $N$')
    ax.set_ylabel('Median CPU runtime (ms)')
    ax.xaxis.set_major_formatter(FuncFormatter(format_sample_count_plain))
    ax.grid(True, linewidth=0.4, alpha=0.35)
    if log_y:
        ax.set_yscale('log')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=(0.03, 0.08, 1.0, 0.98))
    fig.savefig(out_dir / 'runtime_scaling_combined.png', dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / 'runtime_scaling_combined.pdf', bbox_inches='tight')
    plt.close(fig)

def plot_runtime_scaling_dimension(summary: pd.DataFrame, out_dir: Path, log_y: bool=True) -> None:
    color_map, marker_map, linestyle_map = style_maps()
    fig, ax = plt.subplots(1, 1, figsize=(12.5, 5.2), constrained_layout=False)
    sub_d = summary[summary['scaling'] == 'd'].copy()
    for q in Q_LIST:
        for energy in ENERGY_RULES:
            cur = sub_d[(sub_d['q'] == q) & (sub_d['energy_rule'] == energy)].sort_values('var_value')
            ax.plot(cur['var_value'].to_numpy(), cur['median_ms'].to_numpy(), color=color_map[q, energy], marker=marker_map[energy], linestyle=linestyle_map[energy], linewidth=1.8, markersize=5.0, label=f'$q={q}$, {energy}')
    ax.set_xlabel('Ambient dimension $d$')
    ax.set_ylabel('Median CPU runtime (ms)')
    ax.grid(True, linewidth=0.4, alpha=0.35)
    if log_y:
        ax.set_yscale('log')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=(0.03, 0.17, 1.0, 0.98))
    fig.savefig(out_dir / 'runtime_scaling_dimension.png', dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / 'runtime_scaling_dimension.pdf', bbox_inches='tight')
    plt.close(fig)

def run_experiment(args: argparse.Namespace) -> pd.DataFrame:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    records = []
    l_values = parse_int_list(args.l_values)
    n_values = parse_int_list(args.n_values)
    d_values = parse_int_list(args.d_values)
    if args.quick:
        l_values = [20, 100, 200]
        n_values = [512, 2048, 8192]
        d_values = [2, 3, 5]
        args.repeats = min(args.repeats, 5)
        args.warmups = min(args.warmups, 2)
    print('Running GEBSW runtime scaling experiment with q and d studies')
    print(f'Output directory: {out_dir.resolve()}')
    print(f'base d for L/N-scaling = {args.d}, p = {args.p}')
    print(f'L-scaling: fixed N={args.fixed_n}, L values={l_values}')
    print(f'N-scaling: fixed L={args.fixed_l}, N values={n_values}')
    print(f'd-scaling: fixed N={args.fixed_n_d}, fixed L={args.fixed_l_d}, d values={d_values}')
    print(f'q values={Q_LIST}, energy rules={ENERGY_RULES}')
    print(f'repeats={args.repeats}, warmups={args.warmups}')
    print(f'log_y={not args.linear_y}')
    print('Protocol:')
    print('  - CPU-only NumPy timing.')
    print('  - Per-projection evaluation matches the algorithmic complexity analysis.')
    print('  - L-scaling uses one shared dataset and nested projection sets.')
    print('  - N-scaling uses the same dataset across q and energy rules for each N.')
    print('  - d-scaling uses the same dataset across q and energy rules for each d.')
    print('  - Energy-rule timing is interleaved and randomized.')
    print('  - Data generation, optimization, and backpropagation are excluded from timing.')
    x_l, y_l = generate_shifted_gaussians(rng, n=args.fixed_n, d=args.d, shift=args.shift)
    theta_max_by_q: Dict[int, np.ndarray] = {}
    for q in Q_LIST:
        theta_max_by_q[q] = sample_theta_on_sphere(rng, num_projections=max(l_values), dim=feature_dim(args.d, q))
    datasets_by_n: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for n_value in n_values:
        datasets_by_n[n_value] = generate_shifted_gaussians(rng, n=n_value, d=args.d, shift=args.shift)
    theta_by_n_q: Dict[Tuple[int, int], np.ndarray] = {}
    for n_value in n_values:
        for q in Q_LIST:
            theta_by_n_q[n_value, q] = sample_theta_on_sphere(rng, num_projections=args.fixed_l, dim=feature_dim(args.d, q))
    datasets_by_d: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for d_value in d_values:
        datasets_by_d[d_value] = generate_shifted_gaussians(rng, n=args.fixed_n_d, d=d_value, shift=args.shift)
    theta_by_d_q: Dict[Tuple[int, int], np.ndarray] = {}
    for d_value in d_values:
        for q in Q_LIST:
            theta_by_d_q[d_value, q] = sample_theta_on_sphere(rng, num_projections=args.fixed_l_d, dim=feature_dim(d_value, q))
    l_conditions = [(q, l_value) for q in Q_LIST for l_value in l_values]
    n_conditions = [(n_value, q) for n_value in n_values for q in Q_LIST]
    d_conditions = [(d_value, q) for d_value in d_values for q in Q_LIST]
    rng.shuffle(l_conditions)
    rng.shuffle(n_conditions)
    rng.shuffle(d_conditions)
    total_conditions = len(l_conditions) + len(n_conditions) + len(d_conditions)
    completed = 0
    for q, l_value in l_conditions:
        dim_theta = feature_dim(args.d, q)
        theta = theta_max_by_q[q][:l_value]
        timed = time_energy_rules_interleaved(x=x_l, y=y_l, theta=theta, q=q, p=args.p, repeats=args.repeats, warmups=args.warmups, rng=rng)
        for energy_rule, values in timed.items():
            for rep_id, (runtime_ms, estimate) in enumerate(values, start=1):
                records.append({'scaling': 'L', 'var_value': l_value, 'N': args.fixed_n, 'L': l_value, 'd': args.d, 'q': q, 'energy_rule': energy_rule, 'repeat': rep_id, 'runtime_ms': runtime_ms, 'estimate': estimate, 'feature_dim': dim_theta})
        completed += 1
        medians = {rule: np.median([v[0] for v in timed[rule]]) for rule in ENERGY_RULES}
        print(f'[{completed:03d}/{total_conditions:03d}] L-scaling q={q}, L={l_value}: ' + ', '.join([f'{rule}={medians[rule]:.3f} ms' for rule in ENERGY_RULES]))
    for n_value, q in n_conditions:
        x_n, y_n = datasets_by_n[n_value]
        dim_theta = feature_dim(args.d, q)
        theta = theta_by_n_q[n_value, q]
        timed = time_energy_rules_interleaved(x=x_n, y=y_n, theta=theta, q=q, p=args.p, repeats=args.repeats, warmups=args.warmups, rng=rng)
        for energy_rule, values in timed.items():
            for rep_id, (runtime_ms, estimate) in enumerate(values, start=1):
                records.append({'scaling': 'N', 'var_value': n_value, 'N': n_value, 'L': args.fixed_l, 'd': args.d, 'q': q, 'energy_rule': energy_rule, 'repeat': rep_id, 'runtime_ms': runtime_ms, 'estimate': estimate, 'feature_dim': dim_theta})
        completed += 1
        medians = {rule: np.median([v[0] for v in timed[rule]]) for rule in ENERGY_RULES}
        print(f'[{completed:03d}/{total_conditions:03d}] N-scaling q={q}, N={n_value}: ' + ', '.join([f'{rule}={medians[rule]:.3f} ms' for rule in ENERGY_RULES]))
    for d_value, q in d_conditions:
        x_d, y_d = datasets_by_d[d_value]
        dim_theta = feature_dim(d_value, q)
        theta = theta_by_d_q[d_value, q]
        timed = time_energy_rules_interleaved(x=x_d, y=y_d, theta=theta, q=q, p=args.p, repeats=args.repeats, warmups=args.warmups, rng=rng)
        for energy_rule, values in timed.items():
            for rep_id, (runtime_ms, estimate) in enumerate(values, start=1):
                records.append({'scaling': 'd', 'var_value': d_value, 'N': args.fixed_n_d, 'L': args.fixed_l_d, 'd': d_value, 'q': q, 'energy_rule': energy_rule, 'repeat': rep_id, 'runtime_ms': runtime_ms, 'estimate': estimate, 'feature_dim': dim_theta})
        completed += 1
        medians = {rule: np.median([v[0] for v in timed[rule]]) for rule in ENERGY_RULES}
        print(f'[{completed:03d}/{total_conditions:03d}] d-scaling q={q}, d={d_value}: ' + ', '.join([f'{rule}={medians[rule]:.3f} ms' for rule in ENERGY_RULES]))
    raw_df = pd.DataFrame.from_records(records)
    raw_df.to_csv(out_dir / 'runtime_raw.csv', index=False)
    summary = summarize_runtime(raw_df)
    summary.to_csv(out_dir / 'runtime_summary.csv', index=False)
    plot_runtime_scaling_combined(summary=summary, out_dir=out_dir, log_y=not args.linear_y)
    plot_runtime_scaling_dimension(summary=summary, out_dir=out_dir, log_y=not args.linear_y)
    print('Done.')
    print(f"Saved raw results to: {out_dir / 'runtime_raw.csv'}")
    print(f"Saved summary to: {out_dir / 'runtime_summary.csv'}")
    print(f"Saved figure: {out_dir / 'runtime_scaling_combined.png'}")
    print(f"Saved figure: {out_dir / 'runtime_scaling_combined.pdf'}")
    print(f"Saved figure: {out_dir / 'runtime_scaling_dimension.png'}")
    print(f"Saved figure: {out_dir / 'runtime_scaling_dimension.pdf'}")
    return raw_df

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='GEBSW runtime scaling experiment with q and d studies.')
    parser.add_argument('--output_dir', type=str, default='gebsw_runtime_scaling_q_d')
    parser.add_argument('--seed', type=int, default=20260615)
    parser.add_argument('--d', type=int, default=3)
    parser.add_argument('--p', type=int, default=2)
    parser.add_argument('--shift', type=float, default=0.2)
    parser.add_argument('--fixed_n', type=int, default=2048)
    parser.add_argument('--fixed_l', type=int, default=50)
    parser.add_argument('--fixed_n_d', type=int, default=1024)
    parser.add_argument('--fixed_l_d', type=int, default=20)
    parser.add_argument('--l_values', type=str, default='20,30,40,50,60,70,80,90,100,120,140,160,180,200', help='Comma-separated L values for L-scaling.')
    parser.add_argument('--n_values', type=str, default='512,1024,1536,2048,3072,4096,6144,8192,12288,16384', help='Comma-separated N values for N-scaling.')
    parser.add_argument('--d_values', type=str, default='2,3,4,5,6,7,8,9,10', help='Comma-separated d values for d-scaling.')
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--warmups', type=int, default=3)
    parser.add_argument('--quick', action='store_true', help='Run a shorter version for testing the script.')
    parser.add_argument('--linear_y', action='store_true', help='Use linear y-axis instead of logarithmic y-axis.')
    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_experiment(args)
if __name__ == '__main__':
    main()
