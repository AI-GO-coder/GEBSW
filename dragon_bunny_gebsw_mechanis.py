import os
import warnings
import struct
from typing import Dict, List, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import griddata
from scipy.stats import ttest_rel
warnings.filterwarnings('ignore')
SEED = 42
np.random.seed(SEED)
DRAGON_PATH = os.environ.get('DRAGON_PATH', 'DRAGON_PATH')
BUNNY_PATH = os.environ.get('BUNNY_PATH', 'BUNNY_PATH')
RESULT_DIR = 'gebsw_synergy-3x5'
os.makedirs(RESULT_DIR, exist_ok=True)
COLOR_C1 = '#000000'
COLOR_C3 = '#2E8B57'
COLOR_E1 = '#E67300'
COLOR_E3 = '#C70039'
COLOR_E5 = '#8B4513'

def load_ply_robust(filepath: str, n_points: int=2048) -> np.ndarray:
    try:
        with open(filepath, 'rb') as f:
            header_lines = []
            while True:
                line = f.readline().decode('ascii', errors='ignore').strip()
                header_lines.append(line)
                if line == 'end_header':
                    break
            n_vertices = 0
            format_type = 'ascii'
            for line in header_lines:
                parts = line.split()
                if parts[0] == 'element' and parts[1] == 'vertex':
                    n_vertices = int(parts[2])
                elif parts[0] == 'format':
                    format_type = parts[1]
            vertices = []
            if format_type == 'ascii':
                for _ in range(n_vertices):
                    line = f.readline().decode('ascii', errors='ignore').strip()
                    if line:
                        values = line.split()
                        if len(values) >= 3:
                            vertices.append([float(values[0]), float(values[1]), float(values[2])])
            else:
                for _ in range(n_vertices):
                    data = f.read(12)
                    if len(data) == 12:
                        values = struct.unpack('<fff', data)
                        vertices.append([values[0], values[1], values[2]])
            points = np.array(vertices, dtype=np.float32)
            if len(points) > n_points:
                rng = np.random.RandomState(SEED)
                indices = rng.choice(len(points), n_points, replace=False)
                points = points[indices]
            elif len(points) < n_points:
                rng = np.random.RandomState(SEED)
                extra = rng.choice(len(points), n_points - len(points), replace=True)
                points = np.vstack([points, points[extra]])
            points = points - points.mean(axis=0)
            return points
    except Exception as e:
        print(f'[Warning] Failed to load {filepath}: {e}')
        return None

def generate_fallback_data(name: str, n_points: int=2048) -> np.ndarray:
    rng = np.random.RandomState(SEED)
    if name == 'Dragon':
        n_body = int(n_points * 0.6)
        body = rng.randn(n_body, 3) * np.array([1.2, 0.5, 0.4])
        n_head = int(n_points * 0.2)
        head = rng.randn(n_head, 3) * 0.3 + np.array([1.5, 0.2, 0.1])
        n_tail = n_points - n_body - n_head
        t = np.linspace(0, 3 * np.pi, n_tail)
        tail = np.array([-1.0 - 0.2 * t, 0.15 * np.sin(t), 0.1 * np.cos(2 * t)]).T
        pts = np.vstack([body, head, tail])
    elif name == 'Bunny':
        n_body = int(n_points * 0.7)
        body = rng.randn(n_body, 3) * np.array([0.55, 0.38, 0.45]) + np.array([0.0, -0.08, 0.0])
        n_head = int(n_points * 0.16)
        head = rng.randn(n_head, 3) * np.array([0.26, 0.22, 0.25]) + np.array([0.42, 0.28, 0.2])
        n_ear = (n_points - n_body - n_head) // 2
        ear1 = rng.randn(n_ear, 3) * np.array([0.08, 0.1, 0.34]) + np.array([0.44, 0.5, 0.62])
        ear2 = rng.randn(n_points - n_body - n_head - n_ear, 3) * np.array([0.08, 0.1, 0.34]) + np.array([0.6, 0.47, 0.54])
        pts = np.vstack([body, head, ear1, ear2])
    else:
        phi = rng.uniform(0, np.pi, n_points)
        theta = rng.uniform(0, 2 * np.pi, n_points)
        x = 0.8 * np.sin(phi) * np.cos(theta)
        y = 1.0 * np.sin(phi) * np.sin(theta)
        z = 0.9 * np.cos(phi) + 0.2 * np.sin(3 * theta)
        pts = np.column_stack([x, y, z])
    pts = pts - pts.mean(axis=0)
    return pts.astype(np.float32)

def normalize_point_cloud_unit(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).copy()
    points = points - points.mean(axis=0, keepdims=True)
    radius = np.percentile(np.linalg.norm(points, axis=1), 95)
    if not np.isfinite(radius) or radius < 1e-12:
        radius = np.max(np.linalg.norm(points, axis=1)) + 1e-12
    points = points / (radius + 1e-12)
    return points.astype(np.float32)

def apply_twist_deform(source: np.ndarray, strength: float=1.5) -> np.ndarray:
    target = source.copy()
    x, y, z = (target[:, 0], target[:, 1], target[:, 2])
    angle = strength * np.pi * (z + 1.0) / 2.0
    cos_a, sin_a = (np.cos(angle), np.sin(angle))
    x_new = x * cos_a - y * sin_a
    y_new = x * sin_a + y * cos_a
    z_new = z + 0.3 * np.sin(4 * np.pi * x)
    target[:, 0] = x_new
    target[:, 1] = y_new
    target[:, 2] = z_new
    r = np.sqrt(x_new ** 2 + y_new ** 2)
    inflate = 1 + 0.15 * np.sin(5 * r) * strength
    target[:, :2] *= inflate[:, np.newaxis]
    return target.astype(np.float32)

class GEBSW_Metric:

    def __init__(self, n_projections: int=128, poly_order: int=1, use_energy_weight: bool=True, temperature: float=None, hidden_dim: int=64, seed: int=42):
        self.n_projections = n_projections
        self.poly_order = poly_order
        self.use_energy_weight = use_energy_weight
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.seed = seed
        rng = np.random.RandomState(seed)
        self.feature_dim = self._compute_feature_dim(poly_order)
        if poly_order > 1:
            self.use_nonlinear = True
            self.feature_transform = rng.randn(self.feature_dim, hidden_dim).astype(np.float32) * 0.1
            self.proj_matrix = rng.randn(hidden_dim, n_projections).astype(np.float32)
            self.proj_matrix = self.proj_matrix / np.linalg.norm(self.proj_matrix, axis=0, keepdims=True)
        else:
            self.use_nonlinear = False
            self.feature_transform = None
            self.proj_matrix = rng.randn(3, n_projections).astype(np.float32)
            self.proj_matrix = self.proj_matrix / np.linalg.norm(self.proj_matrix, axis=0, keepdims=True)

    def _compute_feature_dim(self, order: int) -> int:
        if order == 1:
            return 3
        elif order == 2:
            return 9
        elif order == 3:
            return 19
        elif order == 4:
            return 34
        elif order == 5:
            return 55
        else:
            raise ValueError(f'Unsupported order: {order}')

    def _polynomial_features(self, X: np.ndarray) -> np.ndarray:
        x, y, z = (X[:, 0], X[:, 1], X[:, 2])
        features = [x, y, z]
        if self.poly_order >= 2:
            features.extend([x ** 2, y ** 2, z ** 2, x * y, x * z, y * z])
        if self.poly_order >= 3:
            features.extend([x ** 3, y ** 3, z ** 3, x ** 2 * y, x ** 2 * z, x * y ** 2, y ** 2 * z, x * z ** 2, y * z ** 2, x * y * z])
        if self.poly_order >= 4:
            features.extend([x ** 4, y ** 4, z ** 4, x ** 3 * y, x ** 3 * z, x * y ** 3, y ** 3 * z, x * z ** 3, y * z ** 3, x ** 2 * y ** 2, x ** 2 * z ** 2, y ** 2 * z ** 2, x ** 2 * y * z, x * y ** 2 * z, x * y * z ** 2])
        if self.poly_order >= 5:
            features.extend([x ** 5, y ** 5, z ** 5, x ** 4 * y, x ** 4 * z, x * y ** 4, y ** 4 * z, x * z ** 4, y * z ** 4, x ** 3 * y ** 2, x ** 3 * z ** 2, x ** 2 * y ** 3, y ** 3 * z ** 2, x ** 2 * z ** 3, y ** 2 * z ** 3, x ** 3 * y * z, x * y ** 3 * z, x * y * z ** 3, x ** 2 * y ** 2 * z, x ** 2 * y * z ** 2, x * y ** 2 * z ** 2])
        feat = np.column_stack(features)
        feat_mean = feat.mean(axis=0)
        feat_std = feat.std(axis=0) + 1e-08
        feat = (feat - feat_mean) / feat_std
        return feat

    def compute(self, X: np.ndarray, Y: np.ndarray) -> Dict:
        if self.use_nonlinear:
            X_feat = self._polynomial_features(X)
            Y_feat = self._polynomial_features(Y)
            X_proj_feat = np.tanh(X_feat @ self.feature_transform)
            Y_proj_feat = np.tanh(Y_feat @ self.feature_transform)
        else:
            X_proj_feat = X
            Y_proj_feat = Y
        assert X_proj_feat.shape[1] == self.proj_matrix.shape[0], f'Dimension mismatch: features {X_proj_feat.shape[1]} vs proj {self.proj_matrix.shape[0]}'
        proj_X = X_proj_feat @ self.proj_matrix
        proj_Y = Y_proj_feat @ self.proj_matrix
        w_dists = []
        for i in range(self.n_projections):
            x_s = np.sort(proj_X[:, i])
            y_s = np.sort(proj_Y[:, i])
            w_dist = np.mean(np.abs(x_s - y_s))
            w_dists.append(max(w_dist, 1e-10))
        w_array = np.array(w_dists, dtype=np.float64)
        if self.use_energy_weight:
            if self.temperature is None:
                temp = np.std(w_array) * 0.5 + 1e-08
            else:
                temp = self.temperature
            w_max = np.max(w_array)
            exp_w = np.exp((w_array - w_max) / temp)
            weights = exp_w / np.sum(exp_w)
            eps = 1e-12
            valid_mask = weights > eps
            if valid_mask.sum() > 0:
                valid_weights = weights[valid_mask]
                valid_weights = valid_weights / valid_weights.sum()
                entropy = -np.sum(valid_weights * np.log(valid_weights))
                max_entropy = np.log(len(valid_weights))
                concentration = max(0.0, min(1.0, 1.0 - entropy / max_entropy))
            else:
                concentration = 0.0
        else:
            weights = np.ones(self.n_projections) / self.n_projections
            concentration = 0.0
        gebsw = float(np.sum(w_array * weights))
        return {'gebsw': gebsw, 'w_array': w_array, 'weights': weights, 'concentration': concentration, 'temperature': temp if self.use_energy_weight else None, 'mean_w': float(np.mean(w_array)), 'std_w': float(np.std(w_array)), 'min_w': float(np.min(w_array)), 'max_w': float(np.max(w_array)), 'top5_ratio': float(np.sum(np.sort(weights)[-5:])), 'proj_X': proj_X, 'proj_Y': proj_Y}

def run_multiple_seeds(X: np.ndarray, Y: np.ndarray, metric_class, config: Dict, n_runs: int=10) -> Dict:
    results_list = []
    for run_idx in range(n_runs):
        seed = SEED + run_idx * 100
        metric = metric_class(seed=seed, **config)
        res = metric.compute(X, Y)
        results_list.append(res)
    gebsw_vals = [r['gebsw'] for r in results_list]
    conc_vals = [r['concentration'] for r in results_list]
    top5_vals = [r['top5_ratio'] for r in results_list]
    aggregated = {'gebsw_mean': np.mean(gebsw_vals), 'gebsw_std': np.std(gebsw_vals), 'gebsw_ci': (np.percentile(gebsw_vals, 2.5), np.percentile(gebsw_vals, 97.5)), 'concentration_mean': np.mean(conc_vals), 'concentration_std': np.std(conc_vals), 'top5_mean': np.mean(top5_vals), 'top5_std': np.std(top5_vals), 'last_run': results_list[-1], 'all_runs': results_list}
    return aggregated

def create_projection_figure(agg_results: Dict, datasets: List[Dict], save_path: str):
    matrix_config = [('GEBSW(C,1)', COLOR_C1, 'GEBSW(C,1) (SW-baseline)'), ('GEBSW(C,3)', COLOR_C3, 'GEBSW(C,3) (GSW-baseline)'), ('GEBSW(e,1)', COLOR_E1, 'GEBSW(e,1) (EBSW-baseline)'), ('GEBSW(e,3)', COLOR_E3, 'GEBSW(e,3) (Ours)'), ('GEBSW(e,5)', COLOR_E5, 'GEBSW(e,5) (Ours)')]
    fig = plt.figure(figsize=(16, 32))
    fig.patch.set_facecolor('white')
    gs = GridSpec(5, 2, figure=fig, hspace=0.15, wspace=0.15, left=0.06, right=0.94, top=0.92, bottom=0.06)
    dataset = datasets[0]
    for idx, (method, color, label) in enumerate(matrix_config):
        ax_proj = fig.add_subplot(gs[idx, 0])
        key = (dataset['name'], method)
        res = agg_results[key]['last_run']
        source = dataset['source']
        best_idx = np.argmax(res['weights'])
        proj_vals = res['proj_X'][:, best_idx]
        scatter = ax_proj.scatter(source[:, 0], source[:, 1], c=proj_vals, cmap='viridis', s=25, alpha=0.85, edgecolors='none')
        try:
            xi = np.linspace(source[:, 0].min(), source[:, 0].max(), 50)
            yi = np.linspace(source[:, 1].min(), source[:, 1].max(), 50)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = griddata((source[:, 0], source[:, 1]), proj_vals, (Xi, Yi), method='linear', fill_value=0)
            if Zi is not None and (not np.all(Zi == 0)):
                ax_proj.contour(Xi, Yi, Zi, levels=8, colors='white', alpha=0.6, linewidths=0.8)
        except Exception:
            pass
        ax_proj.set_title(f'{label}', fontsize=11, fontweight='bold', color=color)
        ax_proj.set_xlabel('X')
        ax_proj.set_ylabel('Y')
        ax_proj.axis('equal')
        ax_weight = fig.add_subplot(gs[idx, 1])
        weights = res['weights']
        w_array = res['w_array']
        sorted_idx = np.argsort(weights)[::-1]
        sorted_w = weights[sorted_idx]
        sorted_d = w_array[sorted_idx]
        if not np.any(weights != weights[0]):
            ax_weight.bar(range(len(weights)), [1.0 / len(weights)] * len(weights), color=color, alpha=0.6, label='Uniform')
            ax_weight.set_title('Uniform Weights (No Energy Focus)', fontsize=10)
        else:
            ax_weight_twin = ax_weight.twinx()
            bars = ax_weight.bar(range(len(weights)), sorted_w, color=color, alpha=0.7, label='Weight')
            line = ax_weight_twin.plot(range(len(weights)), sorted_d, 'o-', color='darkblue', markersize=3, alpha=0.6, label='W₁ Distance')
            top5_sum = np.sum(sorted_w[:5])
            ax_weight.axvline(x=4.5, color='red', linestyle='--', alpha=0.5)
            ax_weight.text(4.5, ax_weight.get_ylim()[1] * 0.9, 'Top-5', ha='center', color='red', fontweight='bold')
            ax_weight.set_title(f'Energy Weights (Top-5 Sum: {top5_sum:.3f})', fontsize=10)
            ax_weight_twin.set_ylabel('W₁ Distance', color='darkblue')
            ax_weight_twin.tick_params(axis='y', labelcolor='darkblue')
        ax_weight.set_xlabel('Projection Rank (by Weight)')
        ax_weight.set_ylabel('Weight')
        ax_weight.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[Saved] Projection & Weight figure (2x5): {save_path}')

def create_projection_figure_combined_top3(agg_results: Dict, datasets: List[Dict], save_path: str):
    matrix_config = [('GEBSW(C,1)', COLOR_C1, 'GEBSW(C,1) (SW-baseline)'), ('GEBSW(C,3)', COLOR_C3, 'GEBSW(C,3) (GSW-baseline)'), ('GEBSW(e,1)', COLOR_E1, 'GEBSW(e,1) (EBSW-baseline)')]
    hspace = 0.15
    fig = plt.figure(figsize=(16, 24))
    fig.patch.set_facecolor('white')
    gs = GridSpec(3, 2, figure=fig, hspace=hspace, wspace=0.15, left=0.06, right=0.94, top=0.92, bottom=0.06)
    dataset = datasets[0]
    for idx, (method, color, label) in enumerate(matrix_config):
        ax_proj = fig.add_subplot(gs[idx, 0])
        key = (dataset['name'], method)
        res = agg_results[key]['last_run']
        source = dataset['source']
        best_idx = np.argmax(res['weights'])
        proj_vals = res['proj_X'][:, best_idx]
        scatter = ax_proj.scatter(source[:, 0], source[:, 1], c=proj_vals, cmap='viridis', s=25, alpha=0.85, edgecolors='none')
        try:
            xi = np.linspace(source[:, 0].min(), source[:, 0].max(), 50)
            yi = np.linspace(source[:, 1].min(), source[:, 1].max(), 50)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = griddata((source[:, 0], source[:, 1]), proj_vals, (Xi, Yi), method='linear', fill_value=0)
            if Zi is not None and (not np.all(Zi == 0)):
                ax_proj.contour(Xi, Yi, Zi, levels=8, colors='white', alpha=0.6, linewidths=0.8)
        except Exception:
            pass
        ax_proj.set_title(f'{label}', fontsize=11, fontweight='bold', color=color)
        ax_proj.set_xlabel('X')
        ax_proj.set_ylabel('Y')
        ax_proj.axis('equal')
        ax_weight = fig.add_subplot(gs[idx, 1])
        weights = res['weights']
        w_array = res['w_array']
        sorted_idx = np.argsort(weights)[::-1]
        sorted_w = weights[sorted_idx]
        sorted_d = w_array[sorted_idx]
        if not np.any(weights != weights[0]):
            ax_weight.bar(range(len(weights)), [1.0 / len(weights)] * len(weights), color=color, alpha=0.6, label='Uniform')
            ax_weight.set_title('Uniform Weights (No Energy Focus)', fontsize=10)
        else:
            ax_weight_twin = ax_weight.twinx()
            bars = ax_weight.bar(range(len(weights)), sorted_w, color=color, alpha=0.7, label='Weight')
            line = ax_weight_twin.plot(range(len(weights)), sorted_d, 'o-', color='darkblue', markersize=3, alpha=0.6, label='W₁ Distance')
            top5_sum = np.sum(sorted_w[:5])
            ax_weight.axvline(x=4.5, color='red', linestyle='--', alpha=0.5)
            ax_weight.text(4.5, ax_weight.get_ylim()[1] * 0.9, 'Top-5', ha='center', color='red', fontweight='bold')
            ax_weight.set_title(f'Energy Weights (Top-5 Sum: {top5_sum:.3f})', fontsize=10)
            ax_weight_twin.set_ylabel('W₁ Distance', color='darkblue')
            ax_weight_twin.tick_params(axis='y', labelcolor='darkblue')
        ax_weight.set_xlabel('Projection Rank (by Weight)')
        ax_weight.set_ylabel('Weight')
        ax_weight.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[Saved] Projection & Weight figure (Combined Top 3, 8mm spacing): {save_path}')

def create_projection_figure_combined_last3(agg_results: Dict, datasets: List[Dict], save_path: str):
    matrix_config = [('GEBSW(e,1)', COLOR_E1, 'GEBSW(e,1) (EBSW-baseline)'), ('GEBSW(e,3)', COLOR_E3, 'GEBSW(e,3) (Ours)'), ('GEBSW(e,5)', COLOR_E5, 'GEBSW(e,5) (Ours)')]
    hspace = 0.15
    fig = plt.figure(figsize=(16, 24))
    fig.patch.set_facecolor('white')
    gs = GridSpec(3, 2, figure=fig, hspace=hspace, wspace=0.15, left=0.06, right=0.94, top=0.92, bottom=0.06)
    dataset = datasets[0]
    for idx, (method, color, label) in enumerate(matrix_config):
        ax_proj = fig.add_subplot(gs[idx, 0])
        key = (dataset['name'], method)
        res = agg_results[key]['last_run']
        source = dataset['source']
        best_idx = np.argmax(res['weights'])
        proj_vals = res['proj_X'][:, best_idx]
        scatter = ax_proj.scatter(source[:, 0], source[:, 1], c=proj_vals, cmap='viridis', s=25, alpha=0.85, edgecolors='none')
        try:
            xi = np.linspace(source[:, 0].min(), source[:, 0].max(), 50)
            yi = np.linspace(source[:, 1].min(), source[:, 1].max(), 50)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = griddata((source[:, 0], source[:, 1]), proj_vals, (Xi, Yi), method='linear', fill_value=0)
            if Zi is not None and (not np.all(Zi == 0)):
                ax_proj.contour(Xi, Yi, Zi, levels=8, colors='white', alpha=0.6, linewidths=0.8)
        except Exception:
            pass
        ax_proj.set_title(f'{label}', fontsize=11, fontweight='bold', color=color)
        ax_proj.set_xlabel('X')
        ax_proj.set_ylabel('Y')
        ax_proj.axis('equal')
        ax_weight = fig.add_subplot(gs[idx, 1])
        weights = res['weights']
        w_array = res['w_array']
        sorted_idx = np.argsort(weights)[::-1]
        sorted_w = weights[sorted_idx]
        sorted_d = w_array[sorted_idx]
        if not np.any(weights != weights[0]):
            ax_weight.bar(range(len(weights)), [1.0 / len(weights)] * len(weights), color=color, alpha=0.6, label='Uniform')
            ax_weight.set_title('Uniform Weights (No Energy Focus)', fontsize=10)
        else:
            ax_weight_twin = ax_weight.twinx()
            bars = ax_weight.bar(range(len(weights)), sorted_w, color=color, alpha=0.7, label='Weight')
            line = ax_weight_twin.plot(range(len(weights)), sorted_d, 'o-', color='darkblue', markersize=3, alpha=0.6, label='W₁ Distance')
            top5_sum = np.sum(sorted_w[:5])
            ax_weight.axvline(x=4.5, color='red', linestyle='--', alpha=0.5)
            ax_weight.text(4.5, ax_weight.get_ylim()[1] * 0.9, 'Top-5', ha='center', color='red', fontweight='bold')
            ax_weight.set_title(f'Energy Weights (Top-5 Sum: {top5_sum:.3f})', fontsize=10)
            ax_weight_twin.set_ylabel('W₁ Distance', color='darkblue')
            ax_weight_twin.tick_params(axis='y', labelcolor='darkblue')
        ax_weight.set_xlabel('Projection Rank (by Weight)')
        ax_weight.set_ylabel('Weight')
        ax_weight.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[Saved] Projection & Weight figure (Combined Last 3, 8mm spacing): {save_path}')

def create_projection_figure_combined_last2(agg_results: Dict, datasets: List[Dict], save_path: str):
    matrix_config = [('GEBSW(e,3)', COLOR_E3, 'GEBSW(e,3) (Ours)'), ('GEBSW(e,5)', COLOR_E5, 'GEBSW(e,5) (Ours)')]
    hspace = 0.15
    fig = plt.figure(figsize=(16, 16))
    fig.patch.set_facecolor('white')
    gs = GridSpec(2, 2, figure=fig, hspace=hspace, wspace=0.15, left=0.06, right=0.94, top=0.92, bottom=0.06)
    dataset = datasets[0]
    for idx, (method, color, label) in enumerate(matrix_config):
        ax_proj = fig.add_subplot(gs[idx, 0])
        key = (dataset['name'], method)
        res = agg_results[key]['last_run']
        source = dataset['source']
        best_idx = np.argmax(res['weights'])
        proj_vals = res['proj_X'][:, best_idx]
        scatter = ax_proj.scatter(source[:, 0], source[:, 1], c=proj_vals, cmap='viridis', s=25, alpha=0.85, edgecolors='none')
        try:
            xi = np.linspace(source[:, 0].min(), source[:, 0].max(), 50)
            yi = np.linspace(source[:, 1].min(), source[:, 1].max(), 50)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = griddata((source[:, 0], source[:, 1]), proj_vals, (Xi, Yi), method='linear', fill_value=0)
            if Zi is not None and (not np.all(Zi == 0)):
                ax_proj.contour(Xi, Yi, Zi, levels=8, colors='white', alpha=0.6, linewidths=0.8)
        except Exception:
            pass
        ax_proj.set_title(f'{label}', fontsize=11, fontweight='bold', color=color)
        ax_proj.set_xlabel('X')
        ax_proj.set_ylabel('Y')
        ax_proj.axis('equal')
        ax_weight = fig.add_subplot(gs[idx, 1])
        weights = res['weights']
        w_array = res['w_array']
        sorted_idx = np.argsort(weights)[::-1]
        sorted_w = weights[sorted_idx]
        sorted_d = w_array[sorted_idx]
        if not np.any(weights != weights[0]):
            ax_weight.bar(range(len(weights)), [1.0 / len(weights)] * len(weights), color=color, alpha=0.6, label='Uniform')
            ax_weight.set_title('Uniform Weights (No Energy Focus)', fontsize=10)
        else:
            ax_weight_twin = ax_weight.twinx()
            bars = ax_weight.bar(range(len(weights)), sorted_w, color=color, alpha=0.7, label='Weight')
            line = ax_weight_twin.plot(range(len(weights)), sorted_d, 'o-', color='darkblue', markersize=3, alpha=0.6, label='W₁ Distance')
            top5_sum = np.sum(sorted_w[:5])
            ax_weight.axvline(x=4.5, color='red', linestyle='--', alpha=0.5)
            ax_weight.text(4.5, ax_weight.get_ylim()[1] * 0.9, 'Top-5', ha='center', color='red', fontweight='bold')
            ax_weight.set_title(f'Energy Weights (Top-5 Sum: {top5_sum:.3f})', fontsize=10)
            ax_weight_twin.set_ylabel('W₁ Distance', color='darkblue')
            ax_weight_twin.tick_params(axis='y', labelcolor='darkblue')
        ax_weight.set_xlabel('Projection Rank (by Weight)')
        ax_weight.set_ylabel('Weight')
        ax_weight.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[Saved] Projection & Weight figure (Combined Last 2, 8mm spacing): {save_path}')

def create_projection_figure_combined_top2(agg_results: Dict, datasets: List[Dict], save_path: str):
    matrix_config = [('GEBSW(C,1)', COLOR_C1, 'GEBSW(C,1) (SW-baseline)'), ('GEBSW(C,3)', COLOR_C3, 'GEBSW(C,3) (GSW-baseline)')]
    hspace = 0.15
    fig = plt.figure(figsize=(16, 16))
    fig.patch.set_facecolor('white')
    gs = GridSpec(2, 2, figure=fig, hspace=hspace, wspace=0.15, left=0.06, right=0.94, top=0.92, bottom=0.06)
    dataset = datasets[0]
    for idx, (method, color, label) in enumerate(matrix_config):
        ax_proj = fig.add_subplot(gs[idx, 0])
        key = (dataset['name'], method)
        res = agg_results[key]['last_run']
        source = dataset['source']
        best_idx = np.argmax(res['weights'])
        proj_vals = res['proj_X'][:, best_idx]
        scatter = ax_proj.scatter(source[:, 0], source[:, 1], c=proj_vals, cmap='viridis', s=25, alpha=0.85, edgecolors='none')
        try:
            xi = np.linspace(source[:, 0].min(), source[:, 0].max(), 50)
            yi = np.linspace(source[:, 1].min(), source[:, 1].max(), 50)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = griddata((source[:, 0], source[:, 1]), proj_vals, (Xi, Yi), method='linear', fill_value=0)
            if Zi is not None and (not np.all(Zi == 0)):
                ax_proj.contour(Xi, Yi, Zi, levels=8, colors='white', alpha=0.6, linewidths=0.8)
        except Exception:
            pass
        ax_proj.set_title(f'{label}', fontsize=11, fontweight='bold', color=color)
        ax_proj.set_xlabel('X')
        ax_proj.set_ylabel('Y')
        ax_proj.axis('equal')
        ax_weight = fig.add_subplot(gs[idx, 1])
        weights = res['weights']
        w_array = res['w_array']
        sorted_idx = np.argsort(weights)[::-1]
        sorted_w = weights[sorted_idx]
        sorted_d = w_array[sorted_idx]
        if not np.any(weights != weights[0]):
            ax_weight.bar(range(len(weights)), [1.0 / len(weights)] * len(weights), color=color, alpha=0.6, label='Uniform')
            ax_weight.set_title('Uniform Weights (No Energy Focus)', fontsize=10)
        else:
            ax_weight_twin = ax_weight.twinx()
            bars = ax_weight.bar(range(len(weights)), sorted_w, color=color, alpha=0.7, label='Weight')
            line = ax_weight_twin.plot(range(len(weights)), sorted_d, 'o-', color='darkblue', markersize=3, alpha=0.6, label='W₁ Distance')
            top5_sum = np.sum(sorted_w[:5])
            ax_weight.axvline(x=4.5, color='red', linestyle='--', alpha=0.5)
            ax_weight.text(4.5, ax_weight.get_ylim()[1] * 0.9, 'Top-5', ha='center', color='red', fontweight='bold')
            ax_weight.set_title(f'Energy Weights (Top-5 Sum: {top5_sum:.3f})', fontsize=10)
            ax_weight_twin.set_ylabel('W₁ Distance', color='darkblue')
            ax_weight_twin.tick_params(axis='y', labelcolor='darkblue')
        ax_weight.set_xlabel('Projection Rank (by Weight)')
        ax_weight.set_ylabel('Weight')
        ax_weight.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[Saved] Projection & Weight figure (Combined Top 2, 8mm spacing): {save_path}')

def create_projection_figure_top3(agg_results: Dict, datasets: List[Dict], save_path: str):
    matrix_config = [('GEBSW(C,1)', COLOR_C1, 'GEBSW(C,1)(SW-baseline)'), ('GEBSW(e,1)', COLOR_E1, 'GEBSW(e,1)(EBSW-baseline)'), ('GEBSW(C,3)', COLOR_C3, 'GEBSW(C,3)(GSW-baseline)')]
    fig = plt.figure(figsize=(16, 20))
    fig.patch.set_facecolor('white')
    gs = GridSpec(3, 2, figure=fig, hspace=0.15, wspace=0.15, left=0.06, right=0.94, top=0.92, bottom=0.06)
    fig.suptitle('Fixed-Projection Metric Validation: Projection & Weight Analysis (Top 3)\nDragon Point Cloud (Preserved Shape)', fontsize=16, fontweight='bold', y=0.98)
    dataset = datasets[0]
    for idx, (method, color, label) in enumerate(matrix_config):
        ax_proj = fig.add_subplot(gs[idx, 0])
        key = (dataset['name'], method)
        res = agg_results[key]['last_run']
        source = dataset['source']
        best_idx = np.argmax(res['weights'])
        proj_vals = res['proj_X'][:, best_idx]
        scatter = ax_proj.scatter(source[:, 0], source[:, 1], c=proj_vals, cmap='viridis', s=25, alpha=0.85, edgecolors='none')
        try:
            xi = np.linspace(source[:, 0].min(), source[:, 0].max(), 50)
            yi = np.linspace(source[:, 1].min(), source[:, 1].max(), 50)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = griddata((source[:, 0], source[:, 1]), proj_vals, (Xi, Yi), method='linear', fill_value=0)
            if Zi is not None and (not np.all(Zi == 0)):
                ax_proj.contour(Xi, Yi, Zi, levels=8, colors='white', alpha=0.6, linewidths=0.8)
        except Exception:
            pass
        ax_proj.set_title(f'{label}', fontsize=11, fontweight='bold', color=color)
        ax_proj.set_xlabel('X')
        ax_proj.set_ylabel('Y')
        ax_proj.axis('equal')
        conc_mean = agg_results[key]['concentration_mean']
        conc_std = agg_results[key]['concentration_std']
        w2_mean = agg_results[key]['gebsw_mean']
        w2_std = agg_results[key]['gebsw_std']
        top5_mean = agg_results[key]['top5_mean']
        ax_proj.text(0.02, 0.98, f'Conc: {conc_mean:.3f}±{conc_std:.3f}\nW₂: {w2_mean:.4f}±{w2_std:.4f}\nTop5: {top5_mean:.3f}', transform=ax_proj.transAxes, ha='left', va='top', fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=color, linewidth=2, alpha=0.95))
        ax_weight = fig.add_subplot(gs[idx, 1])
        weights = res['weights']
        w_array = res['w_array']
        sorted_idx = np.argsort(weights)[::-1]
        sorted_w = weights[sorted_idx]
        sorted_d = w_array[sorted_idx]
        if not np.any(weights != weights[0]):
            ax_weight.bar(range(len(weights)), [1.0 / len(weights)] * len(weights), color=color, alpha=0.6, label='Uniform')
            ax_weight.set_title('Uniform Weights (No Energy Focus)', fontsize=10)
        else:
            ax_weight_twin = ax_weight.twinx()
            bars = ax_weight.bar(range(len(weights)), sorted_w, color=color, alpha=0.7, label='Weight')
            line = ax_weight_twin.plot(range(len(weights)), sorted_d, 'o-', color='darkblue', markersize=3, alpha=0.6, label='W₁ Distance')
            top5_sum = np.sum(sorted_w[:5])
            ax_weight.axvline(x=4.5, color='red', linestyle='--', alpha=0.5)
            ax_weight.text(4.5, ax_weight.get_ylim()[1] * 0.9, 'Top-5', ha='center', color='red', fontweight='bold')
            ax_weight.set_title(f'Energy Weights (Top-5 Sum: {top5_sum:.3f})', fontsize=10)
            ax_weight_twin.set_ylabel('W₁ Distance', color='darkblue')
            ax_weight_twin.tick_params(axis='y', labelcolor='darkblue')
        ax_weight.set_xlabel('Projection Rank (by Weight)')
        ax_weight.set_ylabel('Weight')
        ax_weight.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[Saved] Projection & Weight figure (Top 3): {save_path}')

def create_projection_figure_top2(agg_results: Dict, datasets: List[Dict], save_path: str):
    matrix_config = [('GEBSW(C,1)', COLOR_C1, 'GEBSW(C,1)(SW-baseline)'), ('GEBSW(e,1)', COLOR_E1, 'GEBSW(e,1)(EBSW-baseline)')]
    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor('white')
    gs = GridSpec(2, 2, figure=fig, hspace=0.15, wspace=0.15, left=0.06, right=0.94, top=0.92, bottom=0.06)
    fig.suptitle('Fixed-Projection Metric Validation: Projection & Weight Analysis (Top 2)\nDragon Point Cloud (Preserved Shape)', fontsize=16, fontweight='bold', y=0.98)
    dataset = datasets[0]
    for idx, (method, color, label) in enumerate(matrix_config):
        ax_proj = fig.add_subplot(gs[idx, 0])
        key = (dataset['name'], method)
        res = agg_results[key]['last_run']
        source = dataset['source']
        best_idx = np.argmax(res['weights'])
        proj_vals = res['proj_X'][:, best_idx]
        scatter = ax_proj.scatter(source[:, 0], source[:, 1], c=proj_vals, cmap='viridis', s=25, alpha=0.85, edgecolors='none')
        try:
            xi = np.linspace(source[:, 0].min(), source[:, 0].max(), 50)
            yi = np.linspace(source[:, 1].min(), source[:, 1].max(), 50)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = griddata((source[:, 0], source[:, 1]), proj_vals, (Xi, Yi), method='linear', fill_value=0)
            if Zi is not None and (not np.all(Zi == 0)):
                ax_proj.contour(Xi, Yi, Zi, levels=8, colors='white', alpha=0.6, linewidths=0.8)
        except Exception:
            pass
        ax_proj.set_title(f'{label}', fontsize=11, fontweight='bold', color=color)
        ax_proj.set_xlabel('X')
        ax_proj.set_ylabel('Y')
        ax_proj.axis('equal')
        conc_mean = agg_results[key]['concentration_mean']
        conc_std = agg_results[key]['concentration_std']
        w2_mean = agg_results[key]['gebsw_mean']
        w2_std = agg_results[key]['gebsw_std']
        top5_mean = agg_results[key]['top5_mean']
        ax_proj.text(0.02, 0.98, f'Conc: {conc_mean:.3f}±{conc_std:.3f}\nW₂: {w2_mean:.4f}±{w2_std:.4f}\nTop5: {top5_mean:.3f}', transform=ax_proj.transAxes, ha='left', va='top', fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=color, linewidth=2, alpha=0.95))
        ax_weight = fig.add_subplot(gs[idx, 1])
        weights = res['weights']
        w_array = res['w_array']
        sorted_idx = np.argsort(weights)[::-1]
        sorted_w = weights[sorted_idx]
        sorted_d = w_array[sorted_idx]
        if not np.any(weights != weights[0]):
            ax_weight.bar(range(len(weights)), [1.0 / len(weights)] * len(weights), color=color, alpha=0.6, label='Uniform')
            ax_weight.set_title('Uniform Weights (No Energy Focus)', fontsize=10)
        else:
            ax_weight_twin = ax_weight.twinx()
            bars = ax_weight.bar(range(len(weights)), sorted_w, color=color, alpha=0.7, label='Weight')
            line = ax_weight_twin.plot(range(len(weights)), sorted_d, 'o-', color='darkblue', markersize=3, alpha=0.6, label='W₁ Distance')
            top5_sum = np.sum(sorted_w[:5])
            ax_weight.axvline(x=4.5, color='red', linestyle='--', alpha=0.5)
            ax_weight.text(4.5, ax_weight.get_ylim()[1] * 0.9, 'Top-5', ha='center', color='red', fontweight='bold')
            ax_weight.set_title(f'Energy Weights (Top-5 Sum: {top5_sum:.3f})', fontsize=10)
            ax_weight_twin.set_ylabel('W₁ Distance', color='darkblue')
            ax_weight_twin.tick_params(axis='y', labelcolor='darkblue')
        ax_weight.set_xlabel('Projection Rank (by Weight)')
        ax_weight.set_ylabel('Weight')
        ax_weight.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[Saved] Projection & Weight figure (Top 2): {save_path}')

def create_statistics_figure(agg_results: Dict, datasets: List[Dict], save_path: str):
    fig = plt.figure(figsize=(20, 10))
    fig.patch.set_facecolor('white')
    ax_bar = fig.add_subplot(111)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    methods_plot = ['GEBSW(C,1)', 'GEBSW(e,1)', 'GEBSW(C,3)', 'GEBSW(e,3)', 'GEBSW(e,5)']
    colors_plot = [COLOR_C1, COLOR_E1, COLOR_C3, COLOR_E3, COLOR_E5]
    labels_plot = ['GEBSW(C,1)', 'GEBSW(e,1)', 'GEBSW(C,3)', 'GEBSW(e,3)', 'GEBSW(e,5)']
    gebsw_means = []
    gebsw_stds = []
    conc_means = []
    conc_stds = []
    top5_means = []
    top5_stds = []
    dataset = datasets[0]
    for method in methods_plot:
        key = (dataset['name'], method)
        gebsw_means.append(agg_results[key]['gebsw_mean'])
        gebsw_stds.append(agg_results[key]['gebsw_std'])
        conc_means.append(agg_results[key]['concentration_mean'])
        conc_stds.append(agg_results[key]['concentration_std'])
        top5_means.append(agg_results[key]['top5_mean'])
        top5_stds.append(agg_results[key]['top5_std'])
    x = np.arange(len(methods_plot))
    width = 0.25
    bars1 = ax_bar.bar(x - width * 1.5, gebsw_means, width, yerr=gebsw_stds, label='W₂ Distance', color=colors_plot, alpha=0.9, edgecolor='black', capsize=5)
    bars2 = ax_bar.bar(x - width * 0.5, conc_means, width, yerr=conc_stds, label='Concentration', color=colors_plot, alpha=0.6, hatch='//', edgecolor='black', capsize=5)
    bars3 = ax_bar.bar(x + width * 0.5, top5_means, width, yerr=top5_stds, label='Top-5 Weight Ratio', color=colors_plot, alpha=0.4, hatch='\\\\', edgecolor='black', capsize=5)
    for i, (m, s) in enumerate(zip(gebsw_means, gebsw_stds)):
        m_str = f'{m:.3f}'
        s_str = f'{s:.3f}'
        ax_bar.text(i - width * 1.5, m + s + 0.005, f'{m_str}\n±{s_str}', ha='center', va='bottom', fontsize=9, rotation=0, linespacing=1.2)
    ax_bar.set_ylabel('Metric Value', fontsize=14, fontweight='bold')
    ax_bar.set_title('Quantitative Metric Comparison (Mean ± Std, n=10)', fontsize=16, fontweight='bold')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels_plot, fontsize=12)
    ax_bar.legend(fontsize=12, loc='upper left', ncol=3)
    ax_bar.grid(True, alpha=0.3, axis='y')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[Saved] Statistics bar chart: {save_path}')

def create_synergy_figure(agg_results: Dict, datasets: List[Dict], save_path: str):
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('white')
    ax_synergy = fig.add_subplot(111)
    ax_synergy.spines['top'].set_visible(False)
    ax_synergy.spines['right'].set_visible(False)
    ax_synergy.set_title('Synergy Effect Analysis (Mean ± Std)', fontsize=16, fontweight='bold')
    methods_plot = ['GEBSW(C,1)', 'GEBSW(e,1)', 'GEBSW(C,3)', 'GEBSW(e,3)', 'GEBSW(e,5)']
    dataset = datasets[0]
    gebsw_means = []
    gebsw_stds = []
    for method in methods_plot:
        key = (dataset['name'], method)
        gebsw_means.append(agg_results[key]['gebsw_mean'])
        gebsw_stds.append(agg_results[key]['gebsw_std'])
    if len(gebsw_means) >= 4:
        baseline = gebsw_means[0]
        energy_gain = gebsw_means[1] - baseline
        nl_gain = gebsw_means[2] - baseline
        combined_gain = gebsw_means[3] - baseline
        synergy_bonus = combined_gain - (energy_gain + nl_gain)
        energy_err = np.sqrt(gebsw_stds[1] ** 2 + gebsw_stds[0] ** 2)
        nl_err = np.sqrt(gebsw_stds[2] ** 2 + gebsw_stds[0] ** 2)
        combined_err = np.sqrt(gebsw_stds[3] ** 2 + gebsw_stds[0] ** 2)
        synergy_err = np.sqrt(combined_err ** 2 + energy_err ** 2 + nl_err ** 2)
        categories = ['Energy\nAlone', 'Nonlinear\nAlone', 'Sum of\nIndependent', 'Actual\nCombined', 'Synergy\nBonus']
        values = [energy_gain, nl_gain, energy_gain + nl_gain, combined_gain, synergy_bonus]
        errors = [energy_err, nl_err, np.sqrt(energy_err ** 2 + nl_err ** 2), combined_err, synergy_err]
        bar_colors = [COLOR_E1, COLOR_C3, '#666666', COLOR_E3, '#FF0000' if synergy_bonus > 0 else '#00AA00']
        bars = ax_synergy.bar(categories, values, yerr=errors, color=bar_colors, alpha=0.8, edgecolor='black', capsize=8)
        for bar, val, err in zip(bars, values, errors):
            height = bar.get_height()
            val_str = f'{val:+.3f}'
            err_str = f'{err:.3f}'
            ax_synergy.text(bar.get_x() + bar.get_width() / 2.0, height + err + 0.003, f'{val_str}\n±{err_str}', ha='center', va='bottom', fontsize=10, fontweight='bold', rotation=0, linespacing=1.2)
        ax_synergy.axhline(y=0, color='black', linewidth=2)
        ax_synergy.set_ylabel('Gain of GBESW(e,3) (Ours) over GBESW(C,1) (SW-baseline)', fontsize=14, fontweight='bold')
        ax_synergy.grid(True, alpha=0.3, axis='y')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[Saved] Synergy analysis figure: {save_path}')

def statistical_validation(agg_results: Dict, dataset_name: str) -> Dict:
    methods = ['GEBSW(C,1)', 'GEBSW(e,1)', 'GEBSW(C,3)', 'GEBSW(e,3)', 'GEBSW(e,5)']
    print('\n' + '=' * 80)
    print("Statistical Validation: Paired t-test & Cohen's d")
    print('=' * 80)
    method_values = {}
    for method in methods:
        key = (dataset_name, method)
        method_values[method] = [r['gebsw'] for r in agg_results[key]['all_runs']]
    stats_summary = {}
    baseline_vals = np.array(method_values['GEBSW(C,1)'])
    for method in methods[1:]:
        vals = np.array(method_values[method])
        t_stat, p_value = ttest_rel(vals, baseline_vals)
        diff = vals - baseline_vals
        cohens_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-08)
        stats_summary[method] = {'mean_diff': np.mean(diff), 'std_diff': np.std(diff), 't_stat': t_stat, 'p_value': p_value, 'cohens_d': cohens_d, 'significant': p_value < 0.05}
        print(f'\n{method} vs GEBSW(C,1):')
        print(f'  Mean difference: {np.mean(diff):.6f} ± {np.std(diff) / np.sqrt(len(diff)):.6f} (SEM)')
        print(f'  t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}')
        print(f"  Cohen's d: {cohens_d:.4f} ({('Large' if abs(cohens_d) > 0.8 else 'Medium' if abs(cohens_d) > 0.5 else 'Small')} effect)")
        print(f"  Significant: {('Yes' if p_value < 0.05 else 'No')} (α=0.05)")
    return stats_summary

def _method_visual_configs(names=None):
    all_cfg = [('GEBSW(C,1)', COLOR_C1, 'GEBSW(C,1)\nSW: linear + uniform', 1, False), ('GEBSW(C,3)', COLOR_C3, 'GEBSW(C,3)\nGSW: nonlinear + uniform', 3, False), ('GEBSW(e,1)', COLOR_E1, 'GEBSW(e,1)\nEBSW: linear + energy', 1, True), ('GEBSW(e,3)', COLOR_E3, 'GEBSW(e,3)\nOurs: nonlinear + energy', 3, True), ('GEBSW(e,5)', COLOR_E5, 'GEBSW(e,5)\nOurs: high-order nonlinear + energy', 5, True)]
    if names is None:
        return all_cfg
    keep = set(names)
    return [c for c in all_cfg if c[0] in keep]

def _dragon_xy(points: np.ndarray) -> np.ndarray:
    xy = np.asarray(points[:, :2], dtype=np.float32).copy()
    xy = xy - xy.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(xy, axis=1)) + 1e-12
    return xy / scale

def _estimate_density_2d(xy: np.ndarray, k: int=18) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(xy)
        dists, _ = tree.query(xy, k=min(k + 1, len(xy)))
        kth = dists[:, -1]
    except Exception:
        n = len(xy)
        kth = np.empty(n, dtype=np.float32)
        block = 512
        for i in range(0, n, block):
            diff = xy[i:i + block, None, :] - xy[None, :, :]
            dist = np.sqrt(np.sum(diff * diff, axis=2))
            kth[i:i + block] = np.partition(dist, min(k, n - 1), axis=1)[:, min(k, n - 1)]
    rho = 1.0 / (kth + 0.0001)
    rho = np.log1p(rho)
    rho = (rho - rho.min()) / (rho.max() - rho.min() + 1e-12)
    return rho.astype(np.float32)

def _weighted_quantiles(values: np.ndarray, weights: np.ndarray, qs: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    weights = np.asarray(weights) + 1e-12
    order = np.argsort(values)
    sv = values[order]
    sw = weights[order]
    cdf = np.cumsum(sw)
    cdf /= cdf[-1]
    return np.interp(qs, cdf, sv)

def _nonlinear_phase_grid(X, Y, order: int=3):
    base = 1.05 * X + 0.55 * Y + 0.3 * np.sin(4.0 * X + 1.1 * Y) - 0.22 * np.cos(3.4 * Y - 0.8 * X)
    if order >= 5:
        base = base + 0.16 * np.sin(6.2 * X * Y) + 0.1 * (X ** 2 - Y ** 2)
    return base

def _phase_on_points(xy: np.ndarray, order: int, best_theta_xy=None) -> np.ndarray:
    if order <= 1:
        if best_theta_xy is None or np.linalg.norm(best_theta_xy) < 1e-12:
            best_theta_xy = np.array([1.0, 0.04])
        n = best_theta_xy / (np.linalg.norm(best_theta_xy) + 1e-12)
        return xy @ n
    return _nonlinear_phase_grid(xy[:, 0], xy[:, 1], order=order)

def _dragon_limits(xy: np.ndarray, pad: float=0.045):
    xmin, xmax = (xy[:, 0].min(), xy[:, 0].max())
    ymin, ymax = (xy[:, 1].min(), xy[:, 1].max())
    xr, yr = (xmax - xmin, ymax - ymin)
    return ((xmin - pad * xr, xmax + pad * xr), (ymin - pad * yr, ymax + pad * yr))

def _projected_marginal_separation_score(px: np.ndarray, py: np.ndarray) -> tuple:
    px = np.asarray(px).ravel()
    py = np.asarray(py).ravel()
    lo = min(float(px.min()), float(py.min()))
    hi = max(float(px.max()), float(py.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return (0.0, 0.0)
    bins = np.linspace(lo, hi, 74)
    hx, edges = np.histogram(px, bins=bins, density=True)
    hy, _ = np.histogram(py, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    k = np.array([1, 2, 3, 2, 1], dtype=float)
    k /= k.sum()
    hx = np.convolve(hx, k, mode='same')
    hy = np.convolve(hy, k, mode='same')
    dx = float(edges[1] - edges[0])
    overlap = float(np.sum(np.minimum(hx, hy)) * dx)
    overlap = float(np.clip(overlap, 0.0, 1.0))
    sep_main = 1.0 - overlap
    peak_gap = abs(float(centers[np.argmax(hx)]) - float(centers[np.argmax(hy)])) / (abs(hi - lo) + 1e-12)
    peak_gap = float(np.clip(peak_gap, 0.0, 1.0))
    return (sep_main, peak_gap)

def _robust_norm01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    lo = np.nanmin(values)
    hi = np.nanmax(values)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - lo) / (hi - lo + 1e-12)

def _projected_effect_size_score(px: np.ndarray, py: np.ndarray) -> float:
    px = np.asarray(px, dtype=np.float64).ravel()
    py = np.asarray(py, dtype=np.float64).ravel()
    pooled = np.sqrt(0.5 * (np.var(px) + np.var(py))) + 1e-12
    mean_gap = abs(float(np.mean(px) - np.mean(py))) / pooled
    q_gap = abs(float(np.percentile(px, 90) - np.percentile(py, 90))) / pooled
    q_gap += abs(float(np.percentile(px, 10) - np.percentile(py, 10))) / pooled
    return float(mean_gap + 0.35 * q_gap)

def _select_visualization_projection_index(res: Dict, poly_order: int, adaptive: bool) -> int:
    weights = np.asarray(res['weights'], dtype=np.float64)
    w_array = np.asarray(res['w_array'], dtype=np.float64)
    proj_X = np.asarray(res['proj_X'], dtype=np.float64)
    proj_Y = np.asarray(res['proj_Y'], dtype=np.float64)
    n_proj = proj_X.shape[1]
    if n_proj <= 0:
        return 0
    if adaptive:
        if poly_order >= 3:
            k = min(n_proj, max(48, int(0.4 * n_proj)))
        else:
            k = min(n_proj, max(24, int(0.25 * n_proj)))
        candidate_idx = np.argsort(weights)[::-1][:k]
    else:
        k = min(n_proj, max(24, int(0.25 * n_proj)))
        candidate_idx = np.argsort(w_array)[::-1][:k]
    sep_scores, peak_scores, effect_scores = ([], [], [])
    for idx in candidate_idx:
        sep_main, peak_gap = _projected_marginal_separation_score(proj_X[:, idx], proj_Y[:, idx])
        sep_scores.append(sep_main)
        peak_scores.append(peak_gap)
        effect_scores.append(_projected_effect_size_score(proj_X[:, idx], proj_Y[:, idx]))
    sep_n = _robust_norm01(np.asarray(sep_scores))
    peak_n = _robust_norm01(np.asarray(peak_scores))
    eff_n = _robust_norm01(np.asarray(effect_scores))
    w1_n = _robust_norm01(w_array[candidate_idx])
    weight_n = _robust_norm01(weights[candidate_idx])
    if adaptive and poly_order >= 3:
        score = 0.44 * sep_n + 0.24 * eff_n + 0.14 * peak_n + 0.1 * w1_n + 0.08 * weight_n
    elif adaptive:
        score = 0.34 * sep_n + 0.18 * eff_n + 0.1 * peak_n + 0.18 * w1_n + 0.2 * weight_n
    else:
        score = 0.36 * sep_n + 0.24 * eff_n + 0.12 * peak_n + 0.28 * w1_n
    return int(candidate_idx[int(np.argmax(score))])

def _add_clean_3d_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.grid(False)
    try:
        ax.xaxis.pane.set_alpha(0.0)
        ax.yaxis.pane.set_alpha(0.0)
        ax.zaxis.pane.set_alpha(0.0)
        ax.set_box_aspect([1.25, 1.0, 0.65])
    except Exception:
        pass

def _plot_slice_distribution_3d(ax, poly_order: int, adaptive: bool, color: str):
    u = np.linspace(-1, 1, 58)
    v = np.linspace(-1, 1, 58)
    U, V = np.meshgrid(u, v)
    if poly_order <= 1:
        Z = 0.75 * U - 0.38 * V
        levels = np.linspace(Z.min(), Z.max(), 10 if adaptive else 18)
    else:
        Z = 0.55 * U - 0.35 * V + 0.32 * np.sin(2.8 * U) + 0.24 * np.cos(3.1 * V) + 0.16 * U * V
        if poly_order >= 5:
            Z = Z + 0.13 * np.sin(5.0 * U * V) + 0.08 * (U ** 2 - V ** 2)
        levels = np.linspace(np.percentile(Z, 4), np.percentile(Z, 96), 8 if adaptive else 13)
    if adaptive:
        qs = np.linspace(0.1, 0.9, len(levels))
        flat_z = Z.ravel()
        pseudo_density = np.exp(-2.4 * ((U.ravel() + 0.18) ** 2 + (V.ravel() - 0.06) ** 2))
        levels = _weighted_quantiles(flat_z, pseudo_density, qs)
    ax.plot_surface(U, V, Z, cmap='coolwarm', linewidth=0, antialiased=True, alpha=0.86)
    ax.contour(U, V, Z, zdir='z', offset=np.min(Z) - 0.22, levels=levels, colors=color if adaptive else '0.42', linewidths=0.7 if adaptive else 0.45)
    ax.view_init(elev=27, azim=-55)
    ax.set_zlim(np.min(Z) - 0.25, np.max(Z) + 0.15)
    _add_clean_3d_axis(ax)

def _poly2d_design(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    feats = [np.ones_like(x), x, y]
    if order >= 2:
        feats += [x * x, y * y, x * y]
    if order >= 3:
        feats += [x ** 3, y ** 3, x * x * y, x * y * y, np.sin(3.0 * x + 0.7 * y), np.cos(0.8 * x - 3.0 * y)]
    if order >= 5:
        feats += [x ** 4, y ** 4, x ** 3 * y, x * y ** 3, x * x * y * y, np.sin(5.2 * x * y), np.cos(4.5 * (x * x - y * y))]
    return np.column_stack(feats)

def _fit_selected_projection_phase_grid(xy: np.ndarray, proj_vals: np.ndarray, Xg: np.ndarray, Yg: np.ndarray, poly_order: int) -> tuple:
    vals = np.asarray(proj_vals, dtype=np.float64).ravel()
    vals = (vals - np.mean(vals)) / (np.std(vals) + 1e-12)
    order_for_fit = 1 if poly_order <= 1 else 5 if poly_order >= 5 else 3
    A = _poly2d_design(xy[:, 0], xy[:, 1], order_for_fit)
    B = _poly2d_design(Xg.ravel(), Yg.ravel(), order_for_fit)
    lam = 0.002 if order_for_fit <= 1 else 0.012
    ATA = A.T @ A
    rhs = A.T @ vals
    coef = np.linalg.solve(ATA + lam * np.eye(ATA.shape[0]), rhs)
    phase_pts = A @ coef
    phase_grid = (B @ coef).reshape(Xg.shape)
    return (phase_pts, phase_grid)

def _plot_dragon_projection_slices(ax, source: np.ndarray, target: np.ndarray, res: Dict, poly_order: int, adaptive: bool):
    from matplotlib.colors import LinearSegmentedColormap
    xy = _dragon_xy(source)
    target_xy = _dragon_xy(target)
    density = _estimate_density_2d(xy)
    best_idx = _select_visualization_projection_index(res, poly_order, adaptive)
    src_proj = np.asarray(res['proj_X'])[:, best_idx]
    tgt_proj = np.asarray(res['proj_Y'])[:, best_idx]
    lo = min(np.percentile(src_proj, 1), np.percentile(tgt_proj, 1))
    hi = max(np.percentile(src_proj, 99), np.percentile(tgt_proj, 99))
    src_vals = np.clip((src_proj - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    tgt_vals = np.clip((tgt_proj - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    attention_cmap = LinearSegmentedColormap.from_list('blue_yellow_flame_attention', [(0.0, '#08306B'), (0.28, '#2171B5'), (0.52, '#FFD92F'), (0.76, '#FF7F00'), (1.0, '#B30000')], N=256)
    order = np.argsort(src_vals)
    ax.scatter(xy[order, 0], xy[order, 1], c=src_vals[order], cmap=attention_cmap, s=13, alpha=0.92, edgecolors='none', rasterized=True)
    (xmin, xmax), (ymin, ymax) = _dragon_limits(xy)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    xx = np.linspace(xmin, xmax, 340)
    yy = np.linspace(ymin, ymax, 270)
    Xg, Yg = np.meshgrid(xx, yy)
    try:
        phase_pts, phase_grid = _fit_selected_projection_phase_grid(xy, src_proj, Xg, Yg, poly_order=poly_order)
    except Exception:
        best_theta_xy = None
        if poly_order <= 1:
            try:
                coef, *_ = np.linalg.lstsq(xy, res['proj_X'][:, best_idx], rcond=None)
                best_theta_xy = coef[:2]
            except Exception:
                best_theta_xy = np.array([1.0, 0.04])
            phase_pts = _phase_on_points(xy, 1, best_theta_xy)
            n = best_theta_xy / (np.linalg.norm(best_theta_xy) + 1e-12)
            phase_grid = n[0] * Xg + n[1] * Yg
        else:
            phase_pts = _phase_on_points(xy, poly_order)
            phase_grid = _nonlinear_phase_grid(Xg, Yg, order=poly_order)
    if poly_order <= 1:
        n_levels = 14 if adaptive else 28
    else:
        n_levels = 10 if adaptive else 18
    if adaptive:
        qs = np.linspace(0.08, 0.92, n_levels)
        levels = _weighted_quantiles(phase_pts, density ** (2.25 if poly_order <= 1 else 2.65) + 0.06, qs)
        linewidths = np.linspace(1.05, 2.25, n_levels)
        line_color = '#00D9D9'
        for lev, lw in zip(levels, linewidths):
            ax.contour(Xg, Yg, phase_grid, levels=[lev], colors=line_color, linewidths=float(lw), alpha=0.94)
    else:
        levels = np.linspace(np.percentile(phase_pts, 2), np.percentile(phase_pts, 98), n_levels)
        ax.contour(Xg, Yg, phase_grid, levels=levels, colors='white', linewidths=0.62, alpha=0.56)
    inset = ax.inset_axes([0.705, 0.695, 0.275, 0.275])
    tgt_order = np.argsort(tgt_vals)
    inset.scatter(target_xy[tgt_order, 0], target_xy[tgt_order, 1], c=tgt_vals[tgt_order], cmap=attention_cmap, s=7.5, alpha=0.95, edgecolors='none', rasterized=True)
    (bxmin, bxmax), (bymin, bymax) = _dragon_limits(target_xy, pad=0.1)
    inset.set_xlim(bxmin, bxmax)
    inset.set_ylim(bymin, bymax)
    inset.set_aspect('equal')
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_facecolor('#0F0F0F')
    for sp in inset.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(1.0)
        sp.set_edgecolor('white')
        sp.set_alpha(0.85)
    import matplotlib.transforms as mtransforms
    label_transform = inset.transAxes + mtransforms.ScaledTranslation(0.0, 0.3 / 2.54, ax.figure.dpi_scale_trans)
    inset.text(0.5, 0.96, 'Target (bunny)', transform=label_transform, ha='center', va='top', fontsize=10.5, color='white', fontweight='bold', clip_on=False, bbox=dict(boxstyle='round,pad=0.20', facecolor='black', edgecolor='white', alpha=0.55, linewidth=0.8))
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_facecolor('#090909')

def _set_bottom_title(ax, text: str, fontsize: float=11.0, color: str='black', fontweight: str='normal', y: float=-0.16):
    text_kwargs = dict(transform=ax.transAxes, ha='center', va='top', fontsize=fontsize, color=color, fontweight=fontweight, clip_on=False)
    if hasattr(ax, 'text2D'):
        ax.text2D(0.5, y, text, **text_kwargs)
    else:
        ax.text(0.5, y, text, **text_kwargs)

def _plot_energy_weight(ax, res: Dict, color: str, adaptive: bool, show_ylabel: bool=False):
    weights = np.asarray(res['weights'])
    w_array = np.asarray(res['w_array'])
    sorted_idx = np.argsort(weights)[::-1]
    sorted_w = weights[sorted_idx]
    sorted_d = w_array[sorted_idx]
    x = np.arange(len(sorted_w))
    if adaptive:
        ax.bar(x, sorted_w, color=color, alpha=0.74, width=0.86)
        ax2 = ax.twinx()
        d_norm = (sorted_d - sorted_d.min()) / (sorted_d.max() - sorted_d.min() + 1e-12)
        ax2.plot(x, d_norm, 'o-', color='darkblue', markersize=2.2, lw=1.1, alpha=0.7)
        ax2.set_yticks([])
        top5_sum = float(np.sum(sorted_w[:5]))
        ax.axvspan(-0.5, 4.5, color='cyan', alpha=0.08)
        _set_bottom_title(ax, f'Energy Weights (Top-5 Sum: {top5_sum:.3f})', fontsize=15, fontweight='bold', color='black', y=-0.235)
    else:
        ax.bar(x, np.ones_like(sorted_w) / len(sorted_w), color=color, alpha=0.6, width=0.86)
        _set_bottom_title(ax, 'Uniform Weights', fontsize=15, fontweight='bold', color='black', y=-0.235)
    ax.set_xlim(-1, len(sorted_w))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('projection rank', labelpad=8, fontsize=15)
    ax.xaxis.set_label_coords(0.5, -0.11)
    if show_ylabel:
        ax.set_ylabel('Weight', fontsize=15)
    else:
        ax.set_ylabel('')

def _plot_marginal_density(ax, res: Dict, color: str, poly_order: int, adaptive: bool, show_ylabel: bool=False):
    idx = _select_visualization_projection_index(res, poly_order, adaptive)
    px = np.asarray(res['proj_X'])[:, idx]
    py = np.asarray(res['proj_Y'])[:, idx]
    lo = min(float(px.min()), float(py.min()))
    hi = max(float(px.max()), float(py.max()))
    if abs(hi - lo) < 1e-12:
        hi = lo + 1e-06
    bins = np.linspace(lo, hi, 74)
    hx, edges = np.histogram(px, bins=bins, density=True)
    hy, _ = np.histogram(py, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    k = np.array([1, 2, 3, 2, 1], dtype=float)
    k /= k.sum()
    hx = np.convolve(hx, k, mode='same')
    hy = np.convolve(hy, k, mode='same')
    source_color = '#E67300'
    target_color = '#0066CC'
    ax.plot(centers, hx, color=source_color, lw=2.25, linestyle='-', label='Source (dragon)')
    ax.plot(centers, hy, color=target_color, lw=2.25, linestyle='--', alpha=0.95, label='Target (bunny)')
    ax.fill_between(centers, hx, color=source_color, alpha=0.12)
    ax.fill_between(centers, hy, color=target_color, alpha=0.07)
    _set_bottom_title(ax, 'Projected marginal densities', fontsize=15, fontweight='bold', color='black', y=-0.215)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('projected coordinate', labelpad=7, fontsize=15)
    ax.xaxis.set_label_coords(0.5, -0.1)
    if show_ylabel:
        ax.set_ylabel('Marginal density', fontsize=15)
    else:
        ax.set_ylabel('')

def _top5_sum_from_result(res: Dict) -> float:
    weights = np.asarray(res['weights'], dtype=np.float64)
    if weights.size == 0:
        return 0.0
    return float(np.sum(np.sort(weights)[-min(5, weights.size):]))

def _create_mechanism_projection_figure(agg_results: Dict, datasets: List[Dict], save_path: str, method_names=None):
    cfgs = _method_visual_configs(method_names)
    n_cols = len(cfgs)
    fig_w = max(5.2 * n_cols, 10)
    fig = plt.figure(figsize=(fig_w, 13.9))
    fig.patch.set_facecolor('white')
    gs = GridSpec(5, n_cols, figure=fig, hspace=0.0, wspace=0.055, height_ratios=[0.86, 0.145, 1.34, 0.34, 0.92], left=0.022, right=0.994, top=0.968, bottom=0.075)
    dataset = datasets[0]
    source = dataset['source']
    for col, (method, color, label, poly_order, adaptive) in enumerate(cfgs):
        key = (dataset['name'], method)
        res = agg_results[key]['last_run']
        ax0 = fig.add_subplot(gs[0, col], projection='3d')
        _plot_slice_distribution_3d(ax0, poly_order, adaptive, color)
        _set_bottom_title(ax0, label, fontsize=15, fontweight='bold', color='black', y=-0.075)
        ax1 = fig.add_subplot(gs[2, col])
        _plot_dragon_projection_slices(ax1, source, dataset['target'], res, poly_order, adaptive)
        top5_sum = _top5_sum_from_result(res)
        weight_desc = 'Energy focus' if adaptive else 'Uniform weights'
        _set_bottom_title(ax1, f'Dragon projection slices\n{weight_desc}: Top-5 Sum = {top5_sum:.3f}', fontsize=14.5, fontweight='bold', color='black', y=-0.115)
        ax2 = fig.add_subplot(gs[4, col])
        _plot_marginal_density(ax2, res, color, poly_order, adaptive, show_ylabel=col == 0)
        if col == 0:
            ax2.legend(loc='upper right', fontsize=15, frameon=True, facecolor='white', edgecolor='0.35', framealpha=0.92, handlelength=2.6, handletextpad=0.7)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[Saved] Mechanism projection/marginal figure (Top-5 in row 2): {save_path}')

def create_projection_figure(agg_results: Dict, datasets: List[Dict], save_path: str):
    _create_mechanism_projection_figure(agg_results, datasets, save_path, None)

def create_projection_figure_combined_top3(agg_results: Dict, datasets: List[Dict], save_path: str):
    _create_mechanism_projection_figure(agg_results, datasets, save_path, ['GEBSW(C,1)', 'GEBSW(C,3)', 'GEBSW(e,1)'])

def create_projection_figure_combined_last3(agg_results: Dict, datasets: List[Dict], save_path: str):
    _create_mechanism_projection_figure(agg_results, datasets, save_path, ['GEBSW(e,1)', 'GEBSW(e,3)', 'GEBSW(e,5)'])

def create_projection_figure_combined_top2(agg_results: Dict, datasets: List[Dict], save_path: str):
    _create_mechanism_projection_figure(agg_results, datasets, save_path, ['GEBSW(C,1)', 'GEBSW(C,3)'])

def create_projection_figure_combined_last2(agg_results: Dict, datasets: List[Dict], save_path: str):
    _create_mechanism_projection_figure(agg_results, datasets, save_path, ['GEBSW(e,3)', 'GEBSW(e,5)'])

def create_projection_figure_top3(agg_results: Dict, datasets: List[Dict], save_path: str):
    _create_mechanism_projection_figure(agg_results, datasets, save_path, ['GEBSW(C,1)', 'GEBSW(e,1)', 'GEBSW(C,3)'])

def create_projection_figure_top2(agg_results: Dict, datasets: List[Dict], save_path: str):
    _create_mechanism_projection_figure(agg_results, datasets, save_path, ['GEBSW(C,1)', 'GEBSW(e,1)'])

def main():
    print('=' * 80)
    print('GEBSW Fixed-Projection Metric Validation (Split Version)')
    print('拆分版：第三行权重图已删除，Top-5 Sum 已整合到第二行 Dragon projection slices')
    print('=' * 80)
    dragon_points = load_ply_robust(DRAGON_PATH)
    if dragon_points is None:
        print('[Warning] Using fallback dragon data')
        dragon_points = generate_fallback_data('Dragon')
    bunny_points = load_ply_robust(BUNNY_PATH)
    if bunny_points is None:
        print('[Warning] Using fallback bunny data')
        bunny_points = generate_fallback_data('Bunny')
    dragon_points = normalize_point_cloud_unit(dragon_points)
    bunny_points = normalize_point_cloud_unit(bunny_points)
    datasets = [{'name': 'Dragon_vs_Bunny', 'source': dragon_points, 'target': bunny_points}]
    metric_configs = {'GEBSW(C,1)': {'poly_order': 1, 'use_energy_weight': False}, 'GEBSW(e,1)': {'poly_order': 1, 'use_energy_weight': True}, 'GEBSW(C,3)': {'poly_order': 3, 'use_energy_weight': False}, 'GEBSW(e,3)': {'poly_order': 3, 'use_energy_weight': True}, 'GEBSW(e,5)': {'poly_order': 5, 'use_energy_weight': True}}
    agg_results = {}
    for dataset in datasets:
        X = dataset['source']
        Y = dataset['target']
        for method_name, config in metric_configs.items():
            key = (dataset['name'], method_name)
            agg_results[key] = run_multiple_seeds(X, Y, GEBSW_Metric, config, n_runs=10)
    create_projection_figure(agg_results, datasets, os.path.join(RESULT_DIR, 'gebsw_projection_top5_in_row2_3rows.png'))
    create_projection_figure_combined_top3(agg_results, datasets, os.path.join(RESULT_DIR, 'gebsw_projection_weight_combined_top3.png'))
    create_projection_figure_combined_last3(agg_results, datasets, os.path.join(RESULT_DIR, 'gebsw_projection_weight_combined_last3.png'))
    create_projection_figure_combined_top2(agg_results, datasets, os.path.join(RESULT_DIR, 'gebsw_projection_weight_combined_top2.png'))
    create_projection_figure_combined_last2(agg_results, datasets, os.path.join(RESULT_DIR, 'gebsw_projection_weight_combined_last2.png'))
    create_projection_figure_top3(agg_results, datasets, os.path.join(RESULT_DIR, 'gebsw_projection_weight_top3.png'))
    create_projection_figure_top2(agg_results, datasets, os.path.join(RESULT_DIR, 'gebsw_projection_weight_top2.png'))
    create_statistics_figure(agg_results, datasets, os.path.join(RESULT_DIR, 'gebsw_statistics.png'))
    create_synergy_figure(agg_results, datasets, os.path.join(RESULT_DIR, 'gebsw_synergy_analysis.png'))
    stats_summary = statistical_validation(agg_results, 'Dragon_vs_Bunny')
    print('\n' + '=' * 80)
    print('All figures generated successfully!')
    print(f'Results saved to: {RESULT_DIR}')
    print('=' * 80)
if __name__ == '__main__':
    main()
