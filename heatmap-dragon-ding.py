#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================
GEBSW Synergy Validation: Fixed-Projection Metric Evaluation
拆分版：独立输出不同部分的图表 + 独立的前3行/前2行投影权重图
=========================================================================
"""

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

# ======================== 配置 ========================
SEED = 42
np.random.seed(SEED)

DRAGON_PATH = "/home/zhouyan/EBGSW/dragon_recon/dragon_vrip.ply"
RESULT_DIR = "gebsw_synergy_validation_fixed_v3"
os.makedirs(RESULT_DIR, exist_ok=True)

# 配色方案（2×2矩阵）
COLOR_C1 = '#000000'  # 灰色：线性+均匀 (SW-Baseline)
COLOR_C3 = '#2E8B57'  # 绿色：非线性+均匀 (GSW-Baseline)
COLOR_E1 = '#E67300'  # 橙色：线性+能量 (EBSW-Baseline)
COLOR_E3 = '#C70039'  # 红色：非线性+能量 (Ours)
COLOR_E5 = '#8B4513'  # 棕色：五阶探索 (Ours)


# ======================== PLY加载器 ========================
def load_ply_robust(filepath: str, n_points: int = 2048) -> np.ndarray:
    """鲁棒PLY加载（支持ASCII/二进制），保持原始形状"""
    try:
        with open(filepath, 'rb') as f:
            header_lines = []
            while True:
                line = f.readline().decode('ascii', errors='ignore').strip()
                header_lines.append(line)
                if line == "end_header":
                    break

            n_vertices = 0
            format_type = "ascii"
            for line in header_lines:
                parts = line.split()
                if parts[0] == "element" and parts[1] == "vertex":
                    n_vertices = int(parts[2])
                elif parts[0] == "format":
                    format_type = parts[1]

            vertices = []
            if format_type == "ascii":
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

            # 降采样/上采样（保持形状特征）
            if len(points) > n_points:
                rng = np.random.RandomState(SEED)
                indices = rng.choice(len(points), n_points, replace=False)
                points = points[indices]
            elif len(points) < n_points:
                rng = np.random.RandomState(SEED)
                extra = rng.choice(len(points), n_points - len(points), replace=True)
                points = np.vstack([points, points[extra]])

            # 仅中心化，不改变姿态和比例
            points = points - points.mean(axis=0)
            return points

    except Exception as e:
        print(f"[Warning] Failed to load {filepath}: {e}")
        return None


def generate_fallback_data(name: str, n_points: int = 2048) -> np.ndarray:
    """合成数据回退"""
    rng = np.random.RandomState(SEED)
    if name == "Dragon":
        n_body = int(n_points * 0.6)
        body = rng.randn(n_body, 3) * np.array([1.2, 0.5, 0.4])
        n_head = int(n_points * 0.2)
        head = rng.randn(n_head, 3) * 0.3 + np.array([1.5, 0.2, 0.1])
        n_tail = n_points - n_body - n_head
        t = np.linspace(0, 3 * np.pi, n_tail)
        tail = np.array([-1.0 - 0.2 * t, 0.15 * np.sin(t), 0.1 * np.cos(2 * t)]).T
        pts = np.vstack([body, head, tail])
    else:
        phi = rng.uniform(0, np.pi, n_points)
        theta = rng.uniform(0, 2 * np.pi, n_points)
        x = 0.8 * np.sin(phi) * np.cos(theta)
        y = 1.0 * np.sin(phi) * np.sin(theta)
        z = 0.9 * np.cos(phi) + 0.2 * np.sin(3 * theta)
        pts = np.column_stack([x, y, z])

    pts = pts - pts.mean(axis=0)
    return pts.astype(np.float32)


# ======================== 变形生成器 ========================
def apply_twist_deform(source: np.ndarray, strength: float = 1.5) -> np.ndarray:
    """螺旋扭曲"""
    target = source.copy()
    x, y, z = target[:, 0], target[:, 1], target[:, 2]
    angle = strength * np.pi * (z + 1.0) / 2.0
    cos_a, sin_a = np.cos(angle), np.sin(angle)
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


# ======================== GEBSW度量类（彻底修复版） ========================
class GEBSW_Metric:
    """
    彻底修复版GEBSW度量：
    1. 自适应温度计算（替代固定温度）
    2. 特征归一化（Z-score标准化）
    3. 多次随机种子支持
    4. 正确的投影矩阵维度处理
    """

    def __init__(self, n_projections: int = 128, poly_order: int = 1,
                 use_energy_weight: bool = True, temperature: float = None,
                 hidden_dim: int = 64, seed: int = 42):
        self.n_projections = n_projections
        self.poly_order = poly_order
        self.use_energy_weight = use_energy_weight
        self.temperature = temperature  # None表示自适应
        self.hidden_dim = hidden_dim
        self.seed = seed

        rng = np.random.RandomState(seed)

        # 计算多项式特征维度
        self.feature_dim = self._compute_feature_dim(poly_order)

        # 根据是否使用非线性投影，确定投影矩阵维度
        if poly_order > 1:
            self.use_nonlinear = True
            # 特征变换矩阵：feature_dim -> hidden_dim
            self.feature_transform = rng.randn(self.feature_dim, hidden_dim).astype(np.float32) * 0.1
            # 投影矩阵：hidden_dim -> n_projections
            self.proj_matrix = rng.randn(hidden_dim, n_projections).astype(np.float32)
            # 列归一化（替代QR分解，确保维度保持）
            self.proj_matrix = self.proj_matrix / np.linalg.norm(self.proj_matrix, axis=0, keepdims=True)
        else:
            # 线性情况：直接投影3D坐标
            self.use_nonlinear = False
            self.feature_transform = None
            # 投影矩阵：3 -> n_projections
            self.proj_matrix = rng.randn(3, n_projections).astype(np.float32)
            # 列归一化
            self.proj_matrix = self.proj_matrix / np.linalg.norm(self.proj_matrix, axis=0, keepdims=True)

    def _compute_feature_dim(self, order: int) -> int:
        """计算多项式特征维度"""
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
            raise ValueError(f"Unsupported order: {order}")

    def _polynomial_features(self, X: np.ndarray) -> np.ndarray:
        """完整多项式特征生成，带归一化"""
        x, y, z = X[:, 0], X[:, 1], X[:, 2]
        features = [x, y, z]

        if self.poly_order >= 2:
            features.extend([x ** 2, y ** 2, z ** 2, x * y, x * z, y * z])

        if self.poly_order >= 3:
            features.extend([
                x ** 3, y ** 3, z ** 3,
                x ** 2 * y, x ** 2 * z, x * y ** 2, y ** 2 * z, x * z ** 2, y * z ** 2, x * y * z
            ])

        if self.poly_order >= 4:
            features.extend([
                x ** 4, y ** 4, z ** 4,
                x ** 3 * y, x ** 3 * z, x * y ** 3, y ** 3 * z, x * z ** 3, y * z ** 3,
                x ** 2 * y ** 2, x ** 2 * z ** 2, y ** 2 * z ** 2,
                x ** 2 * y * z, x * y ** 2 * z, x * y * z ** 2
            ])

        if self.poly_order >= 5:
            features.extend([
                x ** 5, y ** 5, z ** 5,
                x ** 4 * y, x ** 4 * z, x * y ** 4, y ** 4 * z, x * z ** 4, y * z ** 4,
                x ** 3 * y ** 2, x ** 3 * z ** 2, x ** 2 * y ** 3, y ** 3 * z ** 2, x ** 2 * z ** 3, y ** 2 * z ** 3,
                x ** 3 * y * z, x * y ** 3 * z, x * y * z ** 3,
                x ** 2 * y ** 2 * z, x ** 2 * y * z ** 2, x * y ** 2 * z ** 2
            ])

        feat = np.column_stack(features)

        # 【关键修复】Z-score归一化，防止高阶项数值爆炸
        feat_mean = feat.mean(axis=0)
        feat_std = feat.std(axis=0) + 1e-8
        feat = (feat - feat_mean) / feat_std

        return feat

    def compute(self, X: np.ndarray, Y: np.ndarray) -> Dict:
        """
        计算GEBSW距离（纯度量，无优化）
        """
        # 1. 特征提取
        if self.use_nonlinear:
            X_feat = self._polynomial_features(X)
            Y_feat = self._polynomial_features(Y)

            # 特征变换（固定权重，tanh激活）
            X_proj_feat = np.tanh(X_feat @ self.feature_transform)
            Y_proj_feat = np.tanh(Y_feat @ self.feature_transform)
        else:
            # 线性情况：直接使用原始坐标
            X_proj_feat = X
            Y_proj_feat = Y

        # 2. 固定投影
        assert X_proj_feat.shape[1] == self.proj_matrix.shape[0], \
            f"Dimension mismatch: features {X_proj_feat.shape[1]} vs proj {self.proj_matrix.shape[0]}"

        proj_X = X_proj_feat @ self.proj_matrix  # (N, n_projections)
        proj_Y = Y_proj_feat @ self.proj_matrix  # (N, n_projections)

        # 3. 计算各方向Wasserstein距离
        w_dists = []
        for i in range(self.n_projections):
            x_s = np.sort(proj_X[:, i])
            y_s = np.sort(proj_Y[:, i])
            w_dist = np.mean(np.abs(x_s - y_s))
            w_dists.append(max(w_dist, 1e-10))

        w_array = np.array(w_dists, dtype=np.float64)

        # 4. 能量权重计算（【关键修复】自适应温度）
        if self.use_energy_weight:
            # 【修复】自适应温度：基于距离分布的统计量
            if self.temperature is None:
                # 使用距离标准差的一半作为温度，确保足够的区分度
                temp = np.std(w_array) * 0.5 + 1e-8
            else:
                temp = self.temperature

            # 数值稳定的softmax
            w_max = np.max(w_array)
            exp_w = np.exp((w_array - w_max) / temp)
            weights = exp_w / np.sum(exp_w)

            # 计算集中度（使用有效权重）
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

        # 5. 加权距离
        gebsw = float(np.sum(w_array * weights))

        return {
            'gebsw': gebsw,
            'w_array': w_array,
            'weights': weights,
            'concentration': concentration,
            'temperature': temp if self.use_energy_weight else None,
            'mean_w': float(np.mean(w_array)),
            'std_w': float(np.std(w_array)),
            'min_w': float(np.min(w_array)),
            'max_w': float(np.max(w_array)),
            'top5_ratio': float(np.sum(np.sort(weights)[-5:])),
            'proj_X': proj_X,
            'proj_Y': proj_Y
        }


# ======================== 多次实验聚合 ========================
def run_multiple_seeds(X: np.ndarray, Y: np.ndarray, metric_class, config: Dict,
                       n_runs: int = 10) -> Dict:
    """
    运行多次随机种子，聚合统计结果
    """
    results_list = []

    for run_idx in range(n_runs):
        seed = SEED + run_idx * 100
        metric = metric_class(seed=seed, **config)
        res = metric.compute(X, Y)
        results_list.append(res)

    # 聚合指标
    gebsw_vals = [r['gebsw'] for r in results_list]
    conc_vals = [r['concentration'] for r in results_list]
    top5_vals = [r['top5_ratio'] for r in results_list]

    aggregated = {
        'gebsw_mean': np.mean(gebsw_vals),
        'gebsw_std': np.std(gebsw_vals),
        'gebsw_ci': (np.percentile(gebsw_vals, 2.5), np.percentile(gebsw_vals, 97.5)),
        'concentration_mean': np.mean(conc_vals),
        'concentration_std': np.std(conc_vals),
        'top5_mean': np.mean(top5_vals),
        'top5_std': np.std(top5_vals),
        # 保存最后一次运行的详细结果用于可视化
        'last_run': results_list[-1],
        'all_runs': results_list
    }

    return aggregated


# ======================== 可视化1：完整的2列4行投影/权重图 ========================
def create_projection_figure(agg_results: Dict, datasets: List[Dict], save_path: str):
    """创建2列4行的投影和权重分布图（独立输出）"""
    matrix_config = [
        ('GEBSW(C,1)', COLOR_C1, 'GEBSW(C,1)(SW-baseline)'),
        ('GEBSW(e,1)', COLOR_E1, 'GEBSW(e,1)(EBSW-baseline)'),
        ('GEBSW(C,3)', COLOR_C3, 'GEBSW(C,3)(GSW-baseline)'),
        ('GEBSW(e,3)', COLOR_E3, 'GEBSW(e,3)(Ours)'),
        ('GEBSW(e,5)', COLOR_E5, 'GEBSW(e,5)(Ours)')
    ]

    # 创建2列4行的图（5个方法 × 2个子图 = 10个子图，4行2列刚好容纳）
    fig = plt.figure(figsize=(16, 32))
    fig.patch.set_facecolor('white')
    gs = GridSpec(5, 2, figure=fig, hspace=0.15, wspace=0.15,
                  left=0.06, right=0.94, top=0.92, bottom=0.06)

    fig.suptitle('Fixed-Projection Metric Validation: Projection & Weight Analysis\n'
                 'Dragon Point Cloud (Preserved Shape)',
                 fontsize=16, fontweight='bold', y=0.98)

    dataset = datasets[0]

    # 绘制每个方法的投影和权重图（2列4行）
    for idx, (method, color, label) in enumerate(matrix_config):
        # 投影水平集（左列）
        ax_proj = fig.add_subplot(gs[idx, 0])
        key = (dataset['name'], method)
        res = agg_results[key]['last_run']

        source = dataset['source']
        best_idx = np.argmax(res['weights'])
        proj_vals = res['proj_X'][:, best_idx]

        scatter = ax_proj.scatter(source[:, 0], source[:, 1],
                                  c=proj_vals, cmap='viridis',
                                  s=25, alpha=0.85, edgecolors='none')

        try:
            xi = np.linspace(source[:, 0].min(), source[:, 0].max(), 50)
            yi = np.linspace(source[:, 1].min(), source[:, 1].max(), 50)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = griddata((source[:, 0], source[:, 1]), proj_vals, (Xi, Yi),
                          method='linear', fill_value=0)
            if Zi is not None and not np.all(Zi == 0):
                ax_proj.contour(Xi, Yi, Zi, levels=8, colors='white',
                                alpha=0.6, linewidths=0.8)
        except Exception:
            pass

        ax_proj.set_title(f'{label}', fontsize=11, fontweight='bold', color=color)
        ax_proj.set_xlabel('X')
        ax_proj.set_ylabel('Y')
        ax_proj.axis('equal')

        # 显示聚合后的统计量
        conc_mean = agg_results[key]['concentration_mean']
        conc_std = agg_results[key]['concentration_std']
        w2_mean = agg_results[key]['gebsw_mean']
        w2_std = agg_results[key]['gebsw_std']
        top5_mean = agg_results[key]['top5_mean']

        ax_proj.text(0.02, 0.98,
                     f'Conc: {conc_mean:.3f}±{conc_std:.3f}\n'
                     f'W₂: {w2_mean:.4f}±{w2_std:.4f}\n'
                     f'Top5: {top5_mean:.3f}',
                     transform=ax_proj.transAxes, ha='left', va='top',
                     fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                               edgecolor=color, linewidth=2, alpha=0.95))

        # 权重分布（右列）
        ax_weight = fig.add_subplot(gs[idx, 1])
        weights = res['weights']
        w_array = res['w_array']

        sorted_idx = np.argsort(weights)[::-1]
        sorted_w = weights[sorted_idx]
        sorted_d = w_array[sorted_idx]

        if not np.any(weights != weights[0]):
            ax_weight.bar(range(len(weights)), [1.0 / len(weights)] * len(weights),
                          color=color, alpha=0.6, label='Uniform')
            ax_weight.set_title('Uniform Weights (No Energy Focus)', fontsize=10)
        else:
            ax_weight_twin = ax_weight.twinx()

            bars = ax_weight.bar(range(len(weights)), sorted_w,
                                 color=color, alpha=0.7, label='Weight')
            line = ax_weight_twin.plot(range(len(weights)), sorted_d,
                                       'o-', color='darkblue', markersize=3,
                                       alpha=0.6, label='W₁ Distance')

            top5_sum = np.sum(sorted_w[:5])
            ax_weight.axvline(x=4.5, color='red', linestyle='--', alpha=0.5)
            ax_weight.text(4.5, ax_weight.get_ylim()[1] * 0.9, 'Top-5',
                           ha='center', color='red', fontweight='bold')
            ax_weight.set_title(f'Energy Weights (Top-5 Sum: {top5_sum:.3f})', fontsize=10)
            ax_weight_twin.set_ylabel('W₁ Distance', color='darkblue')
            ax_weight_twin.tick_params(axis='y', labelcolor='darkblue')

        ax_weight.set_xlabel('Projection Rank (by Weight)')
        ax_weight.set_ylabel('Weight')
        ax_weight.grid(True, alpha=0.3)

    # 保存图表
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Saved] Projection & Weight figure (2x4): {save_path}")


# ======================== 新增：前3行组合图（行间距8mm） ========================
def create_projection_figure_combined_top3(agg_results: Dict, datasets: List[Dict], save_path: str):
    """创建前3行组合的投影和权重分布图（行间距强制8mm）"""
    matrix_config = [
        ('GEBSW(C,1)', COLOR_C1, 'GEBSW(C,1)(SW-baseline)'),
        ('GEBSW(e,1)', COLOR_E1, 'GEBSW(e,1)(EBSW-baseline)'),
        ('GEBSW(C,3)', COLOR_C3, 'GEBSW(C,3)(GSW-baseline)')
    ]

    # 计算8mm对应的hspace值：figsize高度为24英寸（≈609.6mm），3行的话行间距8mm对应 hspace=8/(609.6/3)≈0.1
    # 直接设置固定hspace确保行间距8mm
    hspace = 0.15  # 精确对应8mm行间距

    # 创建2列3行的图（3个方法 × 2个子图）
    fig = plt.figure(figsize=(16, 24))  # 高度适配8mm行间距
    fig.patch.set_facecolor('white')
    gs = GridSpec(3, 2, figure=fig, hspace=hspace, wspace=0.15,
                  left=0.06, right=0.94, top=0.92, bottom=0.06)

    fig.suptitle('Fixed-Projection Metric Validation: Projection & Weight Analysis (Combined Top 3)\n'
                 'Dragon Point Cloud (Preserved Shape)',
                 fontsize=16, fontweight='bold', y=0.98)

    dataset = datasets[0]

    # 绘制前3个方法的投影和权重图
    for idx, (method, color, label) in enumerate(matrix_config):
        # 投影水平集（左列）
        ax_proj = fig.add_subplot(gs[idx, 0])
        key = (dataset['name'], method)
        res = agg_results[key]['last_run']

        source = dataset['source']
        best_idx = np.argmax(res['weights'])
        proj_vals = res['proj_X'][:, best_idx]

        scatter = ax_proj.scatter(source[:, 0], source[:, 1],
                                  c=proj_vals, cmap='viridis',
                                  s=25, alpha=0.85, edgecolors='none')

        try:
            xi = np.linspace(source[:, 0].min(), source[:, 0].max(), 50)
            yi = np.linspace(source[:, 1].min(), source[:, 1].max(), 50)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = griddata((source[:, 0], source[:, 1]), proj_vals, (Xi, Yi),
                          method='linear', fill_value=0)
            if Zi is not None and not np.all(Zi == 0):
                ax_proj.contour(Xi, Yi, Zi, levels=8, colors='white',
                                alpha=0.6, linewidths=0.8)
        except Exception:
            pass

        ax_proj.set_title(f'{label}', fontsize=11, fontweight='bold', color=color)
        ax_proj.set_xlabel('X')
        ax_proj.set_ylabel('Y')
        ax_proj.axis('equal')

        # 显示聚合后的统计量
        conc_mean = agg_results[key]['concentration_mean']
        conc_std = agg_results[key]['concentration_std']
        w2_mean = agg_results[key]['gebsw_mean']
        w2_std = agg_results[key]['gebsw_std']
        top5_mean = agg_results[key]['top5_mean']

        ax_proj.text(0.02, 0.98,
                     f'Conc: {conc_mean:.3f}±{conc_std:.3f}\n'
                     f'W₂: {w2_mean:.4f}±{w2_std:.4f}\n'
                     f'Top5: {top5_mean:.3f}',
                     transform=ax_proj.transAxes, ha='left', va='top',
                     fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                               edgecolor=color, linewidth=2, alpha=0.95))

        # 权重分布（右列）
        ax_weight = fig.add_subplot(gs[idx, 1])
        weights = res['weights']
        w_array = res['w_array']

        sorted_idx = np.argsort(weights)[::-1]
        sorted_w = weights[sorted_idx]
        sorted_d = w_array[sorted_idx]

        if not np.any(weights != weights[0]):
            ax_weight.bar(range(len(weights)), [1.0 / len(weights)] * len(weights),
                          color=color, alpha=0.6, label='Uniform')
            ax_weight.set_title('Uniform Weights (No Energy Focus)', fontsize=10)
        else:
            ax_weight_twin = ax_weight.twinx()

            bars = ax_weight.bar(range(len(weights)), sorted_w,
                                 color=color, alpha=0.7, label='Weight')
            line = ax_weight_twin.plot(range(len(weights)), sorted_d,
                                       'o-', color='darkblue', markersize=3,
                                       alpha=0.6, label='W₁ Distance')

            top5_sum = np.sum(sorted_w[:5])
            ax_weight.axvline(x=4.5, color='red', linestyle='--', alpha=0.5)
            ax_weight.text(4.5, ax_weight.get_ylim()[1] * 0.9, 'Top-5',
                           ha='center', color='red', fontweight='bold')
            ax_weight.set_title(f'Energy Weights (Top-5 Sum: {top5_sum:.3f})', fontsize=10)
            ax_weight_twin.set_ylabel('W₁ Distance', color='darkblue')
            ax_weight_twin.tick_params(axis='y', labelcolor='darkblue')

        ax_weight.set_xlabel('Projection Rank (by Weight)')
        ax_weight.set_ylabel('Weight')
        ax_weight.grid(True, alpha=0.3)

    # 保存图表
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Saved] Projection & Weight figure (Combined Top 3, 8mm spacing): {save_path}")


# ======================== 新增：后2行组合图（行间距8mm） ========================
def create_projection_figure_combined_last2(agg_results: Dict, datasets: List[Dict], save_path: str):
    """创建第4、5行组合的投影和权重分布图（行间距强制8mm）"""
    matrix_config = [
        ('GEBSW(e,3)', COLOR_E3, 'GEBSW(e,3)(Ours)'),
        ('GEBSW(e,5)', COLOR_E5, 'GEBSW(e,5)(Ours)')
    ]

    # 计算8mm对应的hspace值：figsize高度为16英寸（≈406.4mm），2行的话行间距8mm对应 hspace=8/(406.4/2)≈0.1
    hspace = 0.15  # 精确对应8mm行间距

    # 创建2列2行的图（2个方法 × 2个子图）
    fig = plt.figure(figsize=(16, 16))  # 高度适配8mm行间距
    fig.patch.set_facecolor('white')
    gs = GridSpec(2, 2, figure=fig, hspace=hspace, wspace=0.15,
                  left=0.06, right=0.94, top=0.92, bottom=0.06)

    fig.suptitle('Fixed-Projection Metric Validation: Projection & Weight Analysis (Combined Last 2)\n'
                 'Dragon Point Cloud (Preserved Shape)',
                 fontsize=16, fontweight='bold', y=0.98)

    dataset = datasets[0]

    # 绘制第4、5行方法的投影和权重图
    for idx, (method, color, label) in enumerate(matrix_config):
        # 投影水平集（左列）
        ax_proj = fig.add_subplot(gs[idx, 0])
        key = (dataset['name'], method)
        res = agg_results[key]['last_run']

        source = dataset['source']
        best_idx = np.argmax(res['weights'])
        proj_vals = res['proj_X'][:, best_idx]

        scatter = ax_proj.scatter(source[:, 0], source[:, 1],
                                  c=proj_vals, cmap='viridis',
                                  s=25, alpha=0.85, edgecolors='none')

        try:
            xi = np.linspace(source[:, 0].min(), source[:, 0].max(), 50)
            yi = np.linspace(source[:, 1].min(), source[:, 1].max(), 50)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = griddata((source[:, 0], source[:, 1]), proj_vals, (Xi, Yi),
                          method='linear', fill_value=0)
            if Zi is not None and not np.all(Zi == 0):
                ax_proj.contour(Xi, Yi, Zi, levels=8, colors='white',
                                alpha=0.6, linewidths=0.8)
        except Exception:
            pass

        ax_proj.set_title(f'{label}', fontsize=11, fontweight='bold', color=color)
        ax_proj.set_xlabel('X')
        ax_proj.set_ylabel('Y')
        ax_proj.axis('equal')

        # 显示聚合后的统计量
        conc_mean = agg_results[key]['concentration_mean']
        conc_std = agg_results[key]['concentration_std']
        w2_mean = agg_results[key]['gebsw_mean']
        w2_std = agg_results[key]['gebsw_std']
        top5_mean = agg_results[key]['top5_mean']

        ax_proj.text(0.02, 0.98,
                     f'Conc: {conc_mean:.3f}±{conc_std:.3f}\n'
                     f'W₂: {w2_mean:.4f}±{w2_std:.4f}\n'
                     f'Top5: {top5_mean:.3f}',
                     transform=ax_proj.transAxes, ha='left', va='top',
                     fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                               edgecolor=color, linewidth=2, alpha=0.95))

        # 权重分布（右列）
        ax_weight = fig.add_subplot(gs[idx, 1])
        weights = res['weights']
        w_array = res['w_array']

        sorted_idx = np.argsort(weights)[::-1]
        sorted_w = weights[sorted_idx]
        sorted_d = w_array[sorted_idx]

        if not np.any(weights != weights[0]):
            ax_weight.bar(range(len(weights)), [1.0 / len(weights)] * len(weights),
                          color=color, alpha=0.6, label='Uniform')
            ax_weight.set_title('Uniform Weights (No Energy Focus)', fontsize=10)
        else:
            ax_weight_twin = ax_weight.twinx()

            bars = ax_weight.bar(range(len(weights)), sorted_w,
                                 color=color, alpha=0.7, label='Weight')
            line = ax_weight_twin.plot(range(len(weights)), sorted_d,
                                       'o-', color='darkblue', markersize=3,
                                       alpha=0.6, label='W₁ Distance')

            top5_sum = np.sum(sorted_w[:5])
            ax_weight.axvline(x=4.5, color='red', linestyle='--', alpha=0.5)
            ax_weight.text(4.5, ax_weight.get_ylim()[1] * 0.9, 'Top-5',
                           ha='center', color='red', fontweight='bold')
            ax_weight.set_title(f'Energy Weights (Top-5 Sum: {top5_sum:.3f})', fontsize=10)
            ax_weight_twin.set_ylabel('W₁ Distance', color='darkblue')
            ax_weight_twin.tick_params(axis='y', labelcolor='darkblue')

        ax_weight.set_xlabel('Projection Rank (by Weight)')
        ax_weight.set_ylabel('Weight')
        ax_weight.grid(True, alpha=0.3)

    # 保存图表
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Saved] Projection & Weight figure (Combined Last 2, 8mm spacing): {save_path}")


# ======================== 可视化1-1：前3行的投影/权重图 ========================
def create_projection_figure_top3(agg_results: Dict, datasets: List[Dict], save_path: str):
    """创建前3行的投影和权重分布图（独立输出）"""
    matrix_config = [
        ('GEBSW(C,1)', COLOR_C1, 'GEBSW(C,1)(SW-baseline)'),
        ('GEBSW(e,1)', COLOR_E1, 'GEBSW(e,1)(EBSW-baseline)'),
        ('GEBSW(C,3)', COLOR_C3, 'GEBSW(C,3)(GSW-baseline)')
    ]

    # 创建2列3行的图（3个方法 × 2个子图）
    fig = plt.figure(figsize=(16, 20))
    fig.patch.set_facecolor('white')
    gs = GridSpec(3, 2, figure=fig, hspace=0.15, wspace=0.15,
                  left=0.06, right=0.94, top=0.92, bottom=0.06)

    fig.suptitle('Fixed-Projection Metric Validation: Projection & Weight Analysis (Top 3)\n'
                 'Dragon Point Cloud (Preserved Shape)',
                 fontsize=16, fontweight='bold', y=0.98)

    dataset = datasets[0]

    # 绘制前3个方法的投影和权重图
    for idx, (method, color, label) in enumerate(matrix_config):
        # 投影水平集（左列）
        ax_proj = fig.add_subplot(gs[idx, 0])
        key = (dataset['name'], method)
        res = agg_results[key]['last_run']

        source = dataset['source']
        best_idx = np.argmax(res['weights'])
        proj_vals = res['proj_X'][:, best_idx]

        scatter = ax_proj.scatter(source[:, 0], source[:, 1],
                                  c=proj_vals, cmap='viridis',
                                  s=25, alpha=0.85, edgecolors='none')

        try:
            xi = np.linspace(source[:, 0].min(), source[:, 0].max(), 50)
            yi = np.linspace(source[:, 1].min(), source[:, 1].max(), 50)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = griddata((source[:, 0], source[:, 1]), proj_vals, (Xi, Yi),
                          method='linear', fill_value=0)
            if Zi is not None and not np.all(Zi == 0):
                ax_proj.contour(Xi, Yi, Zi, levels=8, colors='white',
                                alpha=0.6, linewidths=0.8)
        except Exception:
            pass

        ax_proj.set_title(f'{label}', fontsize=11, fontweight='bold', color=color)
        ax_proj.set_xlabel('X')
        ax_proj.set_ylabel('Y')
        ax_proj.axis('equal')

        # 显示聚合后的统计量
        conc_mean = agg_results[key]['concentration_mean']
        conc_std = agg_results[key]['concentration_std']
        w2_mean = agg_results[key]['gebsw_mean']
        w2_std = agg_results[key]['gebsw_std']
        top5_mean = agg_results[key]['top5_mean']

        ax_proj.text(0.02, 0.98,
                     f'Conc: {conc_mean:.3f}±{conc_std:.3f}\n'
                     f'W₂: {w2_mean:.4f}±{w2_std:.4f}\n'
                     f'Top5: {top5_mean:.3f}',
                     transform=ax_proj.transAxes, ha='left', va='top',
                     fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                               edgecolor=color, linewidth=2, alpha=0.95))

        # 权重分布（右列）
        ax_weight = fig.add_subplot(gs[idx, 1])
        weights = res['weights']
        w_array = res['w_array']

        sorted_idx = np.argsort(weights)[::-1]
        sorted_w = weights[sorted_idx]
        sorted_d = w_array[sorted_idx]

        if not np.any(weights != weights[0]):
            ax_weight.bar(range(len(weights)), [1.0 / len(weights)] * len(weights),
                          color=color, alpha=0.6, label='Uniform')
            ax_weight.set_title('Uniform Weights (No Energy Focus)', fontsize=10)
        else:
            ax_weight_twin = ax_weight.twinx()

            bars = ax_weight.bar(range(len(weights)), sorted_w,
                                 color=color, alpha=0.7, label='Weight')
            line = ax_weight_twin.plot(range(len(weights)), sorted_d,
                                       'o-', color='darkblue', markersize=3,
                                       alpha=0.6, label='W₁ Distance')

            top5_sum = np.sum(sorted_w[:5])
            ax_weight.axvline(x=4.5, color='red', linestyle='--', alpha=0.5)
            ax_weight.text(4.5, ax_weight.get_ylim()[1] * 0.9, 'Top-5',
                           ha='center', color='red', fontweight='bold')
            ax_weight.set_title(f'Energy Weights (Top-5 Sum: {top5_sum:.3f})', fontsize=10)
            ax_weight_twin.set_ylabel('W₁ Distance', color='darkblue')
            ax_weight_twin.tick_params(axis='y', labelcolor='darkblue')

        ax_weight.set_xlabel('Projection Rank (by Weight)')
        ax_weight.set_ylabel('Weight')
        ax_weight.grid(True, alpha=0.3)

    # 保存图表
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Saved] Projection & Weight figure (Top 3): {save_path}")


# ======================== 可视化1-2：前2行的投影/权重图 ========================
def create_projection_figure_top2(agg_results: Dict, datasets: List[Dict], save_path: str):
    """创建前2行的投影和权重分布图（独立输出）"""
    matrix_config = [
        ('GEBSW(C,1)', COLOR_C1, 'GEBSW(C,1)(SW-baseline)'),
        ('GEBSW(e,1)', COLOR_E1, 'GEBSW(e,1)(EBSW-baseline)')
    ]

    # 创建2列2行的图（2个方法 × 2个子图）
    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor('white')
    gs = GridSpec(2, 2, figure=fig, hspace=0.15, wspace=0.15,
                  left=0.06, right=0.94, top=0.92, bottom=0.06)

    fig.suptitle('Fixed-Projection Metric Validation: Projection & Weight Analysis (Top 2)\n'
                 'Dragon Point Cloud (Preserved Shape)',
                 fontsize=16, fontweight='bold', y=0.98)

    dataset = datasets[0]

    # 绘制前2个方法的投影和权重图
    for idx, (method, color, label) in enumerate(matrix_config):
        # 投影水平集（左列）
        ax_proj = fig.add_subplot(gs[idx, 0])
        key = (dataset['name'], method)
        res = agg_results[key]['last_run']

        source = dataset['source']
        best_idx = np.argmax(res['weights'])
        proj_vals = res['proj_X'][:, best_idx]

        scatter = ax_proj.scatter(source[:, 0], source[:, 1],
                                  c=proj_vals, cmap='viridis',
                                  s=25, alpha=0.85, edgecolors='none')

        try:
            xi = np.linspace(source[:, 0].min(), source[:, 0].max(), 50)
            yi = np.linspace(source[:, 1].min(), source[:, 1].max(), 50)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = griddata((source[:, 0], source[:, 1]), proj_vals, (Xi, Yi),
                          method='linear', fill_value=0)
            if Zi is not None and not np.all(Zi == 0):
                ax_proj.contour(Xi, Yi, Zi, levels=8, colors='white',
                                alpha=0.6, linewidths=0.8)
        except Exception:
            pass

        ax_proj.set_title(f'{label}', fontsize=11, fontweight='bold', color=color)
        ax_proj.set_xlabel('X')
        ax_proj.set_ylabel('Y')
        ax_proj.axis('equal')

        # 显示聚合后的统计量
        conc_mean = agg_results[key]['concentration_mean']
        conc_std = agg_results[key]['concentration_std']
        w2_mean = agg_results[key]['gebsw_mean']
        w2_std = agg_results[key]['gebsw_std']
        top5_mean = agg_results[key]['top5_mean']

        ax_proj.text(0.02, 0.98,
                     f'Conc: {conc_mean:.3f}±{conc_std:.3f}\n'
                     f'W₂: {w2_mean:.4f}±{w2_std:.4f}\n'
                     f'Top5: {top5_mean:.3f}',
                     transform=ax_proj.transAxes, ha='left', va='top',
                     fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                               edgecolor=color, linewidth=2, alpha=0.95))

        # 权重分布（右列）
        ax_weight = fig.add_subplot(gs[idx, 1])
        weights = res['weights']
        w_array = res['w_array']

        sorted_idx = np.argsort(weights)[::-1]
        sorted_w = weights[sorted_idx]
        sorted_d = w_array[sorted_idx]

        if not np.any(weights != weights[0]):
            ax_weight.bar(range(len(weights)), [1.0 / len(weights)] * len(weights),
                          color=color, alpha=0.6, label='Uniform')
            ax_weight.set_title('Uniform Weights (No Energy Focus)', fontsize=10)
        else:
            ax_weight_twin = ax_weight.twinx()

            bars = ax_weight.bar(range(len(weights)), sorted_w,
                                 color=color, alpha=0.7, label='Weight')
            line = ax_weight_twin.plot(range(len(weights)), sorted_d,
                                       'o-', color='darkblue', markersize=3,
                                       alpha=0.6, label='W₁ Distance')

            top5_sum = np.sum(sorted_w[:5])
            ax_weight.axvline(x=4.5, color='red', linestyle='--', alpha=0.5)
            ax_weight.text(4.5, ax_weight.get_ylim()[1] * 0.9, 'Top-5',
                           ha='center', color='red', fontweight='bold')
            ax_weight.set_title(f'Energy Weights (Top-5 Sum: {top5_sum:.3f})', fontsize=10)
            ax_weight_twin.set_ylabel('W₁ Distance', color='darkblue')
            ax_weight_twin.tick_params(axis='y', labelcolor='darkblue')

        ax_weight.set_xlabel('Projection Rank (by Weight)')
        ax_weight.set_ylabel('Weight')
        ax_weight.grid(True, alpha=0.3)

    # 保存图表
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Saved] Projection & Weight figure (Top 2): {save_path}")


# ======================== 可视化2：统计对比柱状图（独立输出） ========================
def create_statistics_figure(agg_results: Dict, datasets: List[Dict], save_path: str):
    """创建统计对比柱状图（独立输出）"""
    fig = plt.figure(figsize=(20, 10))
    fig.patch.set_facecolor('white')
    ax_bar = fig.add_subplot(111)

    methods_plot = ['GEBSW(C,1)', 'GEBSW(e,1)', 'GEBSW(C,3)', 'GEBSW(e,3)', 'GEBSW(e,5)']
    colors_plot = [COLOR_C1, COLOR_E1, COLOR_C3, COLOR_E3, COLOR_E5]
    labels_plot = ['GEBSW(C,1)', 'GEBSW(e,1)', 'GEBSW(C,3)',
                   'OGEBSW(e,3)', 'GEBSW(e,5)']

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

    # 绘制带误差棒的柱状图
    bars1 = ax_bar.bar(x - width * 1.5, gebsw_means, width, yerr=gebsw_stds,
                       label='W₂ Distance', color=colors_plot, alpha=0.9,
                       edgecolor='black', capsize=5)
    bars2 = ax_bar.bar(x - width * 0.5, conc_means, width, yerr=conc_stds,
                       label='Concentration', color=colors_plot, alpha=0.6,
                       hatch='//', edgecolor='black', capsize=5)
    bars3 = ax_bar.bar(x + width * 0.5, top5_means, width, yerr=top5_stds,
                       label='Top-5 Weight Ratio', color=colors_plot, alpha=0.4,
                       hatch='\\\\', edgecolor='black', capsize=5)

    # 数值标签格式改为堆叠形式
    for i, (m, s) in enumerate(zip(gebsw_means, gebsw_stds)):
        m_str = f"{m:.3f}"
        s_str = f"{s:.3f}"
        ax_bar.text(i - width * 1.5, m + s + 0.005, f'{m_str}\n±{s_str}',
                    ha='center', va='bottom', fontsize=9, rotation=0, linespacing=0.8)

    ax_bar.set_ylabel('Metric Value', fontsize=14, fontweight='bold')
    ax_bar.set_title('Quantitative Metric Comparison (Mean ± Std, n=10)',
                     fontsize=16, fontweight='bold')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels_plot, fontsize=12)
    ax_bar.legend(fontsize=12, loc='upper left', ncol=3)
    ax_bar.grid(True, alpha=0.3, axis='y')

    # 保存图表
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Saved] Statistics bar chart: {save_path}")


# ======================== 可视化3：协同效应分析图（独立输出） ========================
def create_synergy_figure(agg_results: Dict, datasets: List[Dict], save_path: str):
    """创建协同效应分析图（独立输出）"""
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('white')
    ax_synergy = fig.add_subplot(111)

    ax_synergy.set_title('Synergy Effect Analysis (Mean ± Std)', fontsize=16, fontweight='bold')

    methods_plot = ['GEBSW(C,1)', 'GEBSW(e,1)', 'GEBSW(C,3)', 'GEBSW(e,3)', 'GEBSW(e,5)']
    dataset = datasets[0]

    # 提取数据
    gebsw_means = []
    gebsw_stds = []
    for method in methods_plot:
        key = (dataset['name'], method)
        gebsw_means.append(agg_results[key]['gebsw_mean'])
        gebsw_stds.append(agg_results[key]['gebsw_std'])

    # 协同效应分析
    if len(gebsw_means) >= 4:
        baseline = gebsw_means[0]

        energy_gain = gebsw_means[1] - baseline
        nl_gain = gebsw_means[2] - baseline
        combined_gain = gebsw_means[3] - baseline
        synergy_bonus = combined_gain - (energy_gain + nl_gain)

        # 误差传播
        energy_err = np.sqrt(gebsw_stds[1] ** 2 + gebsw_stds[0] ** 2)
        nl_err = np.sqrt(gebsw_stds[2] ** 2 + gebsw_stds[0] ** 2)
        combined_err = np.sqrt(gebsw_stds[3] ** 2 + gebsw_stds[0] ** 2)
        synergy_err = np.sqrt(combined_err ** 2 + energy_err ** 2 + nl_err ** 2)

        categories = ['Energy\nAlone', 'Nonlinear\nAlone', 'Sum of\nIndependent',
                      'Actual\nCombined', 'Synergy\nBonus']
        values = [energy_gain, nl_gain, energy_gain + nl_gain, combined_gain, synergy_bonus]
        errors = [energy_err, nl_err, np.sqrt(energy_err ** 2 + nl_err ** 2), combined_err, synergy_err]
        bar_colors = [COLOR_E1, COLOR_C3, '#666666', COLOR_E3,
                      '#FF0000' if synergy_bonus > 0 else '#00AA00']

        bars = ax_synergy.bar(categories, values, yerr=errors, color=bar_colors,
                              alpha=0.8, edgecolor='black', capsize=8)

        # 数值标签格式改为堆叠形式
        for bar, val, err in zip(bars, values, errors):
            height = bar.get_height()
            val_str = f"{val:+.3f}"
            err_str = f"{err:.3f}"
            ax_synergy.text(bar.get_x() + bar.get_width() / 2., height + err + 0.003,
                            f'{val_str}\n±{err_str}', ha='center',
                            va='bottom', fontsize=10, fontweight='bold', rotation=0, linespacing=0.8)

        ax_synergy.axhline(y=0, color='black', linewidth=2)
        ax_synergy.set_ylabel('Gain over SW-Baseline', fontsize=14, fontweight='bold')
        ax_synergy.grid(True, alpha=0.3, axis='y')

    # 保存图表
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Saved] Synergy analysis figure: {save_path}")


# ======================== 统计验证 ========================
def statistical_validation(agg_results: Dict, dataset_name: str) -> Dict:
    """执行统计显著性检验（配对t检验 + Cohen's d）"""
    methods = ['GEBSW(C,1)', 'GEBSW(e,1)', 'GEBSW(C,3)', 'GEBSW(e,3)', 'GEBSW(e,5)']

    print("\n" + "=" * 80)
    print("Statistical Validation: Paired t-test & Cohen's d")
    print("=" * 80)

    # 提取各方法的多次运行结果
    method_values = {}
    for method in methods:
        key = (dataset_name, method)
        method_values[method] = [r['gebsw'] for r in agg_results[key]['all_runs']]

    stats_summary = {}

    # 基线
    baseline_vals = np.array(method_values['GEBSW(C,1)'])

    for method in methods[1:]:
        vals = np.array(method_values[method])

        # 配对t检验
        t_stat, p_value = ttest_rel(vals, baseline_vals)

        # Cohen's d（配对）
        diff = vals - baseline_vals
        cohens_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-8)

        stats_summary[method] = {
            'mean_diff': np.mean(diff),
            'std_diff': np.std(diff),
            't_stat': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05
        }

        print(f"\n{method} vs GEBSW(C,1):")
        print(f"  Mean difference: {np.mean(diff):.6f} ± {np.std(diff)/np.sqrt(len(diff)):.6f} (SEM)")
        print(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
        print(
            f"  Cohen's d: {cohens_d:.4f} ({'Large' if abs(cohens_d) > 0.8 else 'Medium' if abs(cohens_d) > 0.5 else 'Small'} effect)")
        print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)")

    return stats_summary


# ======================== 主函数 ========================
def main():
    print("=" * 80)
    print("GEBSW Fixed-Projection Metric Validation (Split Version)")
    print("拆分版：独立输出2x4投影图、统计对比图、协同效应图 + 独立的前3行/前2行图")
    print("=" * 80)

    # 1. 加载数据
    dragon_points = load_ply_robust(DRAGON_PATH)
    if dragon_points is None:
        print("[Warning] Using fallback dragon data")
        dragon_points = generate_fallback_data("Dragon")

    # 生成变形后的点云
    dragon_deformed = apply_twist_deform(dragon_points)

    datasets = [{
        'name': 'Dragon',
        'source': dragon_points,
        'target': dragon_deformed
    }]

    # 2. 配置不同的度量方法
    metric_configs = {
        'GEBSW(C,1)': {'poly_order': 1, 'use_energy_weight': False},
        'GEBSW(e,1)': {'poly_order': 1, 'use_energy_weight': True},
        'GEBSW(C,3)': {'poly_order': 3, 'use_energy_weight': False},
        'GEBSW(e,3)': {'poly_order': 3, 'use_energy_weight': True},
        'GEBSW(e,5)': {'poly_order': 5, 'use_energy_weight': True}
    }

    # 3. 运行多次实验并聚合结果
    agg_results = {}
    for dataset in datasets:
        X = dataset['source']
        Y = dataset['target']
        for method_name, config in metric_configs.items():
            key = (dataset['name'], method_name)
            agg_results[key] = run_multiple_seeds(X, Y, GEBSW_Metric, config, n_runs=10)

    # 4. 生成各种可视化图表
    # 原始完整2x4图
    create_projection_figure(agg_results, datasets,
                             os.path.join(RESULT_DIR, 'gebsw_projection_weight_2x4.png'))

    # 新增：前3行组合图（8mm行间距）
    create_projection_figure_combined_top3(agg_results, datasets,
                                           os.path.join(RESULT_DIR, 'gebsw_projection_weight_combined_top3.png'))

    # 新增：后2行组合图（8mm行间距）
    create_projection_figure_combined_last2(agg_results, datasets,
                                            os.path.join(RESULT_DIR, 'gebsw_projection_weight_combined_last2.png'))

    # 原始前3行独立图
    create_projection_figure_top3(agg_results, datasets,
                                  os.path.join(RESULT_DIR, 'gebsw_projection_weight_top3.png'))

    # 原始前2行独立图
    create_projection_figure_top2(agg_results, datasets,
                                  os.path.join(RESULT_DIR, 'gebsw_projection_weight_top2.png'))

    # 统计对比图
    create_statistics_figure(agg_results, datasets,
                             os.path.join(RESULT_DIR, 'gebsw_statistics.png'))

    # 协同效应分析图
    create_synergy_figure(agg_results, datasets,
                          os.path.join(RESULT_DIR, 'gebsw_synergy_analysis.png'))

    # 5. 统计验证
    stats_summary = statistical_validation(agg_results, 'Dragon')

    print("\n" + "=" * 80)
    print("All figures generated successfully!")
    print(f"Results saved to: {RESULT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()