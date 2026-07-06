import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings
import hashlib
import trimesh
import multiprocessing as mp
import h5py
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')
if torch.cuda.is_available():
    print(f'GPU型号: {torch.cuda.get_device_name(0)}')
    print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB')
ENERGY_ORDER_SYMBOL = 'r'
NUM_STEPS = 300
TARGET_SAMPLE_SIZE = 6144
LR = 0.005
MOMENTUM = 0.9
REPEAT_TIMES = 10
L_BASE = 50
P = 2
EPS = 1e-12
UNIFIED_MAX_DIM = 256
INITIAL_PERTURBATION = 0.01
PRECISION_DIGITS = 10
REPORT_DIGITS = 4
INCLUDE_EXTERNAL_BASELINES = True
MAXSW_USE_INNER_OPT = False
MAXSW_INNER_STEPS = 5
MAXSW_INNER_LR = 0.05
DSW_NUM_PROJECTIONS = L_BASE
DSW_INNER_STEPS = 3
DSW_INNER_LR = 0.05
DSW_DIVERSITY_WEIGHT = 0.05
DSW_WEIGHT_TEMPERATURE = 0.25
TEMPERATURE_START = 2.0
TEMPERATURE_END = 0.1
USE_TEMPERATURE_ANNEALING = True
USE_SOFTMAX_WEIGHT = True
WEIGHT_CLIP_MAX = 0.3
RUN_SENSITIVITY_ANALYSIS = False
TEMPERATURE_RANGE = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
SENSITIVITY_REPEAT_TIMES = 3
BATCH_MODE = True
TEST_INDEX = 0
USE_MULTI_GPU = True
MULTI_GPU_IDS = [0, 1]
MULTI_GPU_MIN_TASKS = 2
VARIANT_COLORS = ['#E6194B', '#3CB44B', '#FFE119', '#4363D8', '#F58231', '#911EB4', '#46F0F0', '#F032E6', '#BCF60C', '#FABEBE', '#008080', '#E6BEFF', '#9A6324', '#FF0033', '#800000', '#AAFFC3', '#808000', '#FFD8B1']
SOURCE_TARGET_CMAP = 'viridis'
SHAPENET_HDF5_ROOT = os.environ.get('SHAPENET_HDF5_ROOT', 'SHAPENET_HDF5_ROOT')
NUM_PAIRS = 15
PAIR_SEED = 42
PAIR_CROSS_CATEGORY = True
PAIR_BALANCED_BY_CATEGORY = True
PAIR_METADATA_FILENAME = 'ShapeNet_balanced_pair_metadata.csv'
CURRENT_SOURCE_PC = None
CURRENT_TARGET_PC = None

def load_shapenet_hdf5_all(h5_root):
    all_data, all_labels, loaded_files = ([], [], [])
    if not os.path.isdir(h5_root):
        raise FileNotFoundError(f'ShapeNet HDF5 根目录不存在: {h5_root}')
    for fname in sorted(os.listdir(h5_root)):
        if not fname.endswith('.h5'):
            continue
        h5_path = os.path.join(h5_root, fname)
        try:
            with h5py.File(h5_path, 'r') as f:
                all_data.append(f['data'][:])
                all_labels.append(f['label'][:])
                loaded_files.append(fname)
        except Exception as e:
            print(f'  [Warning] 跳过 {fname}: {e}')
    if not all_data:
        raise RuntimeError(f'未在 {h5_root} 中读到任何 .h5 ShapeNet 文件')
    pcs = np.concatenate(all_data, axis=0).astype(np.float32)
    lbls = np.concatenate(all_labels, axis=0).reshape(-1).astype(int)
    print(f'[ShapeNet] {len(pcs)} samples, {len(np.unique(lbls))} cats from {len(loaded_files)} files')
    return (pcs, lbls)

def generate_shapenet_pairs(all_pcs, all_labels, num_pairs=NUM_PAIRS, seed=PAIR_SEED, cross_category=PAIR_CROSS_CATEGORY, balanced_by_category=PAIR_BALANCED_BY_CATEGORY):
    labels = np.asarray(all_labels).reshape(-1).astype(int)
    n = len(all_pcs)
    if n != len(labels):
        raise ValueError(f'点云数量({n})与标签数量({len(labels)})不一致')
    if num_pairs <= 0:
        return []
    rng = np.random.default_rng(seed)
    unique_labels = np.array(sorted(np.unique(labels).tolist()), dtype=int)
    label_to_indices = {int(lbl): np.where(labels == lbl)[0] for lbl in unique_labels}
    if not balanced_by_category:
        pairs = []
        for _ in range(num_pairs):
            if cross_category and len(unique_labels) >= 2:
                src_lbl, tgt_lbl = rng.choice(unique_labels, 2, replace=False)
                src_idx = int(rng.choice(label_to_indices[int(src_lbl)]))
                tgt_idx = int(rng.choice(label_to_indices[int(tgt_lbl)]))
            else:
                lbl = int(rng.choice(unique_labels))
                indices = label_to_indices[lbl]
                src_idx, tgt_idx = rng.choice(indices, 2, replace=len(indices) < 2)
                src_idx, tgt_idx = (int(src_idx), int(tgt_idx))
            pairs.append((src_idx, tgt_idx, int(labels[src_idx]), int(labels[tgt_idx])))
        return pairs
    if cross_category and len(unique_labels) < 2:
        raise ValueError('cross_category=True 至少需要2个类别')
    shuffled_labels = unique_labels.copy()
    rng.shuffle(shuffled_labels)
    src_label_schedule = []
    while len(src_label_schedule) < num_pairs:
        cycle = shuffled_labels.copy()
        rng.shuffle(cycle)
        src_label_schedule.extend([int(x) for x in cycle])
    src_label_schedule = src_label_schedule[:num_pairs]
    if cross_category:
        tgt_label_schedule = src_label_schedule[1:] + src_label_schedule[:1]
        for i, src_lbl in enumerate(src_label_schedule):
            if tgt_label_schedule[i] == src_lbl:
                alternatives = [int(x) for x in unique_labels if int(x) != src_lbl]
                counts = {lbl: tgt_label_schedule[:i].count(lbl) for lbl in alternatives}
                min_count = min(counts.values())
                best = [lbl for lbl, c in counts.items() if c == min_count]
                tgt_label_schedule[i] = int(rng.choice(best))
    else:
        tgt_label_schedule = list(src_label_schedule)
    used_indices = set()

    def choose_index(label, forbidden=None):
        forbidden = set() if forbidden is None else set(forbidden)
        candidates = [int(idx) for idx in label_to_indices[int(label)] if int(idx) not in used_indices and int(idx) not in forbidden]
        if not candidates:
            candidates = [int(idx) for idx in label_to_indices[int(label)] if int(idx) not in forbidden]
        if not candidates:
            candidates = [int(idx) for idx in label_to_indices[int(label)]]
        chosen = int(rng.choice(candidates))
        used_indices.add(chosen)
        return chosen
    pairs = []
    for src_lbl, tgt_lbl in zip(src_label_schedule, tgt_label_schedule):
        src_idx = choose_index(src_lbl)
        tgt_idx = choose_index(tgt_lbl, forbidden={src_idx})
        pairs.append((src_idx, tgt_idx, int(labels[src_idx]), int(labels[tgt_idx])))
    return pairs

def print_pair_balance_report(pair_list):
    if not pair_list:
        print('[Pair Balance] 空配对列表')
        return
    src_labels = [p[2] for p in pair_list]
    tgt_labels = [p[3] for p in pair_list]
    src_counts = pd.Series(src_labels).value_counts().sort_index().to_dict()
    tgt_counts = pd.Series(tgt_labels).value_counts().sort_index().to_dict()
    pair_counts = pd.Series([f'{s}->{t}' for _, _, s, t in pair_list]).value_counts().to_dict()
    print('\n[Pair Balance] Source 类别计数:', src_counts)
    print('[Pair Balance] Target 类别计数:', tgt_counts)
    print('[Pair Balance] Pair 类型计数:', pair_counts)

def save_pair_metadata(pair_list, save_path):
    rows = []
    for pair_id, (src_idx, tgt_idx, src_lbl, tgt_lbl) in enumerate(pair_list):
        rows.append({'pair_id': pair_id, 'src_idx': int(src_idx), 'tgt_idx': int(tgt_idx), 'src_label': int(src_lbl), 'tgt_label': int(tgt_lbl), 'cross_category': bool(src_lbl != tgt_lbl), 'pair_type': f'c{int(src_lbl)}_to_c{int(tgt_lbl)}', 'pair_seed': PAIR_SEED, 'balanced_by_category': PAIR_BALANCED_BY_CATEGORY})
    pd.DataFrame(rows).to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f'[Pair Metadata] 已保存固定配对清单: {save_path}')
source_path = ''
target_path = ''
RECORD_STEPS = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]
VISUAL_STEPS = [20, 40, 100, 200, 300]
result_dir = 'PCR-results'
CURRENT_PAIR_INDEX = None
CURRENT_PAIR_GROUP = None
CURRENT_SOURCE_NAME = None
CURRENT_TARGET_NAME = None

def get_pair_name_tag():
    if CURRENT_SOURCE_NAME and CURRENT_TARGET_NAME:
        return f'{CURRENT_SOURCE_NAME}-{CURRENT_TARGET_NAME}'
    return 'ShapeNet_pair_unknown'

def pair_named_filename(filename):
    return f'{get_pair_name_tag()}_{filename}'

def pair_named_path(directory, filename):
    return os.path.join(directory, pair_named_filename(filename))
F_SCORE_THRESHOLD = 0.01

def _stable_int_seed(*items, modulo=2 ** 32 - 1):
    text = '::'.join((str(x) for x in items))
    h = hashlib.md5(text.encode('utf-8')).hexdigest()
    return int(h[:8], 16) % modulo

def read_off_file_robust(file_path, target_size=TARGET_SAMPLE_SIZE, seed=None):
    if seed is None:
        seed = _stable_int_seed('read_off_file_robust', os.path.abspath(file_path), target_size)
    np_state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        mesh = trimesh.load(file_path, force='mesh')
        if isinstance(mesh, trimesh.PointCloud):
            points = np.asarray(mesh.vertices, dtype=np.float32)
            if len(points) > target_size:
                indices = np.random.choice(len(points), target_size, replace=False)
                points = points[indices]
            elif len(points) < target_size:
                indices = np.random.choice(len(points), target_size, replace=True)
                points = points[indices]
                points = points + np.random.normal(0, 0.001, points.shape)
        else:
            points = mesh.sample(target_size)
        return np.array(points, dtype=np.float32)
    except Exception as e:
        raise ValueError(f'读取文件失败 {file_path}: {str(e)}')
    finally:
        np.random.set_state(np_state)

def normalize_point_cloud(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    scale = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    if scale > 0:
        pc = pc / scale
    return pc

def polynomial_projection_controlled(X, degree=1, max_dim=UNIFIED_MAX_DIM, seed=None):
    batch_size, dim = X.shape
    if degree == 1:
        return X

    def _randn(shape, device, local_seed=None):
        if local_seed is None:
            return torch.randn(shape, device=device)
        generator = torch.Generator(device=device)
        generator.manual_seed(int(local_seed))
        return torch.randn(shape, device=device, generator=generator)
    projections = [X]
    current_dim = dim
    for d in range(2, degree + 1):
        new_terms = dim
        if dim > 1:
            new_terms += dim * (dim - 1) // 2
        if current_dim + new_terms > max_dim and dim > 3:
            local_seed = None if seed is None else int(seed) + d * 1009 + dim * 9173
            proj_mat = _randn((dim, max_dim // degree), X.device, local_seed)
            proj_mat = F.normalize(proj_mat, p=2, dim=0)
            X_reduced = torch.matmul(X, proj_mat)
            return polynomial_projection_controlled(X_reduced, degree, max_dim, seed=local_seed)
        projections.append(X ** d)
        if dim > 1:
            for i in range(min(dim, 5)):
                for j in range(i + 1, min(dim, 5)):
                    cross_term = X[:, i] ** (d // 2) * X[:, j] ** (d - d // 2)
                    projections.append(cross_term.unsqueeze(1))
                    current_dim += 1
        current_dim += dim
    result = torch.cat(projections, dim=1)
    if result.shape[1] > max_dim:
        local_seed = None if seed is None else int(seed) + result.shape[1] * 37 + degree * 10007
        proj_mat = _randn((result.shape[1], max_dim), X.device, local_seed)
        proj_mat = F.normalize(proj_mat, p=2, dim=0)
        result = torch.matmul(result, proj_mat)
    return result

def rand_projections(dim, num_projections=L_BASE, seed=None):
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        projections = torch.randn((num_projections, dim), device=device, generator=generator)
    else:
        projections = torch.randn((num_projections, dim), device=device)
    return F.normalize(projections, p=2, dim=1)

def one_dimensional_wasserstein(X_proj, Y_proj, p=P):
    X_sorted = torch.sort(X_proj, dim=0)[0]
    Y_sorted = torch.sort(Y_proj, dim=0)[0]
    diff = torch.abs(X_sorted - Y_sorted)
    return torch.pow(torch.mean(torch.pow(diff, p)), 1 / p)

def _stable_seed(base_seed, name, offset=0):
    h = hashlib.md5(f'{int(base_seed)}::{name}::{int(offset)}'.encode('utf-8')).hexdigest()
    return int(h[:8], 16)

def max_sliced_wasserstein_loss(X, Y, num_projections=L_BASE, p=P, seed=None, use_inner_opt=MAXSW_USE_INNER_OPT):
    theta = rand_projections(X.shape[1], num_projections=num_projections, seed=seed)
    X_proj = torch.matmul(X, theta.T)
    Y_proj = torch.matmul(Y, theta.T)
    wd_tensor = torch.stack([one_dimensional_wasserstein(X_proj[:, i], Y_proj[:, i], p=p) for i in range(num_projections)])
    max_idx = torch.argmax(wd_tensor)
    if not use_inner_opt:
        return wd_tensor[max_idx]
    with torch.enable_grad():
        theta_best = theta[max_idx].detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([theta_best], lr=MAXSW_INNER_LR)
        X_det, Y_det = (X.detach(), Y.detach())
        for _ in range(MAXSW_INNER_STEPS):
            opt.zero_grad()
            theta_norm = F.normalize(theta_best, p=2, dim=0)
            x1 = torch.matmul(X_det, theta_norm)
            y1 = torch.matmul(Y_det, theta_norm)
            wd = one_dimensional_wasserstein(x1, y1, p=p)
            (-wd).backward()
            opt.step()
        theta_final = F.normalize(theta_best.detach(), p=2, dim=0)
    return one_dimensional_wasserstein(torch.matmul(X, theta_final), torch.matmul(Y, theta_final), p=p)

def _direction_diversity_penalty(theta):
    theta = F.normalize(theta, p=2, dim=1)
    gram = torch.matmul(theta, theta.T).abs()
    eye = torch.eye(theta.shape[0], device=theta.device, dtype=theta.dtype)
    off_diag = gram * (1.0 - eye)
    return off_diag.mean()

def distributional_sliced_wasserstein_loss(X, Y, num_projections=DSW_NUM_PROJECTIONS, p=P, seed=None):
    init_seed = _stable_seed(seed if seed is not None else 0, 'dsw_theta')
    theta = rand_projections(X.shape[1], num_projections=num_projections, seed=init_seed).detach().clone()
    theta.requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=DSW_INNER_LR)
    X_det, Y_det = (X.detach(), Y.detach())
    for _ in range(DSW_INNER_STEPS):
        opt.zero_grad()
        theta_norm = F.normalize(theta, p=2, dim=1)
        X_proj = torch.matmul(X_det, theta_norm.T)
        Y_proj = torch.matmul(Y_det, theta_norm.T)
        wd_tensor = torch.stack([one_dimensional_wasserstein(X_proj[:, i], Y_proj[:, i], p=p) for i in range(num_projections)])
        weights = F.softmax(wd_tensor / DSW_WEIGHT_TEMPERATURE, dim=0)
        objective = torch.sum(weights * wd_tensor) - DSW_DIVERSITY_WEIGHT * _direction_diversity_penalty(theta_norm)
        (-objective).backward()
        opt.step()
    theta_final = F.normalize(theta.detach(), p=2, dim=1)
    X_proj = torch.matmul(X, theta_final.T)
    Y_proj = torch.matmul(Y, theta_final.T)
    wd_tensor = torch.stack([one_dimensional_wasserstein(X_proj[:, i], Y_proj[:, i], p=p) for i in range(num_projections)])
    weights = F.softmax(wd_tensor.detach() / DSW_WEIGHT_TEMPERATURE, dim=0)
    return torch.sum(weights * wd_tensor)

def get_external_baseline_functions(proj_seed, repeat_idx, current_step=0, total_steps=NUM_STEPS):
    distance_functions = {}

    def sampled_maxsw_baseline(X, Y):
        return max_sliced_wasserstein_loss(X, Y, num_projections=L_BASE, p=P, seed=_stable_seed(proj_seed, 'Sampled-MaxSW'), use_inner_opt=MAXSW_USE_INNER_OPT)
    distance_functions['Sampled-MaxSW (External-Baseline)'] = sampled_maxsw_baseline

    def dsw_baseline(X, Y):
        return distributional_sliced_wasserstein_loss(X, Y, num_projections=DSW_NUM_PROJECTIONS, p=P, seed=_stable_seed(proj_seed, 'DSW'))
    distance_functions['DSW (External-Baseline)'] = dsw_baseline
    return distance_functions

def get_all_distance_functions(proj_seed, repeat_idx, current_step=0, total_steps=NUM_STEPS, temp_start=TEMPERATURE_START, temp_end=TEMPERATURE_END):
    distance_functions = get_distance_functions_corrected(proj_seed, repeat_idx, current_step, total_steps, temp_start, temp_end)
    if INCLUDE_EXTERNAL_BASELINES:
        distance_functions.update(get_external_baseline_functions(proj_seed, repeat_idx, current_step, total_steps))
    return distance_functions

def chamfer_distance(pc1, pc2):
    if isinstance(pc1, np.ndarray):
        pc1 = torch.tensor(pc1, device=device, dtype=torch.float32)
    if isinstance(pc2, np.ndarray):
        pc2 = torch.tensor(pc2, device=device, dtype=torch.float32)
    if pc1.requires_grad or pc2.requires_grad:
        dist_matrix = torch.cdist(pc1, pc2, p=2)
    else:
        with torch.no_grad():
            dist_matrix = torch.cdist(pc1, pc2, p=2)
    dist1 = torch.min(dist_matrix, dim=1)[0].mean()
    dist2 = torch.min(dist_matrix, dim=0)[0].mean()
    return ((dist1 + dist2) / 2).item()

def compute_f_score(pc1, pc2, threshold=F_SCORE_THRESHOLD):
    if isinstance(pc1, np.ndarray):
        pc1 = torch.tensor(pc1, device=device, dtype=torch.float32)
    if isinstance(pc2, np.ndarray):
        pc2 = torch.tensor(pc2, device=device, dtype=torch.float32)
    with torch.no_grad():
        dist_matrix = torch.cdist(pc1, pc2, p=2)
        min_dist_to_pc1 = torch.min(dist_matrix, dim=0)[0]
        recall = (min_dist_to_pc1 < threshold).float().mean()
        min_dist_to_pc2 = torch.min(dist_matrix, dim=1)[0]
        precision = (min_dist_to_pc2 < threshold).float().mean()
        if precision + recall == 0:
            return 0.0
        return (2 * precision * recall / (precision + recall)).item()

def compute_hausdorff_distance(pc1, pc2):
    if isinstance(pc1, np.ndarray):
        pc1 = torch.tensor(pc1, device=device, dtype=torch.float32)
    if isinstance(pc2, np.ndarray):
        pc2 = torch.tensor(pc2, device=device, dtype=torch.float32)
    with torch.no_grad():
        dist_matrix = torch.cdist(pc1, pc2, p=2)
        h1 = torch.max(torch.min(dist_matrix, dim=1)[0])
        h2 = torch.max(torch.min(dist_matrix, dim=0)[0])
        return torch.max(h1, h2).item()

def compute_normal_consistency(pc1, pc2, k=20, seed=0):
    max_sample = 512
    rng = np.random.default_rng(int(seed))
    if len(pc1) > max_sample:
        indices = rng.choice(len(pc1), max_sample, replace=False)
        pc1_sample = pc1[indices]
    else:
        pc1_sample = pc1
    if len(pc2) > max_sample:
        indices = rng.choice(len(pc2), max_sample, replace=False)
        pc2_sample = pc2[indices]
    else:
        pc2_sample = pc2

    def estimate_normals(pc, k_eff=10):
        tree = KDTree(pc)
        normals = []
        for point in pc:
            dists, indices = tree.query(point, k=min(k_eff + 1, len(pc)))
            neighbors = pc[indices[1:]]
            centroid = np.mean(neighbors, axis=0)
            neighbors_centered = neighbors - centroid
            cov = np.dot(neighbors_centered.T, neighbors_centered)
            _, _, v = np.linalg.svd(cov)
            normal = v[:, -1]
            normal = normal / (np.linalg.norm(normal) + 1e-08)
            normals.append(normal)
        return np.array(normals)
    normals1 = estimate_normals(pc1_sample, k_eff=min(k, 10))
    normals2 = estimate_normals(pc2_sample, k_eff=min(k, 10))
    tree2 = KDTree(pc2_sample)
    _, indices = tree2.query(pc1_sample, k=1)
    matched_normals2 = normals2[indices.flatten()]
    cos_sim = np.sum(normals1 * matched_normals2, axis=1)
    cos_sim = np.abs(cos_sim)
    return np.mean(cos_sim)

def plot_point_cloud_snapshot(pc, step, distance_name, repeat_idx=0, save_subdir='snapshots', cmap='viridis', point_size=8):
    try:
        safe_name = distance_name.replace('(', '_').replace(')', '_').replace('=', '_').replace('*', 'star').replace('-', '_').replace(' ', '_')
        fig = plt.figure(figsize=(8, 8), dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_axis_off()
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        ax.xaxis.pane.set_alpha(0)
        ax.yaxis.pane.set_alpha(0)
        ax.zaxis.pane.set_alpha(0)
        x, y, z = (pc[:, 0], pc[:, 1], pc[:, 2])
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-10)
        ax.scatter(x, y, z, c=z_norm, cmap=cmap, s=point_size, alpha=0.9, depthshade=True, edgecolors='none')
        ax.view_init(elev=90, azim=-90)
        max_range = np.max(np.ptp(pc, axis=0)) / 2
        mid_x, mid_y, mid_z = (np.mean(pc[:, 0]), np.mean(pc[:, 1]), np.mean(pc[:, 2]))
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        save_path = os.path.join(result_dir, save_subdir, pair_named_filename(f'{safe_name}_rep{repeat_idx}_step_{step}.png'))
        plt.tight_layout(pad=0)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.1)
        plt.close()
        print(f'✓ 已保存: {os.path.basename(save_path)}')
    except Exception as e:
        print(f'警告：保存快照失败 {distance_name} step {step}: {e}')
        plt.close()

def plot_variant_combined_figure(source_pc, target_pc, step_pcs, variant_name, variant_idx, save_dir, record_steps):
    try:
        safe_name = variant_name.replace('(', '').replace(')', '').replace('=', '_').replace('*', 'star').replace('-', '_').replace(' ', '_').replace('{', '').replace('}', '').replace('^', '')
        n_steps = len(record_steps)
        fig = plt.figure(figsize=(2.0 * (2 + n_steps), 2.0), dpi=200)
        axes = []
        for i in range(2 + n_steps):
            ax = fig.add_subplot(1, 2 + n_steps, i + 1, projection='3d')
            axes.append(ax)
        if n_steps == 0:
            axes = [axes[0], axes[1]]
        variant_color = VARIANT_COLORS[variant_idx % len(VARIANT_COLORS)]

        def plot_single_pc(ax, pc, is_variant=False):
            ax.set_axis_off()
            ax.grid(False)
            for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                axis.pane.fill = False
                axis.pane.set_edgecolor('none')
                axis.pane.set_alpha(0)
            x, y, z = (pc[:, 0], pc[:, 1], pc[:, 2])
            if is_variant:
                ax.scatter(x, y, z, c=variant_color, s=8, alpha=0.85, depthshade=True, edgecolors='none')
            else:
                z_norm = (z - z.min()) / (z.max() - z.min() + 1e-10)
                ax.scatter(x, y, z, c=z_norm, cmap=SOURCE_TARGET_CMAP, s=8, alpha=0.9, depthshade=True, edgecolors='none')
            ax.view_init(elev=90, azim=-90)
            max_range = np.max(np.ptp(pc, axis=0)) / 2
            mid = np.mean(pc, axis=0)
            ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        plot_single_pc(axes[0], source_pc, is_variant=False)
        for i, step in enumerate(record_steps):
            if step in step_pcs:
                plot_single_pc(axes[1 + i], step_pcs[step], is_variant=True)
        plot_single_pc(axes[-1], target_pc, is_variant=False)
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
        save_path = pair_named_path(save_dir, f'combined_{safe_name}.png')
        plt.savefig(save_path, dpi=200, bbox_inches=None, pad_inches=0.02, facecolor='white', edgecolor='none')
        plt.close()
        print(f'  ✓ 组合图已保存: {os.path.basename(save_path)}')
    except Exception as e:
        print(f'  ✗ 保存组合图失败 {variant_name}: {e}')
        import traceback
        traceback.print_exc()
        plt.close()

def visualize_point_cloud(pc, title, save_path, cmap='viridis', point_size=8):
    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    ax.grid(False)
    x, y, z = (pc[:, 0], pc[:, 1], pc[:, 2])
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-10)
    ax.scatter(x, y, z, c=z_norm, cmap=cmap, s=point_size, alpha=0.9, depthshade=False, edgecolors='none')
    ax.set_title(title, fontsize=14, pad=20)
    ax.view_init(elev=90, azim=-90)
    max_range = np.max(np.ptp(pc, axis=0)) / 2
    mid = np.mean(pc, axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close()
    print(f'  ✓ 已保存: {os.path.basename(save_path)}')

def visualize_comparison(source_pc, target_pc, save_dir):
    fig = plt.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_axis_off()
    ax1.grid(False)
    for axis in [ax1.xaxis, ax1.yaxis, ax1.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor('none')
    sx, sy, sz = (source_pc[:, 0], source_pc[:, 1], source_pc[:, 2])
    z_norm_s = (sz - sz.min()) / (sz.max() - sz.min() + 1e-10)
    ax1.scatter(sx, sy, sz, c=z_norm_s, cmap='viridis', s=8, alpha=0.9, edgecolors='none')
    ax1.set_title(f'Source', fontsize=14, pad=20)
    ax1.view_init(elev=90, azim=-90)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_axis_off()
    ax2.grid(False)
    for axis in [ax2.xaxis, ax2.yaxis, ax2.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor('none')
    tx, ty, tz = (target_pc[:, 0], target_pc[:, 1], target_pc[:, 2])
    z_norm_t = (tz - tz.min()) / (tz.max() - tz.min() + 1e-10)
    ax2.scatter(tx, ty, tz, c=z_norm_t, cmap='viridis', s=8, alpha=0.9, edgecolors='none')
    ax2.set_title(f'Target', fontsize=14, pad=20)
    ax2.view_init(elev=90, azim=-90)
    for ax in [ax1, ax2]:
        max_range = max(np.max(np.ptp(source_pc, axis=0)), np.max(np.ptp(target_pc, axis=0))) / 2
        mid = np.mean(np.concatenate([source_pc, target_pc], axis=0), axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    plt.tight_layout(pad=0)
    save_path = pair_named_path(save_dir, '00_original_pair.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f'✓ 对比图已保存: {save_path}')

def compute_cohens_d(values1, values2):
    n1, n2 = (len(values1), len(values2))
    if n1 == 0 or n2 == 0:
        return None
    mean1, mean2 = (np.mean(values1), np.mean(values2))
    std1, std2 = (np.std(values1, ddof=1), np.std(values2, ddof=1))
    pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    cohens_d = (mean1 - mean2) / pooled_std
    if abs(cohens_d) < 0.2:
        magnitude = '可忽略(Negligible)'
    elif abs(cohens_d) < 0.5:
        magnitude = '小(Small)'
    elif abs(cohens_d) < 0.8:
        magnitude = '中(Medium)'
    else:
        magnitude = '大(Large)'
    return {'cohens_d': float(cohens_d), 'magnitude': magnitude, 'mean_diff': float(mean1 - mean2), 'pooled_std': float(pooled_std)}

def statistical_significance_test(gebsw_values, gsw_values, sw_values):
    results = {}
    gebsw_arr = np.array(gebsw_values) if gebsw_values else np.array([])
    gsw_arr = np.array(gsw_values) if gsw_values else np.array([])
    sw_arr = np.array(sw_values) if sw_values else np.array([])
    if len(gebsw_arr) < 2:
        return results
    if len(gsw_arr) >= 2:
        try:
            effective_len = min(len(gebsw_arr), len(gsw_arr))
            g_effective = gebsw_arr[:effective_len]
            gsw_effective = gsw_arr[:effective_len]
            t_stat, p_value = stats.ttest_rel(g_effective, gsw_effective)
            cohens_d_result = compute_cohens_d(g_effective, gsw_effective)
            results['GEBSW_vs_GSW'] = {'t_statistic': float(t_stat), 'p_value': float(p_value), 'significant_05': p_value < 0.05, 'significant_01': p_value < 0.01, 'cohens_d': cohens_d_result['cohens_d'] if cohens_d_result else None, 'effect_magnitude': cohens_d_result['magnitude'] if cohens_d_result else None, 'mean_diff': cohens_d_result['mean_diff'] if cohens_d_result else None, 'sample_size': effective_len}
        except Exception as e:
            print(f'检验失败: {e}')
    if len(sw_arr) >= 2:
        try:
            effective_len = min(len(gebsw_arr), len(sw_arr))
            s_effective = gebsw_arr[:effective_len]
            sw_effective = sw_arr[:effective_len]
            t_stat, p_value = stats.ttest_rel(s_effective, sw_effective)
            cohens_d_result = compute_cohens_d(s_effective, sw_effective)
            results['GEBSW_vs_SW'] = {'t_statistic': float(t_stat), 'p_value': float(p_value), 'significant_05': p_value < 0.05, 'significant_01': p_value < 0.01, 'cohens_d': cohens_d_result['cohens_d'] if cohens_d_result else None, 'effect_magnitude': cohens_d_result['magnitude'] if cohens_d_result else None, 'mean_diff': cohens_d_result['mean_diff'] if cohens_d_result else None, 'sample_size': effective_len}
        except Exception as e:
            print(f'检验失败: {e}')
    return results

def get_distance_functions_corrected(proj_seed, repeat_idx, current_step=0, total_steps=NUM_STEPS, temp_start=TEMPERATURE_START, temp_end=TEMPERATURE_END):
    distance_functions = {}
    step_ratio = current_step / total_steps

    def compute_energy_weights(wd_tensor, energy_type, energy_r):
        wd_tensor = wd_tensor.clamp(min=EPS)
        if USE_TEMPERATURE_ANNEALING:
            current_temp = temp_start * (temp_end / temp_start) ** step_ratio
        else:
            current_temp = 1.0
        if energy_type == 'exp':
            logits = wd_tensor / (wd_tensor.mean() + EPS) / current_temp
            weights = F.softmax(logits, dim=0)
        elif energy_type == 'poly':
            logits = torch.pow(wd_tensor, energy_r) / current_temp
            weights = F.softmax(logits, dim=0)
        else:
            weights = torch.ones_like(wd_tensor) / len(wd_tensor)
        if USE_SOFTMAX_WEIGHT:
            weights = weights.clamp(max=WEIGHT_CLIP_MAX)
            weights = weights / weights.sum()
        return weights

    def _projection_seed(degree):
        return int(proj_seed) + int(degree) * 100003

    def _theta_seed(degree):
        return int(proj_seed) + int(degree) * 200003

    def _project(X, degree):
        return polynomial_projection_controlled(X, degree=degree, max_dim=UNIFIED_MAX_DIM, seed=_projection_seed(degree))

    def _shared_projected_wd_tensor(X, Y, degree):
        X_proj = _project(X, degree)
        Y_proj = _project(Y, degree)
        theta = rand_projections(X_proj.shape[1], num_projections=L_BASE, seed=_theta_seed(degree))
        X_proj_1d = torch.matmul(X_proj, theta.T)
        Y_proj_1d = torch.matmul(Y_proj, theta.T)
        return torch.stack([one_dimensional_wasserstein(X_proj_1d[:, i], Y_proj_1d[:, i]) for i in range(L_BASE)])

    def _uniform_sw(X, Y, degree):
        wd_tensor = _shared_projected_wd_tensor(X, Y, degree)
        return torch.mean(wd_tensor)

    def _weighted_sw(X, Y, degree, energy_type, energy_r):
        wd_tensor = _shared_projected_wd_tensor(X, Y, degree)
        weights = compute_energy_weights(wd_tensor, energy_type, energy_r)
        return torch.sum(wd_tensor * weights)

    def gsw_poly_q1(X, Y):
        return _uniform_sw(X, Y, degree=1)
    distance_functions['GEBW(C,1) (SW-Baseline)'] = gsw_poly_q1

    def gebsw_exp_poly_q1(X, Y):
        return _weighted_sw(X, Y, degree=1, energy_type='exp', energy_r=1)
    distance_functions['GEBW(e,1) (EBSW-Baseline)'] = gebsw_exp_poly_q1

    def gebsw_poly_r_1_poly_q1(X, Y):
        return _weighted_sw(X, Y, degree=1, energy_type='poly', energy_r=1)
    distance_functions['GEBW(1,1) (EBSW-Baseline)'] = gebsw_poly_r_1_poly_q1

    def gebsw_poly_r_2_poly_q1(X, Y):
        return _weighted_sw(X, Y, degree=1, energy_type='poly', energy_r=2)
    distance_functions['GEBW(2,1) (EBSW-Baseline)'] = gebsw_poly_r_2_poly_q1

    def gebsw_poly_r_3_poly_q1(X, Y):
        return _weighted_sw(X, Y, degree=1, energy_type='poly', energy_r=3)
    distance_functions['GEBW(3,1) (EBSW-Baseline)'] = gebsw_poly_r_3_poly_q1

    def gebsw_poly_r_4_poly_q1(X, Y):
        return _weighted_sw(X, Y, degree=1, energy_type='poly', energy_r=4)
    distance_functions['GEBW(4,1) (EBSW-Baseline)'] = gebsw_poly_r_4_poly_q1

    def gsw_poly_q3(X, Y):
        return _uniform_sw(X, Y, degree=3)
    distance_functions['GEBW(C,3) (GSW-Baseline)'] = gsw_poly_q3

    def gebsw_exp_poly_q3(X, Y):
        return _weighted_sw(X, Y, degree=3, energy_type='exp', energy_r=1)
    distance_functions['GEBW(e,3)'] = gebsw_exp_poly_q3

    def gebsw_poly_r_1_poly_q3(X, Y):
        return _weighted_sw(X, Y, degree=3, energy_type='poly', energy_r=1)
    distance_functions['GEBW(1,3)'] = gebsw_poly_r_1_poly_q3

    def gebsw_poly_r_2_poly_q3(X, Y):
        return _weighted_sw(X, Y, degree=3, energy_type='poly', energy_r=2)
    distance_functions['GEBW(2,3)'] = gebsw_poly_r_2_poly_q3

    def gebsw_poly_r_3_poly_q3(X, Y):
        return _weighted_sw(X, Y, degree=3, energy_type='poly', energy_r=3)
    distance_functions['GEBW(3,3)'] = gebsw_poly_r_3_poly_q3

    def gebsw_poly_r_4_poly_q3(X, Y):
        return _weighted_sw(X, Y, degree=3, energy_type='poly', energy_r=4)
    distance_functions['GEBW(4,3)'] = gebsw_poly_r_4_poly_q3

    def gsw_poly_q5(X, Y):
        return _uniform_sw(X, Y, degree=5)
    distance_functions['GEBW(C,5) (GSW-Baseline)'] = gsw_poly_q5

    def gebsw_exp_poly_q5(X, Y):
        return _weighted_sw(X, Y, degree=5, energy_type='exp', energy_r=1)
    distance_functions['GEBW(e,5)'] = gebsw_exp_poly_q5

    def gebsw_poly_r_1_poly_q5(X, Y):
        return _weighted_sw(X, Y, degree=5, energy_type='poly', energy_r=1)
    distance_functions['GEBW(1,5)'] = gebsw_poly_r_1_poly_q5

    def gebsw_poly_r_2_poly_q5(X, Y):
        return _weighted_sw(X, Y, degree=5, energy_type='poly', energy_r=2)
    distance_functions['GEBW(2,5)'] = gebsw_poly_r_2_poly_q5

    def gebsw_poly_r_3_poly_q5(X, Y):
        return _weighted_sw(X, Y, degree=5, energy_type='poly', energy_r=3)
    distance_functions['GEBW(3,5)'] = gebsw_poly_r_3_poly_q5

    def gebsw_poly_r_4_poly_q5(X, Y):
        return _weighted_sw(X, Y, degree=5, energy_type='poly', energy_r=4)
    distance_functions['GEBW(4,5)'] = gebsw_poly_r_4_poly_q5
    return distance_functions

def run_single_experiment(source_pc, target_pc, dist_name, proj_seed, perturbation_seed, temp_start=2.0, temp_end=0.1):
    target_tensor = torch.tensor(target_pc, dtype=torch.float32, device=device)
    np.random.seed(perturbation_seed)
    torch.manual_seed(perturbation_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(perturbation_seed)
    source_tensor = torch.tensor(source_pc, dtype=torch.float32, device=device, requires_grad=True)
    perturbation = torch.randn_like(source_tensor) * INITIAL_PERTURBATION
    source_tensor.data = source_tensor.data + perturbation
    optimizer = torch.optim.Adam([source_tensor], lr=LR, betas=(MOMENTUM, 0.999), weight_decay=1e-05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_STEPS, eta_min=LR * 0.01)
    for step in range(NUM_STEPS + 1):
        optimizer.zero_grad()
        current_dist_funcs = get_distance_functions_corrected(proj_seed, 0, step, NUM_STEPS, temp_start, temp_end)
        current_dist_func = current_dist_funcs[dist_name]
        distance = current_dist_func(source_tensor, target_tensor)
        distance.backward()
        torch.nn.utils.clip_grad_norm_([source_tensor], max_norm=1.0)
        optimizer.step()
        scheduler.step()
    final_pc = source_tensor.detach()
    cd = chamfer_distance(final_pc, target_pc)
    fscore = compute_f_score(final_pc, target_pc)
    nc = compute_normal_consistency(final_pc.cpu().numpy(), target_pc, seed=perturbation_seed + NUM_STEPS)
    hd = compute_hausdorff_distance(final_pc, target_pc)
    return {'CD': cd, 'FScore': fscore, 'NC': nc, 'HD': hd}

def hyperparameter_sensitivity_analysis(source_pc, target_pc):
    print(f"\n{'=' * 60}")
    print('开始超参数敏感性分析...')
    print(f"{'=' * 60}")
    test_variants = ['GEBSW-f^*_e-Proj-poly(q=3)', f'GEBSW-f^*_{{{ENERGY_ORDER_SYMBOL}=2}}-Proj-poly(q=3)']
    baseline_name = 'GSW-Proj-poly(q=3) (GSW-Baseline)'
    results = {variant: {temp: [] for temp in TEMPERATURE_RANGE} for variant in test_variants}
    baseline_results = []
    print(f'\n[1/3] 计算基线 {baseline_name} ...')
    for rep in range(SENSITIVITY_REPEAT_TIMES):
        proj_seed = 2024 + rep * 1000 + 50000
        perturbation_seed = proj_seed + 100
        metrics = run_single_experiment(source_pc, target_pc, baseline_name, proj_seed, perturbation_seed)
        baseline_results.append(metrics)
        print(f"  重复{rep + 1}: CD={metrics['CD']:.4f}, FScore={metrics['FScore']:.4f}")
    baseline_mean = {k: np.mean([r[k] for r in baseline_results]) for k in ['CD', 'FScore', 'NC', 'HD']}
    baseline_std = {k: np.std([r[k] for r in baseline_results]) for k in ['CD', 'FScore', 'NC', 'HD']}
    print(f"\n基线均值: CD={baseline_mean['CD']:.4f}±{baseline_std['CD']:.4f}, FScore={baseline_mean['FScore']:.4f}±{baseline_std['FScore']:.4f}")
    for idx, temp in enumerate(TEMPERATURE_RANGE, 1):
        print(f'\n[2/3] 测试温度 T={temp} ({idx}/{len(TEMPERATURE_RANGE)})...')
        for variant in test_variants:
            variant_results = []
            for rep in range(SENSITIVITY_REPEAT_TIMES):
                proj_seed = 2024 + rep * 1000 + int(temp * 1000)
                perturbation_seed = proj_seed + 100
                metrics = run_single_experiment(source_pc, target_pc, variant, proj_seed, perturbation_seed, temp, 0.1)
                variant_results.append(metrics)
                print(f"  {variant} 重复{rep + 1}: CD={metrics['CD']:.4f}, FScore={metrics['FScore']:.4f}")
            results[variant][temp] = variant_results
    print(f'\n[3/3] 生成敏感性分析报告...')
    analyze_and_save_sensitivity(results, baseline_mean, baseline_std, test_variants)
    return (results, baseline_mean)

def analyze_and_save_sensitivity(results, baseline_mean, baseline_std, test_variants):
    summary_data = []
    for variant in test_variants:
        safe_variant = variant.replace('^', '').replace('{', '').replace('}', '').replace('*', 'star')
        temps = []
        cd_means, cd_stds = ([], [])
        fscore_means, fscore_stds = ([], [])
        for temp in TEMPERATURE_RANGE:
            if temp in results[variant] and len(results[variant][temp]) > 0:
                temps.append(temp)
                cds = [r['CD'] for r in results[variant][temp]]
                fscores = [r['FScore'] for r in results[variant][temp]]
                cd_means.append(np.mean(cds))
                cd_stds.append(np.std(cds))
                fscore_means.append(np.mean(fscores))
                fscore_stds.append(np.std(fscores))
                from scipy import stats
                if len(cds) == len(results[variant][temp]):
                    t_stat_cd, p_val_cd = stats.ttest_ind(cds, [baseline_mean['CD']] * len(cds))
                    t_stat_fs, p_val_fs = stats.ttest_ind(fscores, [baseline_mean['FScore']] * len(fscores))
                else:
                    p_val_cd, p_val_fs = (1.0, 1.0)
                summary_data.append({'Variant': variant, 'Temperature': temp, 'CD_Mean': np.mean(cds), 'CD_Std': np.std(cds), 'FScore_Mean': np.mean(fscores), 'FScore_Std': np.std(fscores), 'Better_than_GSW_CD': np.mean(cds) < baseline_mean['CD'], 'Better_than_GSW_FScore': np.mean(fscores) > baseline_mean['FScore'], 'P_value_CD': p_val_cd, 'P_value_FScore': p_val_fs})
        if len(temps) == 0:
            print(f'  警告：{variant} 无有效数据')
            continue
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            ax1.errorbar(temps, cd_means, yerr=cd_stds, fmt='b-o', capsize=5, label=variant, linewidth=2, markersize=6)
            ax1.axhline(y=baseline_mean['CD'], color='r', linestyle='--', label=f"GSW Baseline ({baseline_mean['CD']:.4f})")
            ax1.axhspan(baseline_mean['CD'] - baseline_std['CD'], baseline_mean['CD'] + baseline_std['CD'], alpha=0.1, color='r')
            ax1.set_xlabel('Temperature Start', fontsize=12)
            ax1.set_ylabel('Chamfer Distance (lower is better)', fontsize=12)
            ax1.set_title(f'{variant}\nSensitivity to Temperature - CD', fontsize=12)
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            ax2.errorbar(temps, fscore_means, yerr=fscore_stds, fmt='g-o', capsize=5, label=variant, linewidth=2, markersize=6)
            ax2.axhline(y=baseline_mean['FScore'], color='r', linestyle='--', label=f"GSW Baseline ({baseline_mean['FScore']:.4f})")
            ax2.axhspan(baseline_mean['FScore'] - baseline_std['FScore'], baseline_mean['FScore'] + baseline_std['FScore'], alpha=0.1, color='r')
            ax2.set_xlabel('Temperature Start', fontsize=12)
            ax2.set_ylabel('F-Score (higher is better)', fontsize=12)
            ax2.set_title(f'{variant}\nSensitivity to Temperature - FScore', fontsize=12)
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            save_path = pair_named_path(os.path.join(result_dir, 'sensitivity'), f'sensitivity_{safe_variant}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f'  ✓ 敏感性曲线已保存: {save_path}')
        except Exception as e:
            print(f'  ✗ 绘制 {variant} 曲线时出错: {e}')
            plt.close()
    try:
        summary_df = pd.DataFrame(summary_data)
        excel_path = pair_named_path(os.path.join(result_dir, 'sensitivity'), 'sensitivity_analysis_summary.xlsx')
        summary_df.to_excel(excel_path, index=False)
        print(f'  ✓ 敏感性数据已保存: {excel_path}')
        print(f'\n敏感性分析结论:')
        print(f"  基线 GSW: CD={baseline_mean['CD']:.4f}±{baseline_std['CD']:.4f}, FScore={baseline_mean['FScore']:.4f}±{baseline_std['FScore']:.4f}")
        for variant in test_variants:
            better_temps_cd = [t for t in TEMPERATURE_RANGE if t in results[variant] and np.mean([r['CD'] for r in results[variant][t]]) < baseline_mean['CD']]
            better_temps_fs = [t for t in TEMPERATURE_RANGE if t in results[variant] and np.mean([r['FScore'] for r in results[variant][t]]) > baseline_mean['FScore']]
            print(f'\n  {variant}:')
            print(f"    优于基线的温度范围(CD): {(better_temps_cd if better_temps_cd else '无')}")
            print(f"    优于基线的温度范围(FScore): {(better_temps_fs if better_temps_fs else '无')}")
    except Exception as e:
        print(f'  ✗ 保存Excel时出错: {e}')

def plot_convergence_curves(all_results_by_dist, distance_names, metrics):
    for metric_name, metric_key in metrics:
        try:
            n_methods = max(1, len(distance_names))
            n_cols = 5 if n_methods > 18 else 6
            n_rows = int(np.ceil(n_methods / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
            axes = np.array(axes).reshape(-1)
            for idx, dist_name in enumerate(distance_names):
                if idx >= len(axes):
                    break
                ax = axes[idx]
                steps, means, stds = ([], [], [])
                for step in RECORD_STEPS:
                    key = f'step_{step}_{metric_key}'
                    if key in all_results_by_dist.get(dist_name, {}):
                        values = all_results_by_dist[dist_name][key]
                        steps.append(step)
                        means.append(np.mean(values))
                        stds.append(np.std(values))
                if steps:
                    ax.plot(steps, means, 'b-', linewidth=2, label=dist_name[:30])
                    ax.fill_between(steps, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.3)
                    ax.set_xlabel('Step')
                    ax.set_ylabel(metric_name)
                    ax.set_title(dist_name[:40])
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=8)
            for idx in range(len(distance_names), len(axes)):
                axes[idx].set_visible(False)
            plt.tight_layout()
            save_path = pair_named_path(os.path.join(result_dir, 'curves'), f"convergence_{metric_name.replace(' ', '_')}.png")
            plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f'  ✓ 收敛曲线已保存: {save_path}')
        except Exception as e:
            print(f'  ✗ 绘制 {metric_name} 曲线时出错: {e}')
            plt.close()

def parse_gebsw_design(method_name):
    name = method_name
    if 'C,1' in name:
        return {'family_role': 'framework_special_case', 'energy_rule': 'C', 'energy_type': 'uniform', 'energy_order': 0, 'q': 1, 'special_case': 'SW'}
    if 'C,3' in name:
        return {'family_role': 'framework_special_case', 'energy_rule': 'C', 'energy_type': 'uniform', 'energy_order': 0, 'q': 3, 'special_case': 'GSW'}
    if 'C,5' in name:
        return {'family_role': 'framework_special_case', 'energy_rule': 'C', 'energy_type': 'uniform', 'energy_order': 0, 'q': 5, 'special_case': 'GSW'}
    if 'e,1' in name:
        return {'family_role': 'framework_special_case', 'energy_rule': 'e', 'energy_type': 'exp', 'energy_order': np.nan, 'q': 1, 'special_case': 'EBSW'}
    if 'e,3' in name:
        return {'family_role': 'GEBSW_configuration', 'energy_rule': 'e', 'energy_type': 'exp', 'energy_order': np.nan, 'q': 3, 'special_case': 'GEBSW'}
    if 'e,5' in name:
        return {'family_role': 'GEBSW_configuration', 'energy_rule': 'e', 'energy_type': 'exp', 'energy_order': np.nan, 'q': 5, 'special_case': 'GEBSW'}
    for r in [1, 2, 3, 4]:
        for q in [1, 3, 5]:
            if f'({r},{q})' in name:
                return {'family_role': 'framework_special_case' if q == 1 else 'GEBSW_configuration', 'energy_rule': f'r={r}', 'energy_type': 'poly', 'energy_order': r, 'q': q, 'special_case': 'EBSW' if q == 1 else 'GEBSW'}
    if 'Sampled-MaxSW' in name or 'Max-SW' in name:
        return {'family_role': 'external_sliced_ot_baseline', 'energy_rule': 'adaptive-max', 'energy_type': 'max', 'energy_order': np.nan, 'q': np.nan, 'special_case': 'Sampled-MaxSW'}
    if 'DSW' in name or 'Distributional' in name:
        return {'family_role': 'external_sliced_ot_baseline', 'energy_rule': 'learned-distribution', 'energy_type': 'distributional', 'energy_order': np.nan, 'q': np.nan, 'special_case': 'DSW'}
    return {'family_role': 'unknown', 'energy_rule': 'unknown', 'energy_type': 'unknown', 'energy_order': np.nan, 'q': np.nan, 'special_case': 'unknown'}

def _dominates(a, b, metric_specs):
    not_worse_all = True
    strictly_better_any = False
    for metric, lower_better in metric_specs:
        av, bv = (a[metric], b[metric])
        if pd.isna(av) or pd.isna(bv):
            return False
        if lower_better:
            if av > bv + 1e-12:
                not_worse_all = False
                break
            if av < bv - 1e-12:
                strictly_better_any = True
        else:
            if av < bv - 1e-12:
                not_worse_all = False
                break
            if av > bv + 1e-12:
                strictly_better_any = True
    return not_worse_all and strictly_better_any

def compute_pareto_flags(df, metric_specs, group_cols=('pair_index', 'repeat')):
    out = df.copy()
    out['pareto'] = False
    out['dominated_by_count'] = 0
    out['dominates_count'] = 0
    for _, g in out.groupby(list(group_cols), dropna=False):
        idxs = list(g.index)
        for i in idxs:
            dominated_by = 0
            dominates = 0
            for j in idxs:
                if i == j:
                    continue
                if _dominates(out.loc[j], out.loc[i], metric_specs):
                    dominated_by += 1
                if _dominates(out.loc[i], out.loc[j], metric_specs):
                    dominates += 1
            out.at[i, 'dominated_by_count'] = dominated_by
            out.at[i, 'dominates_count'] = dominates
            out.at[i, 'pareto'] = dominated_by == 0
    return out

def build_framework_long_table(all_results_by_dist, final_step):
    rows = []
    for method, data in all_results_by_dist.items():
        n = len(data.get('total_time', []))
        for repeat_idx in range(n):
            row = {'pair_index': CURRENT_PAIR_INDEX, 'pair_group': CURRENT_PAIR_GROUP, 'source': CURRENT_SOURCE_NAME, 'target': CURRENT_TARGET_NAME, 'repeat': repeat_idx + 1, 'method': method, 'W2': data.get(f'step_{final_step}_distance', [np.nan] * n)[repeat_idx] if repeat_idx < len(data.get(f'step_{final_step}_distance', [])) else np.nan, 'CD': data.get(f'step_{final_step}_cd', [np.nan] * n)[repeat_idx] if repeat_idx < len(data.get(f'step_{final_step}_cd', [])) else np.nan, 'FScore': data.get(f'step_{final_step}_fscore', [np.nan] * n)[repeat_idx] if repeat_idx < len(data.get(f'step_{final_step}_fscore', [])) else np.nan, 'NC': data.get(f'step_{final_step}_normal_consistency', [np.nan] * n)[repeat_idx] if repeat_idx < len(data.get(f'step_{final_step}_normal_consistency', [])) else np.nan, 'HD': data.get(f'step_{final_step}_hausdorff', [np.nan] * n)[repeat_idx] if repeat_idx < len(data.get(f'step_{final_step}_hausdorff', [])) else np.nan, 'Runtime': data.get('total_time', [np.nan] * n)[repeat_idx] if repeat_idx < len(data.get('total_time', [])) else np.nan}
            row.update(parse_gebsw_design(method))
            rows.append(row)
    return pd.DataFrame(rows)

def add_average_ranks(df):
    out = df.copy()
    rank_specs = [('W2', True), ('CD', True), ('FScore', False), ('NC', False), ('HD', True), ('Runtime', True)]
    for metric, lower_better in rank_specs:
        out[f'rank_{metric}'] = out.groupby(['pair_index', 'repeat'])[metric].rank(ascending=lower_better, method='average')
    quality_rank_cols = ['rank_W2', 'rank_CD', 'rank_FScore', 'rank_NC', 'rank_HD']
    out['avg_quality_rank'] = out[quality_rank_cols].mean(axis=1)
    out['avg_all_rank'] = out[quality_rank_cols + ['rank_Runtime']].mean(axis=1)
    return out

def compute_factor_effects(df):
    rows = []
    metrics = ['W2', 'CD', 'FScore', 'NC', 'HD', 'Runtime', 'avg_quality_rank']
    d = df.dropna(subset=['q', 'energy_rule']).copy()
    d['q'] = d['q'].astype(int).astype(str)
    for metric in metrics:
        dd = d.dropna(subset=[metric])
        if dd.empty:
            continue
        y = dd[metric].astype(float)
        grand = y.mean()
        ss_total = float(((y - grand) ** 2).sum())
        if ss_total <= 1e-20:
            continue
        means_q = dd.groupby('q')[metric].mean()
        counts_q = dd.groupby('q')[metric].count()
        ss_q = float(sum((counts_q[k] * (means_q[k] - grand) ** 2 for k in means_q.index)))
        means_e = dd.groupby('energy_rule')[metric].mean()
        counts_e = dd.groupby('energy_rule')[metric].count()
        ss_e = float(sum((counts_e[k] * (means_e[k] - grand) ** 2 for k in means_e.index)))
        means_qe = dd.groupby(['q', 'energy_rule'])[metric].mean()
        counts_qe = dd.groupby(['q', 'energy_rule'])[metric].count()
        ss_qe_cells = float(sum((counts_qe[k] * (means_qe[k] - grand) ** 2 for k in means_qe.index)))
        ss_interaction = max(0.0, ss_qe_cells - ss_q - ss_e)
        rows.append({'metric': metric, 'eta2_q': ss_q / ss_total, 'eta2_energy': ss_e / ss_total, 'eta2_q_x_energy': ss_interaction / ss_total, 'interpretation': 'larger eta2 means this design axis explains more variance'})
    return pd.DataFrame(rows)

def save_design_space_heatmaps(summary_df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    try:
        metrics = ['W2', 'CD', 'FScore', 'NC', 'HD', 'Runtime', 'pareto_quality_rate', 'avg_quality_rank']
        for metric in metrics:
            if metric not in summary_df.columns:
                continue
            pivot = summary_df.pivot_table(index='energy_rule', columns='q', values=metric, aggfunc='mean')
            if pivot.empty:
                continue
            plt.figure(figsize=(7, 4.5), dpi=160)
            im = plt.imshow(pivot.values, aspect='auto')
            plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
            plt.yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
            plt.xlabel('Projection order q')
            plt.ylabel('Energy / weighting rule')
            plt.title(f'GEBSW design-space heatmap: {metric}')
            plt.colorbar(im, fraction=0.046, pad=0.04)
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    val = pivot.values[i, j]
                    if pd.notna(val):
                        plt.text(j, i, f'{val:.3g}', ha='center', va='center', fontsize=8)
            plt.tight_layout()
            plt.savefig(pair_named_path(save_dir, f'heatmap_{metric}.png'), bbox_inches='tight', facecolor='white')
            plt.close()
    except Exception as e:
        print(f'  ✗ heatmap绘制失败: {e}')
        plt.close()

def framework_design_space_analysis(all_results_by_dist, final_step):
    print('\n开始统一框架设计空间分析...')
    out_dir = f'{result_dir}/framework_analysis'
    os.makedirs(out_dir, exist_ok=True)
    long_df = build_framework_long_table(all_results_by_dist, final_step)
    if long_df.empty:
        print('  ⚠️  无可分析数据')
        return
    quality_specs = [('W2', True), ('CD', True), ('FScore', False), ('NC', False), ('HD', True)]
    efficiency_specs = quality_specs + [('Runtime', True)]
    q_pareto = compute_pareto_flags(long_df, quality_specs).rename(columns={'pareto': 'pareto_quality'})
    e_pareto = compute_pareto_flags(long_df, efficiency_specs).rename(columns={'pareto': 'pareto_efficiency'})
    long_df['pareto_quality'] = q_pareto['pareto_quality'].values
    long_df['quality_dominated_by_count'] = q_pareto['dominated_by_count'].values
    long_df['quality_dominates_count'] = q_pareto['dominates_count'].values
    long_df['pareto_efficiency'] = e_pareto['pareto_efficiency'].values
    long_df['efficiency_dominated_by_count'] = e_pareto['dominated_by_count'].values
    long_df['efficiency_dominates_count'] = e_pareto['dominates_count'].values
    long_df = add_average_ranks(long_df)
    summary = long_df.groupby(['method', 'q', 'energy_rule', 'energy_type', 'special_case', 'family_role'], dropna=False).agg(W2=('W2', 'mean'), CD=('CD', 'mean'), FScore=('FScore', 'mean'), NC=('NC', 'mean'), HD=('HD', 'mean'), Runtime=('Runtime', 'mean'), W2_std=('W2', 'std'), CD_std=('CD', 'std'), FScore_std=('FScore', 'std'), NC_std=('NC', 'std'), HD_std=('HD', 'std'), Runtime_std=('Runtime', 'std'), pareto_quality_rate=('pareto_quality', 'mean'), pareto_efficiency_rate=('pareto_efficiency', 'mean'), avg_quality_rank=('avg_quality_rank', 'mean'), avg_all_rank=('avg_all_rank', 'mean'), quality_dominates_count=('quality_dominates_count', 'mean'), quality_dominated_by_count=('quality_dominated_by_count', 'mean'), n=('method', 'count')).reset_index()
    summary['pareto_quality_rate'] *= 100.0
    summary['pareto_efficiency_rate'] *= 100.0
    summary = summary.sort_values(['pareto_quality_rate', 'avg_quality_rank'], ascending=[False, True])
    difficulty_summary = long_df.groupby(['pair_group', 'method', 'q', 'energy_rule'], dropna=False).agg(pareto_quality_rate=('pareto_quality', 'mean'), pareto_efficiency_rate=('pareto_efficiency', 'mean'), avg_quality_rank=('avg_quality_rank', 'mean'), W2=('W2', 'mean'), CD=('CD', 'mean'), FScore=('FScore', 'mean'), NC=('NC', 'mean'), HD=('HD', 'mean'), Runtime=('Runtime', 'mean'), n=('method', 'count')).reset_index()
    difficulty_summary['pareto_quality_rate'] *= 100.0
    difficulty_summary['pareto_efficiency_rate'] *= 100.0
    factor_effects = compute_factor_effects(long_df)
    excel_path = pair_named_path(out_dir, 'GEBSW_framework_design_space_analysis.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        long_df.to_excel(writer, sheet_name='long_pair_repeat_method', index=False)
        summary.to_excel(writer, sheet_name='design_space_summary', index=False)
        difficulty_summary.to_excel(writer, sheet_name='difficulty_summary', index=False)
        factor_effects.to_excel(writer, sheet_name='factor_effect_eta2', index=False)
        long_df[long_df['pareto_quality']].to_excel(writer, sheet_name='quality_pareto_points', index=False)
        long_df[long_df['pareto_efficiency']].to_excel(writer, sheet_name='efficiency_pareto_points', index=False)
    save_design_space_heatmaps(summary, f'{out_dir}/heatmaps')
    print(f'  ✓ 统一框架分析已保存: {excel_path}')

def aggregate_framework_analysis_from_dirs(base_dir='.'):
    files = []
    for root, dirs, names in os.walk(base_dir):
        for name in names:
            if name.endswith('GEBSW_framework_design_space_analysis.xlsx'):
                files.append(os.path.join(root, name))
    if not files:
        print('未找到可汇总的framework_analysis文件。')
        return
    all_long = []
    for f in files:
        try:
            df = pd.read_excel(f, sheet_name='long_pair_repeat_method')
            df['source_file'] = f
            all_long.append(df)
        except Exception as e:
            print(f'  跳过 {f}: {e}')
    if not all_long:
        return
    combined = pd.concat(all_long, ignore_index=True)
    out_dir = 'PCR-ShapeNet-GEBSW-Framework-ExternalBaselines-Aggregate' if INCLUDE_EXTERNAL_BASELINES else 'PCR-ShapeNet-GEBSW-Framework-Aggregate'
    os.makedirs(out_dir, exist_ok=True)
    quality_specs = [('W2', True), ('CD', True), ('FScore', False), ('NC', False), ('HD', True)]
    efficiency_specs = quality_specs + [('Runtime', True)]
    combined_q = compute_pareto_flags(combined, quality_specs, group_cols=('pair_index', 'repeat'))
    combined_e = compute_pareto_flags(combined, efficiency_specs, group_cols=('pair_index', 'repeat'))
    combined['pareto_quality_global'] = combined_q['pareto'].values
    combined['pareto_efficiency_global'] = combined_e['pareto'].values
    combined = add_average_ranks(combined)
    summary = combined.groupby(['method', 'q', 'energy_rule', 'energy_type', 'special_case', 'family_role'], dropna=False).agg(W2=('W2', 'mean'), CD=('CD', 'mean'), FScore=('FScore', 'mean'), NC=('NC', 'mean'), HD=('HD', 'mean'), Runtime=('Runtime', 'mean'), pareto_quality_rate=('pareto_quality_global', 'mean'), pareto_efficiency_rate=('pareto_efficiency_global', 'mean'), avg_quality_rank=('avg_quality_rank', 'mean'), avg_all_rank=('avg_all_rank', 'mean'), n=('method', 'count')).reset_index()
    summary['pareto_quality_rate'] *= 100.0
    summary['pareto_efficiency_rate'] *= 100.0
    summary = summary.sort_values(['pareto_quality_rate', 'avg_quality_rank'], ascending=[False, True])
    difficulty_summary = combined.groupby(['pair_group', 'method', 'q', 'energy_rule'], dropna=False).agg(pareto_quality_rate=('pareto_quality_global', 'mean'), pareto_efficiency_rate=('pareto_efficiency_global', 'mean'), avg_quality_rank=('avg_quality_rank', 'mean'), W2=('W2', 'mean'), CD=('CD', 'mean'), FScore=('FScore', 'mean'), NC=('NC', 'mean'), HD=('HD', 'mean'), Runtime=('Runtime', 'mean'), n=('method', 'count')).reset_index()
    difficulty_summary['pareto_quality_rate'] *= 100.0
    difficulty_summary['pareto_efficiency_rate'] *= 100.0
    effects = compute_factor_effects(combined)
    out_xlsx = os.path.join(out_dir, 'ShapeNet_all_pairs_framework_analysis.xlsx')
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
        combined.to_excel(writer, sheet_name='all_long_results', index=False)
        summary.to_excel(writer, sheet_name='overall_design_summary', index=False)
        difficulty_summary.to_excel(writer, sheet_name='difficulty_summary', index=False)
        effects.to_excel(writer, sheet_name='factor_effect_eta2', index=False)
    save_design_space_heatmaps(summary, os.path.join(out_dir, 'heatmaps'))
    print(f'\n✅ ShapeNet统一框架总分析已保存: {out_xlsx}')

def point_cloud_reconstruction_experiment_final():
    print('加载点云数据...')
    global CURRENT_SOURCE_PC, CURRENT_TARGET_PC
    if CURRENT_SOURCE_PC is not None and CURRENT_TARGET_PC is not None:
        source_pc = np.asarray(CURRENT_SOURCE_PC, dtype=np.float32).copy()
        target_pc = np.asarray(CURRENT_TARGET_PC, dtype=np.float32).copy()
    else:
        source_pc = read_off_file_robust(source_path)
        target_pc = read_off_file_robust(target_path)
    source_pc = normalize_point_cloud(source_pc)
    target_pc = normalize_point_cloud(target_pc)
    target_tensor = torch.tensor(target_pc, dtype=torch.float32, device=device)
    raw_results = []
    all_results_by_dist = {}
    temp_funcs = get_all_distance_functions(2024, 0, 0, NUM_STEPS)
    all_distance_names = list(temp_funcs.keys())
    external_note = ' + 2 个外部 sliced-OT baseline' if INCLUDE_EXTERNAL_BASELINES else ''
    print(f'共有 {len(all_distance_names)} 种距离变体{external_note}: {all_distance_names}')
    variant_name_to_idx = {name: idx for idx, name in enumerate(all_distance_names)}
    for repeat in range(REPEAT_TIMES):
        print(f'\n===== 重复实验 {repeat + 1}/{REPEAT_TIMES} =====')
        base_seed = 2024 + repeat * 100000
        proj_seed = base_seed + 1000
        perturbation_seed = base_seed + 5000
        distance_functions = get_all_distance_functions(proj_seed, repeat, current_step=0)
        for dist_name in distance_functions.keys():
            if dist_name not in all_results_by_dist:
                all_results_by_dist[dist_name] = {}
        for dist_idx, (dist_name, dist_func) in enumerate(tqdm(distance_functions.items(), desc=f'Rep{repeat + 1}')):
            try:
                np.random.seed(perturbation_seed)
                torch.manual_seed(perturbation_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(perturbation_seed)
                source_tensor = torch.tensor(source_pc, dtype=torch.float32, device=device, requires_grad=True)
                perturbation = torch.randn_like(source_tensor) * INITIAL_PERTURBATION
                source_tensor.data = source_tensor.data + perturbation
                optimizer = torch.optim.Adam([source_tensor], lr=LR, betas=(MOMENTUM, 0.999), weight_decay=1e-05)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_STEPS, eta_min=LR * 0.01)
                start_time = time.time()
                single_run_record = {'distance_name': dist_name, 'repeat': repeat + 1, 'base_seed': base_seed, 'proj_seed': proj_seed, 'total_time': 0.0}
                variant_step_pcs = {} if repeat == 0 else None
                for step in range(NUM_STEPS + 1):
                    optimizer.zero_grad()
                    current_dist_funcs = get_all_distance_functions(proj_seed, repeat, step, NUM_STEPS)
                    current_dist_func = current_dist_funcs[dist_name]
                    distance = current_dist_func(source_tensor, target_tensor)
                    distance.backward()
                    torch.nn.utils.clip_grad_norm_([source_tensor], max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    if step in RECORD_STEPS:
                        with torch.no_grad():
                            current_source_pc_gpu = source_tensor.detach()
                            dist_val = distance.item()
                            cd_val = chamfer_distance(current_source_pc_gpu, target_tensor)
                            fscore_val = compute_f_score(current_source_pc_gpu, target_tensor)
                            hd_val = compute_hausdorff_distance(current_source_pc_gpu, target_tensor)
                            nc_val = compute_normal_consistency(current_source_pc_gpu.cpu().numpy(), target_pc, seed=base_seed + step)
                            single_run_record[f'step_{step}_distance'] = dist_val
                            single_run_record[f'step_{step}_cd'] = cd_val
                            single_run_record[f'step_{step}_fscore'] = fscore_val
                            single_run_record[f'step_{step}_normal_consistency'] = nc_val
                            single_run_record[f'step_{step}_hausdorff'] = hd_val
                            for metric_en, val in [('distance', dist_val), ('cd', cd_val), ('fscore', fscore_val), ('normal_consistency', nc_val), ('hausdorff', hd_val)]:
                                key = f'step_{step}_{metric_en}'
                                if key not in all_results_by_dist[dist_name]:
                                    all_results_by_dist[dist_name][key] = []
                                all_results_by_dist[dist_name][key].append(val)
                            if repeat == 0 and variant_step_pcs is not None and (step in VISUAL_STEPS):
                                current_source_pc_np = current_source_pc_gpu.cpu().numpy()
                                variant_step_pcs[step] = current_source_pc_np
                if repeat == 0 and variant_step_pcs and (len(variant_step_pcs) > 0):
                    variant_idx = variant_name_to_idx[dist_name]
                    plot_variant_combined_figure(source_pc, target_pc, variant_step_pcs, dist_name, variant_idx, f'{result_dir}/snapshots', VISUAL_STEPS)
                total_time = time.time() - start_time
                single_run_record['total_time'] = total_time
                raw_results.append(single_run_record)
                time_key = 'total_time'
                if time_key not in all_results_by_dist[dist_name]:
                    all_results_by_dist[dist_name][time_key] = []
                all_results_by_dist[dist_name][time_key].append(total_time)
            except Exception as e:
                print(f'\n错误：处理 {dist_name} 时出错: {e}')
                import traceback
                traceback.print_exc()
                continue
    raw_df = pd.DataFrame(raw_results)
    raw_df.to_excel(pair_named_path(os.path.join(result_dir, 'debug'), 'raw_results.xlsx'), index=False)
    print(f'\n原始数据已保存（包含 {REPEAT_TIMES} 次重复的完整指标）')
    metrics_config = [('距离值W2', 'distance', True), ('倒角距离CD', 'cd', True), ('FScore', 'fscore', False), ('法向量一致性NC', 'normal_consistency', False), ('豪斯多夫距离HD', 'hausdorff', True), ('总耗时(秒)', 'total_time', True)]
    final_step = RECORD_STEPS[-1]
    all_p_values = []
    statistical_results = {}
    print('\n生成指标文件...')
    baseline_mapping = {'GEBSW-f^*_e-Proj-poly(q=1) (EBSW-Baseline)': 'GSW-Proj-poly(q=1) (SW-Baseline)', f'GEBSW-f^*_{{{ENERGY_ORDER_SYMBOL}=1}}-Proj-poly(q=1) (EBSW-Baseline)': 'GSW-Proj-poly(q=1) (SW-Baseline)', f'GEBSW-f^*_{{{ENERGY_ORDER_SYMBOL}=2}}-Proj-poly(q=1) (EBSW-Baseline)': 'GSW-Proj-poly(q=1) (SW-Baseline)', f'GEBSW-f^*_{{{ENERGY_ORDER_SYMBOL}=3}}-Proj-poly(q=1) (EBSW-Baseline)': 'GSW-Proj-poly(q=1) (SW-Baseline)', f'GEBSW-f^*_{{{ENERGY_ORDER_SYMBOL}=4}}-Proj-poly(q=1) (EBSW-Baseline)': 'GSW-Proj-poly(q=1) (SW-Baseline)', 'GEBSW-f^*_e-Proj-poly(q=3)': 'GSW-Proj-poly(q=3) (GSW-Baseline)', f'GEBSW-f^*_{{{ENERGY_ORDER_SYMBOL}=1}}-Proj-poly(q=3)': 'GSW-Proj-poly(q=3) (GSW-Baseline)', f'GEBSW-f^*_{{{ENERGY_ORDER_SYMBOL}=2}}-Proj-poly(q=3)': 'GSW-Proj-poly(q=3) (GSW-Baseline)', f'GEBSW-f^*_{{{ENERGY_ORDER_SYMBOL}=3}}-Proj-poly(q=3)': 'GSW-Proj-poly(q=3) (GSW-Baseline)', f'GEBSW-f^*_{{{ENERGY_ORDER_SYMBOL}=4}}-Proj-poly(q=3)': 'GSW-Proj-poly(q=3) (GSW-Baseline)', 'GEBSW-f^*_e-Proj-poly(q=5)': 'GSW-Proj-poly(q=5) (GSW-Baseline)', f'GEBSW-f^*_{{{ENERGY_ORDER_SYMBOL}=1}}-Proj-poly(q=5)': 'GSW-Proj-poly(q=5) (GSW-Baseline)', f'GEBSW-f^*_{{{ENERGY_ORDER_SYMBOL}=2}}-Proj-poly(q=5)': 'GSW-Proj-poly(q=5) (GSW-Baseline)', f'GEBSW-f^*_{{{ENERGY_ORDER_SYMBOL}=3}}-Proj-poly(q=5)': 'GSW-Proj-poly(q=5) (GSW-Baseline)', f'GEBSW-f^*_{{{ENERGY_ORDER_SYMBOL}=4}}-Proj-poly(q=5)': 'GSW-Proj-poly(q=5) (GSW-Baseline)'}
    gebsw_keys = [k for k in all_results_by_dist.keys() if 'GEBSW' in k or 'EBSW' in k]
    sw_key = 'GSW-Proj-poly(q=1) (SW-Baseline)'
    for metric_cn, metric_en, is_lower_better in metrics_config:
        print(f'\n处理指标: {metric_cn}')
        rows = []
        for dist_name in all_results_by_dist.keys():
            row = {'距离变体名称': dist_name}
            if metric_en == 'total_time':
                key = 'total_time'
                if key in all_results_by_dist[dist_name]:
                    values = all_results_by_dist[dist_name][key]
                    row['总耗时_Mean'] = round(np.mean(values), PRECISION_DIGITS)
                    row['总耗时_Std'] = round(np.std(values, ddof=1), PRECISION_DIGITS)
                    row['总耗时_Raw'] = str([round(v, PRECISION_DIGITS) for v in values])
            else:
                for step in RECORD_STEPS:
                    key = f'step_{step}_{metric_en}'
                    if key in all_results_by_dist[dist_name]:
                        values = all_results_by_dist[dist_name][key]
                        row[f'Step{step}_Mean'] = round(np.mean(values), PRECISION_DIGITS)
                        row[f'Step{step}_Std'] = round(np.std(values, ddof=1), PRECISION_DIGITS)
                        row[f'Step{step}_Raw'] = str([round(v, PRECISION_DIGITS) for v in values])
            rows.append(row)
        metric_df = pd.DataFrame(rows)
        excel_filename = f'{metric_cn}.xlsx' if metric_cn != '总耗时(秒)' else '总耗时.xlsx'
        metric_df.to_excel(pair_named_path(os.path.join(result_dir, 'metrics'), excel_filename), index=False)
        print(f"  ✓ 已保存: {pair_named_path(os.path.join(result_dir, 'metrics'), excel_filename)} ({len(rows)}行)")
    print(f"\n{'=' * 70}")
    print(f'开始统计显著性检验 (REPEAT_TIMES={REPEAT_TIMES})')
    print(f"{'=' * 70}")
    if REPEAT_TIMES >= 2:
        test_metrics = [('倒角距离CD', 'cd', True), ('FScore', 'fscore', False)]
        success_count = 0
        for metric_cn, metric_en, is_lower_better in test_metrics:
            print(f'\n【指标: {metric_cn}】')
            step_key = f'step_{final_step}_{metric_en}'
            for gebsw_key in gebsw_keys:
                print(f'\n  变体: {gebsw_key[:45]}...')
                gebsw_vals = all_results_by_dist.get(gebsw_key, {}).get(step_key, [])
                if not gebsw_vals or len(gebsw_vals) < 2:
                    print(f'    ❌ 数据不足')
                    continue
                gsw_key = baseline_mapping.get(gebsw_key)
                gsw_vals = all_results_by_dist.get(gsw_key, {}).get(step_key, []) if gsw_key else []
                sw_vals = all_results_by_dist.get(sw_key, {}).get(step_key, [])
                test_results = statistical_significance_test(gebsw_vals, gsw_vals, sw_vals)
                if test_results:
                    test_key = f'{gebsw_key}_vs_Baselines_{metric_en}'
                    statistical_results[test_key] = test_results
                    success_count += 1
                    if 'GEBSW_vs_GSW' in test_results:
                        all_p_values.append({'test_name': f'{test_key}_vs_GSW', 'metric': metric_cn, 'p_value': test_results['GEBSW_vs_GSW']['p_value'], 'details': test_results['GEBSW_vs_GSW']})
                    if 'GEBSW_vs_SW' in test_results:
                        all_p_values.append({'test_name': f'{test_key}_vs_SW', 'metric': metric_cn, 'p_value': test_results['GEBSW_vs_SW']['p_value'], 'details': test_results['GEBSW_vs_SW']})
        print(f'\n✓ 成功生成 {success_count} 个统计检验结果')
        if statistical_results:
            print(f'\n正在保存统计检验结果...')
            try:
                with pd.ExcelWriter(pair_named_path(os.path.join(result_dir, 'metrics'), '统计显著性检验_详细结果.xlsx'), engine='openpyxl') as writer:
                    for test_name, test_results in statistical_results.items():
                        rows = []
                        for comparison, values in test_results.items():
                            rows.append({'对比组': comparison, '样本量(n)': values.get('sample_size', REPEAT_TIMES), 't统计量': values.get('t_statistic'), 'p值': values.get('p_value'), '显著性(α=0.05)': '是' if values.get('significant_05') else '否', '显著性(α=0.01)': '是' if values.get('significant_01') else '否', 'Cohens_d': values.get('cohens_d'), '效应量': values.get('effect_magnitude'), '均值差异': values.get('mean_diff'), '备注': values.get('note', '')})
                        if rows:
                            df = pd.DataFrame(rows)
                            safe_name = test_name[:28] + '...' if len(test_name) > 31 else test_name
                            df.to_excel(writer, sheet_name=safe_name, index=False)
                print(f'✅ 统计检验详情已保存')
            except Exception as e:
                print(f'❌ 保存失败: {e}')
        if all_p_values:
            pvals = [item['p_value'] for item in all_p_values]
            reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
            pd.DataFrame({'检验名称': [item['test_name'] for item in all_p_values], '指标': [item['metric'] for item in all_p_values], '原始p值': pvals, 'FDR校正后p值': pvals_corrected, '显著性(FDR)': reject, 'Cohens_d': [item['details'].get('cohens_d') for item in all_p_values]}).to_excel(pair_named_path(os.path.join(result_dir, 'metrics'), '多重比较校正_FDR.xlsx'), index=False)
            print(f'✅ FDR校正已保存 ({len(pvals)}个检验)')
    else:
        print(f'跳过统计检验（REPEAT_TIMES={REPEAT_TIMES} < 2）')
    summary_rows = []
    for dist_name in all_results_by_dist.keys():
        summary_row = {'距离变体': dist_name}
        for metric_cn, metric_en, is_lower_better in metrics_config:
            if metric_en == 'total_time':
                key = 'total_time'
                if key in all_results_by_dist[dist_name]:
                    values = all_results_by_dist[dist_name][key]
                    summary_row['总耗时(秒)_Mean'] = round(np.mean(values), REPORT_DIGITS)
                    summary_row['总耗时(秒)_Std'] = round(np.std(values, ddof=1), REPORT_DIGITS)
            else:
                key = f'step_{final_step}_{metric_en}'
                if key in all_results_by_dist[dist_name]:
                    values = all_results_by_dist[dist_name][key]
                    summary_row[f'{metric_cn}_Mean'] = round(np.mean(values), REPORT_DIGITS)
                    summary_row[f'{metric_cn}_Std'] = round(np.std(values, ddof=1), REPORT_DIGITS)
        summary_rows.append(summary_row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_excel(pair_named_path(os.path.join(result_dir, 'metrics'), f'汇总_最终步骤_Step{final_step}_Complete.xlsx'), index=False)
    print(f'  ✓ 汇总表已保存')
    print('\n绘制收敛曲线...')
    plot_convergence_curves(all_results_by_dist, list(all_results_by_dist.keys()), [(m[0], m[1]) for m in metrics_config if m[1] != 'total_time'])
    print(f"\n{'=' * 60}")
    print('主实验完成！')
    print(f'共处理 {len(all_results_by_dist)} 种距离变体')
    print(f'结果目录: {result_dir}/')
    print(f"{'=' * 60}")
    return (source_pc, target_pc, all_results_by_dist)

def _configure_worker_gpu(gpu_id=None):
    global device
    if gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(int(gpu_id))
        device = torch.device(f'cuda:{int(gpu_id)}')
        print(f'[worker pid={os.getpid()}] 使用 GPU {gpu_id}: {torch.cuda.get_device_name(int(gpu_id))}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[worker pid={os.getpid()}] 使用设备: {device}')

def _run_shapenet_pair(test_idx, pair_tuple, all_pcs, gpu_id=None):
    global CURRENT_SOURCE_PC, CURRENT_TARGET_PC
    global source_path, target_path, result_dir
    global CURRENT_PAIR_INDEX, CURRENT_PAIR_GROUP, CURRENT_SOURCE_NAME, CURRENT_TARGET_NAME
    _configure_worker_gpu(gpu_id)
    src_idx, tgt_idx, src_lbl, tgt_lbl = pair_tuple
    pair_type = f'c{int(src_lbl)}_to_c{int(tgt_lbl)}'
    group = 'CrossCategory' if src_lbl != tgt_lbl else 'SameCategory'
    CURRENT_SOURCE_PC = all_pcs[int(src_idx)]
    CURRENT_TARGET_PC = all_pcs[int(tgt_idx)]
    source_path = f'ShapeNet_index_{int(src_idx)}_class_{int(src_lbl)}'
    target_path = f'ShapeNet_index_{int(tgt_idx)}_class_{int(tgt_lbl)}'
    result_dir = f'PCR-ShapeNet-pair{test_idx:03d}-{pair_type}'
    CURRENT_PAIR_INDEX = test_idx
    CURRENT_PAIR_GROUP = group
    CURRENT_SOURCE_NAME = f'pair{test_idx:03d}-c{int(src_lbl)}'
    CURRENT_TARGET_NAME = f'c{int(tgt_lbl)}'
    print(f"\n{'=' * 70}")
    print(f"【GPU {(gpu_id if gpu_id is not None else 'single')}】处理 ShapeNet Pair {test_idx}: {pair_type}")
    print(f'source index={src_idx}, label={src_lbl}; target index={tgt_idx}, label={tgt_lbl}')
    print(f'结果目录: {result_dir}')
    print(f'输出文件前缀: {get_pair_name_tag()}')
    print(f"{'-' * 70}\n")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(f'{result_dir}/snapshots', exist_ok=True)
    os.makedirs(f'{result_dir}/metrics', exist_ok=True)
    os.makedirs(f'{result_dir}/curves', exist_ok=True)
    os.makedirs(f'{result_dir}/debug', exist_ok=True)
    os.makedirs(f'{result_dir}/sensitivity', exist_ok=True)
    try:
        source_pc, target_pc, all_results = point_cloud_reconstruction_experiment_final()
        if RUN_SENSITIVITY_ANALYSIS:
            print(f"\n{'=' * 60}")
            print('开始敏感性分析...')
            hyperparameter_sensitivity_analysis(source_pc, target_pc)
        print(f'\n✅ ShapeNet Pair {test_idx} ({pair_type}) 实验完成！')
        return (True, test_idx, src_idx, tgt_idx, pair_type, '')
    except Exception as e:
        print(f'\n❌ 错误：处理 ShapeNet Pair {test_idx} ({pair_type}) 时发生异常: {e}')
        import traceback
        traceback.print_exc()
        return (False, test_idx, src_idx, tgt_idx, pair_type, str(e))

def _run_shapenet_pair_chunk(indexed_pair_chunk, gpu_id):
    _configure_worker_gpu(gpu_id)
    print(f'[GPU {gpu_id}] 正在加载 ShapeNet Part Segmentation 数据...')
    all_pcs, _all_labels = load_shapenet_hdf5_all(SHAPENET_HDF5_ROOT)
    results = []
    for test_idx, pair_tuple in indexed_pair_chunk:
        results.append(_run_shapenet_pair(int(test_idx), pair_tuple, all_pcs, gpu_id=gpu_id))
    return results

def _select_gpu_ids():
    if not torch.cuda.is_available():
        return []
    visible_count = torch.cuda.device_count()
    if MULTI_GPU_IDS is None:
        return list(range(visible_count))
    return [int(g) for g in MULTI_GPU_IDS if int(g) < visible_count]
if __name__ == '__main__':
    print('正在加载 ShapeNet Part Segmentation 数据以生成固定配对...')
    all_pcs_parent, all_labels_parent = load_shapenet_hdf5_all(SHAPENET_HDF5_ROOT)
    pair_list = generate_shapenet_pairs(all_pcs_parent, all_labels_parent, num_pairs=NUM_PAIRS, seed=PAIR_SEED, cross_category=PAIR_CROSS_CATEGORY, balanced_by_category=PAIR_BALANCED_BY_CATEGORY)
    pair_mode_name = '跨类别' if PAIR_CROSS_CATEGORY else '同类别'
    balance_name = '类别均衡固定配对' if PAIR_BALANCED_BY_CATEGORY else '全局随机配对'
    print(f'已生成 {len(pair_list)} 组{pair_mode_name}配对（{balance_name}, seed={PAIR_SEED}）')
    print_pair_balance_report(pair_list)
    save_pair_metadata(pair_list, PAIR_METADATA_FILENAME)
    if BATCH_MODE:
        indexed_pair_list = list(enumerate(pair_list))
        print(f"\n{'=' * 70}")
        print(f'【ShapeNet 批量运行模式】共 {len(indexed_pair_list)} 组实验待执行')
        print(f'能量函数阶数符号: {ENERGY_ORDER_SYMBOL}')
        print(f"外部 baseline: {('开启' if INCLUDE_EXTERNAL_BASELINES else '关闭')}")
        print(f"{'=' * 70}\n")
    else:
        if TEST_INDEX < 0 or TEST_INDEX >= len(pair_list):
            raise ValueError(f'TEST_INDEX={TEST_INDEX} 越界，可选范围为 0 到 {len(pair_list) - 1}')
        indexed_pair_list = [(TEST_INDEX, pair_list[TEST_INDEX])]
        print(f"\n{'=' * 70}")
        print(f'【ShapeNet 单组运行模式】只运行 Pair {TEST_INDEX}')
        print(f"{'=' * 70}")
    total_groups = len(indexed_pair_list)
    success_groups, failed_groups = ([], [])
    gpu_ids = _select_gpu_ids()
    use_multi = BATCH_MODE and USE_MULTI_GPU and (len(gpu_ids) >= 2) and (total_groups >= MULTI_GPU_MIN_TASKS)
    if use_multi:
        print(f'[Multi-GPU] 启用 pair 级并行，GPU 列表: {gpu_ids}')
        chunks = [[] for _ in gpu_ids]
        for i, item in enumerate(indexed_pair_list):
            chunks[i % len(gpu_ids)].append(item)
        for gpu, chunk in zip(gpu_ids, chunks):
            print(f'  GPU {gpu} -> pair indices: {[idx for idx, _ in chunk]}')
        del all_pcs_parent, all_labels_parent
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=len(gpu_ids)) as pool:
            nested = pool.starmap(_run_shapenet_pair_chunk, [(chunk, gpu) for gpu, chunk in zip(gpu_ids, chunks) if chunk])
        flat_results = [item for sub in nested for item in sub]
    else:
        print('[Single-GPU] 未启用多 GPU：可能只有一张 GPU、BATCH_MODE=False，或任务数过少。')
        gpu0 = gpu_ids[0] if gpu_ids else None
        flat_results = []
        for test_idx, pair_tuple in indexed_pair_list:
            flat_results.append(_run_shapenet_pair(int(test_idx), pair_tuple, all_pcs_parent, gpu_id=gpu0))
    for ok, idx, src_idx, tgt_idx, pair_type, err in flat_results:
        if ok:
            success_groups.append((idx, src_idx, tgt_idx, pair_type))
        else:
            failed_groups.append((idx, src_idx, tgt_idx, pair_type, err))
    print(f"\n{'=' * 70}")
    print('【ShapeNet 批量运行完成总结】' if BATCH_MODE else '【ShapeNet 单组运行完成总结】')
    print(f'总组数: {total_groups}')
    print(f'成功: {len(success_groups)} 组')
    print(f'失败: {len(failed_groups)} 组')
    if success_groups:
        print('\n成功完成的组:')
        for idx, src_idx, tgt_idx, pair_type in sorted(success_groups):
            print(f'  [{idx}] {pair_type}: src_idx={src_idx} → tgt_idx={tgt_idx}')
    if failed_groups:
        print('\n失败的组:')
        for idx, src_idx, tgt_idx, pair_type, err in sorted(failed_groups):
            print(f'  [{idx}] {pair_type}: src_idx={src_idx} → tgt_idx={tgt_idx} | 错误: {err[:80]}...')
    try:
        aggregate_framework_analysis_from_dirs(base_dir='.')
    except Exception as e:
        print(f'[Warning] 汇总 ShapeNet framework analysis 失败: {e}')
    print(f"\n{'=' * 70}")
    print('所有 ShapeNet 实验执行完毕！各组结果保存在独立目录中。')
    print(f'能量函数阶数符号: {ENERGY_ORDER_SYMBOL}')
    print(f"{'=' * 70}\n")
