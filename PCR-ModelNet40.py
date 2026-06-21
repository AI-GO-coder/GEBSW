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

warnings.filterwarnings('ignore')

# ======================== 配置参数 ========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

# ======================== 符号配置（可自定义）========================
# 能量函数阶数符号：可统一替换为 "r", "p", "alpha", "k", "m", "beta", "n" 等任意字符串
# 修改此变量即可全局统一替换所有能量函数阶数的符号表示（函数名、报告、图表中的显示）
ENERGY_ORDER_SYMBOL = "r"  # 默认使用 "r"，可改为 "p", "q", "alpha", "k", "s", "t" 等

# 核心参数
NUM_STEPS = 300
TARGET_SAMPLE_SIZE = 6144
LR = 5e-3
MOMENTUM = 0.9
REPEAT_TIMES = 10
L_BASE = 50
P = 2
EPS = 1e-12
UNIFIED_MAX_DIM = 256  # 统一projection输出容量，避免q=1/3/5因max_dim不同造成容量混杂
INITIAL_PERTURBATION = 1e-2
PRECISION_DIGITS = 10
REPORT_DIGITS = 4

# ======================== 【新增】外部 sliced-OT baseline 配置 ========================
# 主实验默认同时运行：18 个 GEBSW-family 距离 + 2 个外部 sliced-OT baseline。
# 注意：这里的 Max-SW 是 sampled-MaxSW，即在同样 L_BASE 个随机投影方向中取最大值；
# 若不做连续方向内层优化，论文中建议报告为 Sampled-MaxSW。
INCLUDE_EXTERNAL_BASELINES = True

MAXSW_USE_INNER_OPT = False
MAXSW_INNER_STEPS = 5
MAXSW_INNER_LR = 5e-2

# DSW 使用与 GEBSW 相同的 projection/particle budget，保证对比公平。
DSW_NUM_PROJECTIONS = L_BASE
DSW_INNER_STEPS = 3
DSW_INNER_LR = 5e-2
DSW_DIVERSITY_WEIGHT = 0.05
DSW_WEIGHT_TEMPERATURE = 0.25

# 温度退火参数
TEMPERATURE_START = 2.0
TEMPERATURE_END = 0.1
USE_TEMPERATURE_ANNEALING = True
USE_SOFTMAX_WEIGHT = True
WEIGHT_CLIP_MAX = 0.3

# 敏感性分析配置
RUN_SENSITIVITY_ANALYSIS = False  # 统一框架主实验默认关闭温度敏感性，避免与设计空间分析混杂
TEMPERATURE_RANGE = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
SENSITIVITY_REPEAT_TIMES = 3

# ======================== 【新增】运行模式配置 ========================
BATCH_MODE = True  # True: 批量运行15组；统一框架叙事建议运行全部pair
# BATCH_MODE = False
TEST_INDEX = 0  # 当BATCH_MODE=False时，指定运行哪一组(0-14)

# ======================== 【新增】双 GPU / 多 GPU pair 级并行配置 ========================
# 说明：本实验每个 pair、每个距离变体都是相对独立的优化任务，
# 因此最适合采用 pair-level multiprocessing，而不是 DataParallel。
# 默认使用 GPU0 和 GPU1；如果机器只有一张 GPU 或 BATCH_MODE=False，会自动退回单 GPU/单进程。
USE_MULTI_GPU = True
MULTI_GPU_IDS = [0, 1]      # 两块 GPU；如需自动使用全部可见 GPU，可设为 None
MULTI_GPU_MIN_TASKS = 2


# ======================== 【新增】组合图颜色配置 ========================
# 18种变体的专属颜色（区分度高，不与源/目标的viridis重复）
VARIANT_COLORS = [
    '#E6194B',  # 红
    '#3CB44B',  # 绿
    '#FFE119',  # 黄
    '#4363D8',  # 蓝
    '#F58231',  # 橙
    '#911EB4',  # 紫
    '#46F0F0',  # 青
    '#F032E6',  # 品红
    '#BCF60C',  # 荧光绿
    '#FABEBE',  # 粉
    '#008080',  # 深青
    '#E6BEFF',  # 薰衣草
    '#9A6324',  # 棕
    '#FF0033',  # 米黄
    '#800000',  # 深红
    '#AAFFC3',  # 薄荷绿
    '#808000',  # 橄榄
    '#FFD8B1',  # 桃色
]

# 源点云和目标点云使用统一颜色方案（立体感强的 viridis）
SOURCE_TARGET_CMAP = 'viridis'

# ======================== 15对点云配对配置 ========================
BASE_PATH = "/home/zhouyan/EBGSW/ModelNet40-Examples/"

PAIR_CONFIGS = [
    # 简单组 (0-4)
    ("chair_0986.off", "table_0407.off", "Simple"),  # 0
    ("bed_0519.off", "sofa_0763.off", "Simple"),  # 1
    ("desk_0251.off", "night_stand_0212.off", "Simple"),  # 2
    ("monitor_0530.off", "sink_0147.off", "Simple"),  # 3
    ("bathtub_0154.off", "piano_0277.off", "Simple"),  # 4
    # 中等组 (5-9)
    ("airplane_0661.off", "car_0236.off", "Medium"),  # 5
    ("lamp_0126.off", "vase_0493.off", "Medium"),  # 6
    ("bottle_0348.off", "cup_0086.off", "Medium"),  # 7
    ("glass_box_0259.off", "bench_0181.off", "Medium"),  # 8
    ("guitar_0208.off", "laptop_0157.off", "Medium"),  # 9
    # 复杂组 (10-14)
    ("plant_0243.off", "person_0094.off", "Complex"),  # 10
    ("flower_pot_0151.off", "stool_0091.off", "Complex"),  # 11
    ("stairs_0135.off", "toilet_0363.off", "Complex"),  # 12
    ("cone_0168.off", "range_hood_0181.off", "Complex"),  # 13
    ("bowl_0079.off", "airplane_0661.off", "Complex"),  # 14
]

# 点云路径（将在主循环中动态设置）
source_path = ""
target_path = ""

# 数值结果与收敛曲线使用的记录步数：用于 distance/CD/FScore/NC/HD/Runtime 表格与 convergence curves。
RECORD_STEPS = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]

# 点云可视化使用的记录步数：仅用于 snapshot / combined point-cloud visualization。
VISUAL_STEPS = [20, 40, 100, 200, 300]

result_dir = "PCR-results"

# 当前pair元信息：由main循环动态写入，用于统一框架分析表
CURRENT_PAIR_INDEX = None
CURRENT_PAIR_GROUP = None
CURRENT_SOURCE_NAME = None
CURRENT_TARGET_NAME = None


def get_pair_name_tag():
    """返回当前点云组标签，用于所有单组输出文件名前缀。

    格式：源点云名称-目标点云名称，例如 chair_0986-table_0407。
    若当前pair元信息尚未初始化，则返回 ModelNet40_pair_unknown，避免空文件名。
    """
    if CURRENT_SOURCE_NAME and CURRENT_TARGET_NAME:
        return f"{CURRENT_SOURCE_NAME}-{CURRENT_TARGET_NAME}"
    return "ModelNet40_pair_unknown"


def pair_named_filename(filename):
    """给单组输出文件名加上“源点云-目标点云”前缀。"""
    return f"{get_pair_name_tag()}_{filename}"


def pair_named_path(directory, filename):
    """构造带pair前缀的输出路径。"""
    return os.path.join(directory, pair_named_filename(filename))

F_SCORE_THRESHOLD = 0.01


def _stable_int_seed(*items, modulo=2**32 - 1):
    """生成跨进程、跨运行稳定的整数 seed，用于点云采样和可复现评价。"""
    text = "::".join(str(x) for x in items)
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % modulo


# ======================== 【修复】健壮的工具函数 ========================
def read_off_file_robust(file_path, target_size=TARGET_SAMPLE_SIZE, seed=None):
    """
    【修正版】使用 trimesh 从表面均匀采样，并显式固定随机种子。
    - 对于 Mesh 模型：使用 mesh.sample(target_size) 从表面均匀采样；
    - 对于 PointCloud 模型：使用顶点并调整数量；
    - 每个文件的采样 seed 由 file_path 和 target_size 稳定生成，因此跨运行可复现。

    注意：该函数会临时设置 NumPy RNG 状态，采样完成后恢复原 RNG 状态，
    避免影响后续优化实验的随机种子控制。
    """
    if seed is None:
        seed = _stable_int_seed("read_off_file_robust", os.path.abspath(file_path), target_size)

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
        raise ValueError(f"读取文件失败 {file_path}: {str(e)}")
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
    """
    多项式投影映射。

    公平性修复：当维度超过 max_dim 需要随机降维时，使用显式 seed 生成投影矩阵，
    避免不同方法或不同 step 因隐式 torch.randn 而引入额外随机性。
    对 3D 点云的 q=1/3/5 情况通常不会触发随机降维，但保留该修复以保证通用性。
    """
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
                    cross_term = (X[:, i] ** (d // 2)) * (X[:, j] ** (d - d // 2))
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


# ======================== 【新增】外部 sliced-OT baseline：Sampled-MaxSW / DSW ========================

def _stable_seed(base_seed, name, offset=0):
    """根据基准 seed 和方法名生成稳定整数 seed，避免 Python hash 随进程变化。"""
    h = hashlib.md5(f"{int(base_seed)}::{name}::{int(offset)}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def max_sliced_wasserstein_loss(X, Y, num_projections=L_BASE, p=P, seed=None,
                                use_inner_opt=MAXSW_USE_INNER_OPT):
    """
    Sampled-MaxSW baseline.

    默认实现为 sampled-MaxSW：在与 GEBSW 相同的 L_BASE 个随机方向中
    选择 1D Wasserstein 最大的方向。若 MAXSW_USE_INNER_OPT=True，
    会从 sampled max 方向出发做少量内层梯度上升，但默认关闭以控制计算量。
    """
    theta = rand_projections(X.shape[1], num_projections=num_projections, seed=seed)
    X_proj = torch.matmul(X, theta.T)
    Y_proj = torch.matmul(Y, theta.T)
    wd_tensor = torch.stack([
        one_dimensional_wasserstein(X_proj[:, i], Y_proj[:, i], p=p)
        for i in range(num_projections)
    ])
    max_idx = torch.argmax(wd_tensor)
    if not use_inner_opt:
        return wd_tensor[max_idx]

    with torch.enable_grad():
        theta_best = theta[max_idx].detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([theta_best], lr=MAXSW_INNER_LR)
        X_det, Y_det = X.detach(), Y.detach()
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
    """投影方向多样性惩罚；值越大表示方向越相似。"""
    theta = F.normalize(theta, p=2, dim=1)
    gram = torch.matmul(theta, theta.T).abs()
    eye = torch.eye(theta.shape[0], device=theta.device, dtype=theta.dtype)
    off_diag = gram * (1.0 - eye)
    return off_diag.mean()


def distributional_sliced_wasserstein_loss(X, Y, num_projections=DSW_NUM_PROJECTIONS,
                                           p=P, seed=None):
    """
    DSW / Distributional Sliced-Wasserstein baseline 的实用可微近似。

    内层在 detached X/Y 上优化一组 projection particles，使其偏向高 Wasserstein 方向，
    同时加入方向多样性正则；外层用优化后的方向和 distribution weights 对原始 X 求梯度。
    """
    init_seed = _stable_seed(seed if seed is not None else 0, "dsw_theta")
    theta = rand_projections(X.shape[1], num_projections=num_projections, seed=init_seed).detach().clone()
    theta.requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=DSW_INNER_LR)
    X_det, Y_det = X.detach(), Y.detach()

    for _ in range(DSW_INNER_STEPS):
        opt.zero_grad()
        theta_norm = F.normalize(theta, p=2, dim=1)
        X_proj = torch.matmul(X_det, theta_norm.T)
        Y_proj = torch.matmul(Y_det, theta_norm.T)
        wd_tensor = torch.stack([
            one_dimensional_wasserstein(X_proj[:, i], Y_proj[:, i], p=p)
            for i in range(num_projections)
        ])
        weights = F.softmax(wd_tensor / DSW_WEIGHT_TEMPERATURE, dim=0)
        objective = torch.sum(weights * wd_tensor) - DSW_DIVERSITY_WEIGHT * _direction_diversity_penalty(theta_norm)
        (-objective).backward()
        opt.step()

    theta_final = F.normalize(theta.detach(), p=2, dim=1)
    X_proj = torch.matmul(X, theta_final.T)
    Y_proj = torch.matmul(Y, theta_final.T)
    wd_tensor = torch.stack([
        one_dimensional_wasserstein(X_proj[:, i], Y_proj[:, i], p=p)
        for i in range(num_projections)
    ])
    weights = F.softmax(wd_tensor.detach() / DSW_WEIGHT_TEMPERATURE, dim=0)
    return torch.sum(weights * wd_tensor)


def get_external_baseline_functions(proj_seed, repeat_idx, current_step=0, total_steps=NUM_STEPS):
    """返回外部 sliced-OT baselines：Sampled-MaxSW 与 DSW。"""
    distance_functions = {}

    def sampled_maxsw_baseline(X, Y):
        return max_sliced_wasserstein_loss(
            X, Y, num_projections=L_BASE, p=P,
            seed=_stable_seed(proj_seed, "Sampled-MaxSW"),
            use_inner_opt=MAXSW_USE_INNER_OPT,
        )
    distance_functions["Sampled-MaxSW (External-Baseline)"] = sampled_maxsw_baseline

    def dsw_baseline(X, Y):
        return distributional_sliced_wasserstein_loss(
            X, Y, num_projections=DSW_NUM_PROJECTIONS, p=P,
            seed=_stable_seed(proj_seed, "DSW"),
        )
    distance_functions["DSW (External-Baseline)"] = dsw_baseline

    return distance_functions


def get_all_distance_functions(proj_seed, repeat_idx, current_step=0, total_steps=NUM_STEPS,
                               temp_start=TEMPERATURE_START, temp_end=TEMPERATURE_END):
    """18 个 GEBSW-family 距离 + 可选外部 sliced-OT baselines。"""
    distance_functions = get_distance_functions_corrected(
        proj_seed, repeat_idx, current_step, total_steps, temp_start, temp_end
    )
    if INCLUDE_EXTERNAL_BASELINES:
        distance_functions.update(get_external_baseline_functions(proj_seed, repeat_idx, current_step, total_steps))
    return distance_functions


# ======================== 【关键优化】GPU加速的指标计算 ========================

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
    """
    计算法向量一致性。

    公平性修复：使用显式 seed 控制子采样，保证评价指标跨运行可复现。
    """
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
            normal = normal / (np.linalg.norm(normal) + 1e-8)
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


# ======================== 可视化函数（保持CPU）========================

def plot_point_cloud_snapshot(pc, step, distance_name, repeat_idx=0, save_subdir="snapshots",
                              cmap='viridis', point_size=8):
    """
    【修复版】视觉友好型点云快照 - 简洁无坐标轴版本（仅第1次重复）
    """
    try:
        safe_name = distance_name.replace('(', '_').replace(')', '_').replace('=', '_').replace('*', 'star').replace(
            '-', '_').replace(' ', '_')

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

        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]

        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-10)
        ax.scatter(x, y, z,
                   c=z_norm,
                   cmap=cmap,
                   s=point_size,
                   alpha=0.9,
                   depthshade=True,
                   edgecolors='none')

        ax.view_init(elev=30, azim=45)

        max_range = np.max(np.ptp(pc, axis=0)) / 2
        mid_x, mid_y, mid_z = np.mean(pc[:, 0]), np.mean(pc[:, 1]), np.mean(pc[:, 2])
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        save_path = os.path.join(result_dir, save_subdir, pair_named_filename(f"{safe_name}_rep{repeat_idx}_step_{step}.png"))
        plt.tight_layout(pad=0)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.1)
        plt.close()
        print(f"✓ 已保存: {os.path.basename(save_path)}")

    except Exception as e:
        print(f"警告：保存快照失败 {distance_name} step {step}: {e}")
        plt.close()


def plot_variant_combined_figure(source_pc, target_pc, step_pcs, variant_name, variant_idx,
                                 save_dir, record_steps):
    """
    绘制横向组合图：源 | step1 | step2 | ... | stepN | 目标
    无标题、无坐标轴、无间隔，适合论文插入
    """
    try:
        safe_name = variant_name.replace('(', '').replace(')', '').replace('=', '_').replace('*', 'star').replace(
            '-', '_').replace(' ', '_').replace('{', '').replace('}', '').replace('^', '')

        n_steps = len(record_steps)
        fig = plt.figure(figsize=(2.0 * (2 + n_steps), 2.0), dpi=200)

        # 【关键修复】使用 projection='3d' 创建 3D 子图
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

            x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]

            if is_variant:
                ax.scatter(x, y, z, c=variant_color, s=8,
                           alpha=0.85, depthshade=True, edgecolors='none')
            else:
                z_norm = (z - z.min()) / (z.max() - z.min() + 1e-10)
                ax.scatter(x, y, z, c=z_norm, cmap=SOURCE_TARGET_CMAP, s=8,
                           alpha=0.9, depthshade=True, edgecolors='none')

            ax.view_init(elev=30, azim=45)

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

        save_path = pair_named_path(save_dir, f"combined_{safe_name}.png")
        plt.savefig(save_path, dpi=200, bbox_inches=None, pad_inches=0.02,
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f"  ✓ 组合图已保存: {os.path.basename(save_path)}")

    except Exception as e:
        print(f"  ✗ 保存组合图失败 {variant_name}: {e}")
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

    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]

    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-10)

    ax.scatter(x, y, z, c=z_norm, cmap=cmap, s=point_size,
               alpha=0.9, depthshade=False, edgecolors='none')

    ax.set_title(title, fontsize=14, pad=20)

    ax.view_init(elev=30, azim=45)

    max_range = np.max(np.ptp(pc, axis=0)) / 2
    mid = np.mean(pc, axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close()
    print(f"  ✓ 已保存: {os.path.basename(save_path)}")


def visualize_comparison(source_pc, target_pc, save_dir):
    fig = plt.figure(figsize=(16, 7))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_axis_off()
    ax1.grid(False)
    for axis in [ax1.xaxis, ax1.yaxis, ax1.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor('none')

    sx, sy, sz = source_pc[:, 0], source_pc[:, 1], source_pc[:, 2]
    z_norm_s = (sz - sz.min()) / (sz.max() - sz.min() + 1e-10)
    ax1.scatter(sx, sy, sz, c=z_norm_s, cmap='viridis', s=8, alpha=0.9, edgecolors='none')
    ax1.set_title(f'Source', fontsize=14, pad=20)
    ax1.view_init(elev=30, azim=45)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_axis_off()
    ax2.grid(False)
    for axis in [ax2.xaxis, ax2.yaxis, ax2.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor('none')

    tx, ty, tz = target_pc[:, 0], target_pc[:, 1], target_pc[:, 2]
    z_norm_t = (tz - tz.min()) / (tz.max() - tz.min() + 1e-10)
    ax2.scatter(tx, ty, tz, c=z_norm_t, cmap='viridis', s=8, alpha=0.9, edgecolors='none')
    ax2.set_title(f'Target', fontsize=14, pad=20)
    ax2.view_init(elev=30, azim=45)

    for ax in [ax1, ax2]:
        max_range = max(np.max(np.ptp(source_pc, axis=0)),
                        np.max(np.ptp(target_pc, axis=0))) / 2
        mid = np.mean(np.concatenate([source_pc, target_pc], axis=0), axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout(pad=0)
    save_path = pair_named_path(save_dir, "00_original_pair.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"✓ 对比图已保存: {save_path}")


# ======================== 统计检验功能 ========================

def compute_cohens_d(values1, values2):
    n1, n2 = len(values1), len(values2)
    if n1 == 0 or n2 == 0:
        return None

    mean1, mean2 = np.mean(values1), np.mean(values2)
    std1, std2 = np.std(values1, ddof=1), np.std(values2, ddof=1)

    pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    cohens_d = (mean1 - mean2) / pooled_std

    if abs(cohens_d) < 0.2:
        magnitude = "可忽略(Negligible)"
    elif abs(cohens_d) < 0.5:
        magnitude = "小(Small)"
    elif abs(cohens_d) < 0.8:
        magnitude = "中(Medium)"
    else:
        magnitude = "大(Large)"

    return {
        'cohens_d': float(cohens_d),
        'magnitude': magnitude,
        'mean_diff': float(mean1 - mean2),
        'pooled_std': float(pooled_std)
    }


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
            results['GEBSW_vs_GSW'] = {
                't_statistic': float(t_stat), 'p_value': float(p_value),
                'significant_05': p_value < 0.05, 'significant_01': p_value < 0.01,
                'cohens_d': cohens_d_result['cohens_d'] if cohens_d_result else None,
                'effect_magnitude': cohens_d_result['magnitude'] if cohens_d_result else None,
                'mean_diff': cohens_d_result['mean_diff'] if cohens_d_result else None,
                'sample_size': effective_len
            }
        except Exception as e:
            print(f"检验失败: {e}")

    if len(sw_arr) >= 2:
        try:
            effective_len = min(len(gebsw_arr), len(sw_arr))
            s_effective = gebsw_arr[:effective_len]
            sw_effective = sw_arr[:effective_len]
            t_stat, p_value = stats.ttest_rel(s_effective, sw_effective)
            cohens_d_result = compute_cohens_d(s_effective, sw_effective)
            results['GEBSW_vs_SW'] = {
                't_statistic': float(t_stat), 'p_value': float(p_value),
                'significant_05': p_value < 0.05, 'significant_01': p_value < 0.01,
                'cohens_d': cohens_d_result['cohens_d'] if cohens_d_result else None,
                'effect_magnitude': cohens_d_result['magnitude'] if cohens_d_result else None,
                'mean_diff': cohens_d_result['mean_diff'] if cohens_d_result else None,
                'sample_size': effective_len
            }
        except Exception as e:
            print(f"检验失败: {e}")
    return results


# ======================== 距离函数定义（全部保留）========================

def get_distance_functions_corrected(proj_seed, repeat_idx, current_step=0, total_steps=NUM_STEPS,
                                     temp_start=TEMPERATURE_START, temp_end=TEMPERATURE_END):
    """
    统一版 18 个 GEBSW-family 距离函数集合。

    公平性修复：
    1) 同一 projection order q 下的所有 variants 共享同一组投影方向 theta_q；
       例如 q=3 时，GEBW(C,3)、GEBW(e,3)、GEBW(1,3)...GEBW(4,3)
       都使用完全相同的 theta_q3。
    2) 多项式投影中若触发随机降维，也使用按 q 固定的 projection seed。
    3) 温度退火只随 step 变化，不引入额外随机性。
    """
    distance_functions = {}
    step_ratio = current_step / total_steps

    def compute_energy_weights(wd_tensor, energy_type, energy_r):
        wd_tensor = wd_tensor.clamp(min=EPS)
        if USE_TEMPERATURE_ANNEALING:
            current_temp = temp_start * (temp_end / temp_start) ** step_ratio
        else:
            current_temp = 1.0

        if energy_type == "exp":
            logits = wd_tensor / (wd_tensor.mean() + EPS) / current_temp
            weights = F.softmax(logits, dim=0)
        elif energy_type == "poly":
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
        # 同一 q/degree 共享同一个 theta seed；不同 q 使用不同 seed。
        return int(proj_seed) + int(degree) * 200003

    def _project(X, degree):
        return polynomial_projection_controlled(
            X, degree=degree, max_dim=UNIFIED_MAX_DIM, seed=_projection_seed(degree)
        )

    def _shared_projected_wd_tensor(X, Y, degree):
        X_proj = _project(X, degree)
        Y_proj = _project(Y, degree)
        theta = rand_projections(X_proj.shape[1], num_projections=L_BASE, seed=_theta_seed(degree))
        X_proj_1d = torch.matmul(X_proj, theta.T)
        Y_proj_1d = torch.matmul(Y_proj, theta.T)
        return torch.stack([
            one_dimensional_wasserstein(X_proj_1d[:, i], Y_proj_1d[:, i])
            for i in range(L_BASE)
        ])

    def _uniform_sw(X, Y, degree):
        wd_tensor = _shared_projected_wd_tensor(X, Y, degree)
        return torch.mean(wd_tensor)

    def _weighted_sw(X, Y, degree, energy_type, energy_r):
        wd_tensor = _shared_projected_wd_tensor(X, Y, degree)
        weights = compute_energy_weights(wd_tensor, energy_type, energy_r)
        return torch.sum(wd_tensor * weights)

    # q=1: SW / EBSW special cases. All q=1 variants share theta_q1.
    def gsw_poly_q1(X, Y):
        return _uniform_sw(X, Y, degree=1)
    distance_functions["GEBW(C,1) (SW-Baseline)"] = gsw_poly_q1

    def gebsw_exp_poly_q1(X, Y):
        return _weighted_sw(X, Y, degree=1, energy_type="exp", energy_r=1)
    distance_functions["GEBW(e,1) (EBSW-Baseline)"] = gebsw_exp_poly_q1

    def gebsw_poly_r_1_poly_q1(X, Y):
        return _weighted_sw(X, Y, degree=1, energy_type="poly", energy_r=1)
    distance_functions["GEBW(1,1) (EBSW-Baseline)"] = gebsw_poly_r_1_poly_q1

    def gebsw_poly_r_2_poly_q1(X, Y):
        return _weighted_sw(X, Y, degree=1, energy_type="poly", energy_r=2)
    distance_functions["GEBW(2,1) (EBSW-Baseline)"] = gebsw_poly_r_2_poly_q1

    def gebsw_poly_r_3_poly_q1(X, Y):
        return _weighted_sw(X, Y, degree=1, energy_type="poly", energy_r=3)
    distance_functions["GEBW(3,1) (EBSW-Baseline)"] = gebsw_poly_r_3_poly_q1

    def gebsw_poly_r_4_poly_q1(X, Y):
        return _weighted_sw(X, Y, degree=1, energy_type="poly", energy_r=4)
    distance_functions["GEBW(4,1) (EBSW-Baseline)"] = gebsw_poly_r_4_poly_q1

    # q=3: GSW / GEBSW variants. All q=3 variants share theta_q3.
    def gsw_poly_q3(X, Y):
        return _uniform_sw(X, Y, degree=3)
    distance_functions["GEBW(C,3) (GSW-Baseline)"] = gsw_poly_q3

    def gebsw_exp_poly_q3(X, Y):
        return _weighted_sw(X, Y, degree=3, energy_type="exp", energy_r=1)
    distance_functions["GEBW(e,3)"] = gebsw_exp_poly_q3

    def gebsw_poly_r_1_poly_q3(X, Y):
        return _weighted_sw(X, Y, degree=3, energy_type="poly", energy_r=1)
    distance_functions["GEBW(1,3)"] = gebsw_poly_r_1_poly_q3

    def gebsw_poly_r_2_poly_q3(X, Y):
        return _weighted_sw(X, Y, degree=3, energy_type="poly", energy_r=2)
    distance_functions["GEBW(2,3)"] = gebsw_poly_r_2_poly_q3

    def gebsw_poly_r_3_poly_q3(X, Y):
        return _weighted_sw(X, Y, degree=3, energy_type="poly", energy_r=3)
    distance_functions["GEBW(3,3)"] = gebsw_poly_r_3_poly_q3

    def gebsw_poly_r_4_poly_q3(X, Y):
        return _weighted_sw(X, Y, degree=3, energy_type="poly", energy_r=4)
    distance_functions["GEBW(4,3)"] = gebsw_poly_r_4_poly_q3

    # q=5: GSW / GEBSW variants. All q=5 variants share theta_q5.
    def gsw_poly_q5(X, Y):
        return _uniform_sw(X, Y, degree=5)
    distance_functions["GEBW(C,5) (GSW-Baseline)"] = gsw_poly_q5

    def gebsw_exp_poly_q5(X, Y):
        return _weighted_sw(X, Y, degree=5, energy_type="exp", energy_r=1)
    distance_functions["GEBW(e,5)"] = gebsw_exp_poly_q5

    def gebsw_poly_r_1_poly_q5(X, Y):
        return _weighted_sw(X, Y, degree=5, energy_type="poly", energy_r=1)
    distance_functions["GEBW(1,5)"] = gebsw_poly_r_1_poly_q5

    def gebsw_poly_r_2_poly_q5(X, Y):
        return _weighted_sw(X, Y, degree=5, energy_type="poly", energy_r=2)
    distance_functions["GEBW(2,5)"] = gebsw_poly_r_2_poly_q5

    def gebsw_poly_r_3_poly_q5(X, Y):
        return _weighted_sw(X, Y, degree=5, energy_type="poly", energy_r=3)
    distance_functions["GEBW(3,5)"] = gebsw_poly_r_3_poly_q5

    def gebsw_poly_r_4_poly_q5(X, Y):
        return _weighted_sw(X, Y, degree=5, energy_type="poly", energy_r=4)
    distance_functions["GEBW(4,5)"] = gebsw_poly_r_4_poly_q5

    return distance_functions


# ======================== 敏感性分析（完整保留）========================

def run_single_experiment(source_pc, target_pc, dist_name, proj_seed, perturbation_seed,
                          temp_start=2.0, temp_end=0.1):
    target_tensor = torch.tensor(target_pc, dtype=torch.float32, device=device)

    np.random.seed(perturbation_seed)
    torch.manual_seed(perturbation_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(perturbation_seed)

    source_tensor = torch.tensor(source_pc, dtype=torch.float32, device=device, requires_grad=True)
    perturbation = torch.randn_like(source_tensor) * INITIAL_PERTURBATION
    source_tensor.data = source_tensor.data + perturbation

    optimizer = torch.optim.Adam([source_tensor], lr=LR, betas=(MOMENTUM, 0.999), weight_decay=1e-5)
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
    print("开始超参数敏感性分析...")
    print(f"{'=' * 60}")

    test_variants = ["GEBSW-f^*_e-Proj-poly(q=3)", f"GEBSW-f^*_{{{ENERGY_ORDER_SYMBOL}=2}}-Proj-poly(q=3)"]
    baseline_name = "GSW-Proj-poly(q=3) (GSW-Baseline)"

    results = {variant: {temp: [] for temp in TEMPERATURE_RANGE} for variant in test_variants}
    baseline_results = []

    print(f"\n[1/3] 计算基线 {baseline_name} ...")
    for rep in range(SENSITIVITY_REPEAT_TIMES):
        proj_seed = 2024 + rep * 1000 + 50000
        perturbation_seed = proj_seed + 100
        metrics = run_single_experiment(source_pc, target_pc, baseline_name, proj_seed, perturbation_seed)
        baseline_results.append(metrics)
        print(f"  重复{rep + 1}: CD={metrics['CD']:.4f}, FScore={metrics['FScore']:.4f}")

    baseline_mean = {k: np.mean([r[k] for r in baseline_results]) for k in ['CD', 'FScore', 'NC', 'HD']}
    baseline_std = {k: np.std([r[k] for r in baseline_results]) for k in ['CD', 'FScore', 'NC', 'HD']}
    print(f"\n基线均值: CD={baseline_mean['CD']:.4f}±{baseline_std['CD']:.4f}, "
          f"FScore={baseline_mean['FScore']:.4f}±{baseline_std['FScore']:.4f}")

    for idx, temp in enumerate(TEMPERATURE_RANGE, 1):
        print(f"\n[2/3] 测试温度 T={temp} ({idx}/{len(TEMPERATURE_RANGE)})...")
        for variant in test_variants:
            variant_results = []
            for rep in range(SENSITIVITY_REPEAT_TIMES):
                proj_seed = 2024 + rep * 1000 + int(temp * 1000)
                perturbation_seed = proj_seed + 100
                metrics = run_single_experiment(source_pc, target_pc, variant, proj_seed, perturbation_seed, temp, 0.1)
                variant_results.append(metrics)
                print(f"  {variant} 重复{rep + 1}: CD={metrics['CD']:.4f}, FScore={metrics['FScore']:.4f}")
            results[variant][temp] = variant_results

    print(f"\n[3/3] 生成敏感性分析报告...")
    analyze_and_save_sensitivity(results, baseline_mean, baseline_std, test_variants)

    return results, baseline_mean


def analyze_and_save_sensitivity(results, baseline_mean, baseline_std, test_variants):
    summary_data = []

    for variant in test_variants:
        safe_variant = variant.replace('^', '').replace('{', '').replace('}', '').replace('*', 'star')

        temps = []
        cd_means, cd_stds = [], []
        fscore_means, fscore_stds = [], []

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
                    p_val_cd, p_val_fs = 1.0, 1.0

                summary_data.append({
                    'Variant': variant, 'Temperature': temp,
                    'CD_Mean': np.mean(cds), 'CD_Std': np.std(cds),
                    'FScore_Mean': np.mean(fscores), 'FScore_Std': np.std(fscores),
                    'Better_than_GSW_CD': np.mean(cds) < baseline_mean['CD'],
                    'Better_than_GSW_FScore': np.mean(fscores) > baseline_mean['FScore'],
                    'P_value_CD': p_val_cd, 'P_value_FScore': p_val_fs
                })

        if len(temps) == 0:
            print(f"  警告：{variant} 无有效数据")
            continue

        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            ax1.errorbar(temps, cd_means, yerr=cd_stds, fmt='b-o', capsize=5,
                         label=variant, linewidth=2, markersize=6)
            ax1.axhline(y=baseline_mean['CD'], color='r', linestyle='--',
                        label=f'GSW Baseline ({baseline_mean["CD"]:.4f})')
            ax1.axhspan(baseline_mean['CD'] - baseline_std['CD'],
                        baseline_mean['CD'] + baseline_std['CD'], alpha=0.1, color='r')
            ax1.set_xlabel('Temperature Start', fontsize=12)
            ax1.set_ylabel('Chamfer Distance (lower is better)', fontsize=12)
            ax1.set_title(f'{variant}\nSensitivity to Temperature - CD', fontsize=12)
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)

            ax2.errorbar(temps, fscore_means, yerr=fscore_stds, fmt='g-o', capsize=5,
                         label=variant, linewidth=2, markersize=6)
            ax2.axhline(y=baseline_mean['FScore'], color='r', linestyle='--',
                        label=f'GSW Baseline ({baseline_mean["FScore"]:.4f})')
            ax2.axhspan(baseline_mean['FScore'] - baseline_std['FScore'],
                        baseline_mean['FScore'] + baseline_std['FScore'], alpha=0.1, color='r')
            ax2.set_xlabel('Temperature Start', fontsize=12)
            ax2.set_ylabel('F-Score (higher is better)', fontsize=12)
            ax2.set_title(f'{variant}\nSensitivity to Temperature - FScore', fontsize=12)
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = pair_named_path(os.path.join(result_dir, "sensitivity"), f"sensitivity_{safe_variant}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  ✓ 敏感性曲线已保存: {save_path}")

        except Exception as e:
            print(f"  ✗ 绘制 {variant} 曲线时出错: {e}")
            plt.close()

    try:
        summary_df = pd.DataFrame(summary_data)
        excel_path = pair_named_path(os.path.join(result_dir, "sensitivity"), "sensitivity_analysis_summary.xlsx")
        summary_df.to_excel(excel_path, index=False)
        print(f"  ✓ 敏感性数据已保存: {excel_path}")

        print(f"\n敏感性分析结论:")
        print(f"  基线 GSW: CD={baseline_mean['CD']:.4f}±{baseline_std['CD']:.4f}, "
              f"FScore={baseline_mean['FScore']:.4f}±{baseline_std['FScore']:.4f}")

        for variant in test_variants:
            better_temps_cd = [t for t in TEMPERATURE_RANGE
                               if t in results[variant]
                               and np.mean([r['CD'] for r in results[variant][t]]) < baseline_mean['CD']]
            better_temps_fs = [t for t in TEMPERATURE_RANGE
                               if t in results[variant]
                               and np.mean([r['FScore'] for r in results[variant][t]]) > baseline_mean['FScore']]

            print(f"\n  {variant}:")
            print(f"    优于基线的温度范围(CD): {better_temps_cd if better_temps_cd else '无'}")
            print(f"    优于基线的温度范围(FScore): {better_temps_fs if better_temps_fs else '无'}")

    except Exception as e:
        print(f"  ✗ 保存Excel时出错: {e}")


# ======================== 主实验函数（完整版）========================

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
                steps, means, stds = [], [], []

                for step in RECORD_STEPS:
                    key = f"step_{step}_{metric_key}"
                    if key in all_results_by_dist.get(dist_name, {}):
                        values = all_results_by_dist[dist_name][key]
                        steps.append(step)
                        means.append(np.mean(values))
                        stds.append(np.std(values))

                if steps:
                    ax.plot(steps, means, 'b-', linewidth=2, label=dist_name[:30])
                    ax.fill_between(steps,
                                    np.array(means) - np.array(stds),
                                    np.array(means) + np.array(stds),
                                    alpha=0.3)
                    ax.set_xlabel('Step')
                    ax.set_ylabel(metric_name)
                    ax.set_title(dist_name[:40])
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=8)

            for idx in range(len(distance_names), len(axes)):
                axes[idx].set_visible(False)

            plt.tight_layout()
            save_path = pair_named_path(os.path.join(result_dir, "curves"), f"convergence_{metric_name.replace(' ', '_')}.png")
            plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  ✓ 收敛曲线已保存: {save_path}")

        except Exception as e:
            print(f"  ✗ 绘制 {metric_name} 曲线时出错: {e}")
            plt.close()



# ======================== 统一框架叙事：设计空间与Pareto分析 ========================

def parse_gebsw_design(method_name):
    """把18个GEBSW配置解析为统一设计空间坐标：projection_order q 与 energy_rule。"""
    name = method_name
    if "C,1" in name:
        return {"family_role": "framework_special_case", "energy_rule": "C", "energy_type": "uniform", "energy_order": 0, "q": 1, "special_case": "SW"}
    if "C,3" in name:
        return {"family_role": "framework_special_case", "energy_rule": "C", "energy_type": "uniform", "energy_order": 0, "q": 3, "special_case": "GSW"}
    if "C,5" in name:
        return {"family_role": "framework_special_case", "energy_rule": "C", "energy_type": "uniform", "energy_order": 0, "q": 5, "special_case": "GSW"}
    if "e,1" in name:
        return {"family_role": "framework_special_case", "energy_rule": "e", "energy_type": "exp", "energy_order": np.nan, "q": 1, "special_case": "EBSW"}
    if "e,3" in name:
        return {"family_role": "GEBSW_configuration", "energy_rule": "e", "energy_type": "exp", "energy_order": np.nan, "q": 3, "special_case": "GEBSW"}
    if "e,5" in name:
        return {"family_role": "GEBSW_configuration", "energy_rule": "e", "energy_type": "exp", "energy_order": np.nan, "q": 5, "special_case": "GEBSW"}
    for r in [1, 2, 3, 4]:
        for q in [1, 3, 5]:
            if f"({r},{q})" in name:
                return {
                    "family_role": "framework_special_case" if q == 1 else "GEBSW_configuration",
                    "energy_rule": f"r={r}",
                    "energy_type": "poly",
                    "energy_order": r,
                    "q": q,
                    "special_case": "EBSW" if q == 1 else "GEBSW"
                }
    if "Sampled-MaxSW" in name or "Max-SW" in name:
        return {"family_role": "external_sliced_ot_baseline", "energy_rule": "adaptive-max", "energy_type": "max", "energy_order": np.nan, "q": np.nan, "special_case": "Sampled-MaxSW"}
    if "DSW" in name or "Distributional" in name:
        return {"family_role": "external_sliced_ot_baseline", "energy_rule": "learned-distribution", "energy_type": "distributional", "energy_order": np.nan, "q": np.nan, "special_case": "DSW"}
    return {"family_role": "unknown", "energy_rule": "unknown", "energy_type": "unknown", "energy_order": np.nan, "q": np.nan, "special_case": "unknown"}


def _dominates(a, b, metric_specs):
    """a 是否 Pareto 支配 b。metric_specs: [(metric, lower_better_bool), ...]"""
    not_worse_all = True
    strictly_better_any = False
    for metric, lower_better in metric_specs:
        av, bv = a[metric], b[metric]
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


def compute_pareto_flags(df, metric_specs, group_cols=("pair_index", "repeat")):
    """在每个pair-repeat内部计算Pareto前沿。"""
    out = df.copy()
    out["pareto"] = False
    out["dominated_by_count"] = 0
    out["dominates_count"] = 0
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
            out.at[i, "dominated_by_count"] = dominated_by
            out.at[i, "dominates_count"] = dominates
            out.at[i, "pareto"] = dominated_by == 0
    return out


def build_framework_long_table(all_results_by_dist, final_step):
    """从 all_results_by_dist 构造 pair-repeat-method 长表。"""
    rows = []
    for method, data in all_results_by_dist.items():
        n = len(data.get("total_time", []))
        for repeat_idx in range(n):
            row = {
                "pair_index": CURRENT_PAIR_INDEX,
                "pair_group": CURRENT_PAIR_GROUP,
                "source": CURRENT_SOURCE_NAME,
                "target": CURRENT_TARGET_NAME,
                "repeat": repeat_idx + 1,
                "method": method,
                "W2": data.get(f"step_{final_step}_distance", [np.nan] * n)[repeat_idx] if repeat_idx < len(data.get(f"step_{final_step}_distance", [])) else np.nan,
                "CD": data.get(f"step_{final_step}_cd", [np.nan] * n)[repeat_idx] if repeat_idx < len(data.get(f"step_{final_step}_cd", [])) else np.nan,
                "FScore": data.get(f"step_{final_step}_fscore", [np.nan] * n)[repeat_idx] if repeat_idx < len(data.get(f"step_{final_step}_fscore", [])) else np.nan,
                "NC": data.get(f"step_{final_step}_normal_consistency", [np.nan] * n)[repeat_idx] if repeat_idx < len(data.get(f"step_{final_step}_normal_consistency", [])) else np.nan,
                "HD": data.get(f"step_{final_step}_hausdorff", [np.nan] * n)[repeat_idx] if repeat_idx < len(data.get(f"step_{final_step}_hausdorff", [])) else np.nan,
                "Runtime": data.get("total_time", [np.nan] * n)[repeat_idx] if repeat_idx < len(data.get("total_time", [])) else np.nan,
            }
            row.update(parse_gebsw_design(method))
            rows.append(row)
    return pd.DataFrame(rows)


def add_average_ranks(df):
    """对每个pair-repeat内按指标排名；越低越好指标升序，越高越好指标降序。"""
    out = df.copy()
    rank_specs = [("W2", True), ("CD", True), ("FScore", False), ("NC", False), ("HD", True), ("Runtime", True)]
    for metric, lower_better in rank_specs:
        out[f"rank_{metric}"] = out.groupby(["pair_index", "repeat"])[metric].rank(ascending=lower_better, method="average")
    quality_rank_cols = ["rank_W2", "rank_CD", "rank_FScore", "rank_NC", "rank_HD"]
    out["avg_quality_rank"] = out[quality_rank_cols].mean(axis=1)
    out["avg_all_rank"] = out[quality_rank_cols + ["rank_Runtime"]].mean(axis=1)
    return out


def compute_factor_effects(df):
    """描述性two-way effect size: q、energy、q×energy 对各指标方差的解释比例。"""
    rows = []
    metrics = ["W2", "CD", "FScore", "NC", "HD", "Runtime", "avg_quality_rank"]
    d = df.dropna(subset=["q", "energy_rule"]).copy()
    d["q"] = d["q"].astype(int).astype(str)
    for metric in metrics:
        dd = d.dropna(subset=[metric])
        if dd.empty:
            continue
        y = dd[metric].astype(float)
        grand = y.mean()
        ss_total = float(((y - grand) ** 2).sum())
        if ss_total <= 1e-20:
            continue
        means_q = dd.groupby("q")[metric].mean()
        counts_q = dd.groupby("q")[metric].count()
        ss_q = float(sum(counts_q[k] * (means_q[k] - grand) ** 2 for k in means_q.index))
        means_e = dd.groupby("energy_rule")[metric].mean()
        counts_e = dd.groupby("energy_rule")[metric].count()
        ss_e = float(sum(counts_e[k] * (means_e[k] - grand) ** 2 for k in means_e.index))
        means_qe = dd.groupby(["q", "energy_rule"])[metric].mean()
        counts_qe = dd.groupby(["q", "energy_rule"])[metric].count()
        ss_qe_cells = float(sum(counts_qe[k] * (means_qe[k] - grand) ** 2 for k in means_qe.index))
        ss_interaction = max(0.0, ss_qe_cells - ss_q - ss_e)
        rows.append({
            "metric": metric,
            "eta2_q": ss_q / ss_total,
            "eta2_energy": ss_e / ss_total,
            "eta2_q_x_energy": ss_interaction / ss_total,
            "interpretation": "larger eta2 means this design axis explains more variance"
        })
    return pd.DataFrame(rows)


def save_design_space_heatmaps(summary_df, save_dir):
    """为每个核心指标输出 q×energy heatmap。"""
    os.makedirs(save_dir, exist_ok=True)
    try:
        metrics = ["W2", "CD", "FScore", "NC", "HD", "Runtime", "pareto_quality_rate", "avg_quality_rank"]
        for metric in metrics:
            if metric not in summary_df.columns:
                continue
            pivot = summary_df.pivot_table(index="energy_rule", columns="q", values=metric, aggfunc="mean")
            if pivot.empty:
                continue
            plt.figure(figsize=(7, 4.5), dpi=160)
            im = plt.imshow(pivot.values, aspect="auto")
            plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
            plt.yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
            plt.xlabel("Projection order q")
            plt.ylabel("Energy / weighting rule")
            plt.title(f"GEBSW design-space heatmap: {metric}")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    val = pivot.values[i, j]
                    if pd.notna(val):
                        plt.text(j, i, f"{val:.3g}", ha="center", va="center", fontsize=8)
            plt.tight_layout()
            plt.savefig(pair_named_path(save_dir, f"heatmap_{metric}.png"), bbox_inches="tight", facecolor="white")
            plt.close()
    except Exception as e:
        print(f"  ✗ heatmap绘制失败: {e}")
        plt.close()


def framework_design_space_analysis(all_results_by_dist, final_step):
    """统一框架叙事的核心分析：设计空间、Pareto、rank、effect size。"""
    print("\n开始统一框架设计空间分析...")
    out_dir = f"{result_dir}/framework_analysis"
    os.makedirs(out_dir, exist_ok=True)

    long_df = build_framework_long_table(all_results_by_dist, final_step)
    if long_df.empty:
        print("  ⚠️  无可分析数据")
        return

    quality_specs = [("W2", True), ("CD", True), ("FScore", False), ("NC", False), ("HD", True)]
    efficiency_specs = quality_specs + [("Runtime", True)]
    q_pareto = compute_pareto_flags(long_df, quality_specs).rename(columns={"pareto": "pareto_quality"})
    e_pareto = compute_pareto_flags(long_df, efficiency_specs).rename(columns={"pareto": "pareto_efficiency"})
    long_df["pareto_quality"] = q_pareto["pareto_quality"].values
    long_df["quality_dominated_by_count"] = q_pareto["dominated_by_count"].values
    long_df["quality_dominates_count"] = q_pareto["dominates_count"].values
    long_df["pareto_efficiency"] = e_pareto["pareto_efficiency"].values
    long_df["efficiency_dominated_by_count"] = e_pareto["dominated_by_count"].values
    long_df["efficiency_dominates_count"] = e_pareto["dominates_count"].values
    long_df = add_average_ranks(long_df)

    summary = long_df.groupby(["method", "q", "energy_rule", "energy_type", "special_case", "family_role"], dropna=False).agg(
        W2=("W2", "mean"), CD=("CD", "mean"), FScore=("FScore", "mean"), NC=("NC", "mean"), HD=("HD", "mean"), Runtime=("Runtime", "mean"),
        W2_std=("W2", "std"), CD_std=("CD", "std"), FScore_std=("FScore", "std"), NC_std=("NC", "std"), HD_std=("HD", "std"), Runtime_std=("Runtime", "std"),
        pareto_quality_rate=("pareto_quality", "mean"), pareto_efficiency_rate=("pareto_efficiency", "mean"),
        avg_quality_rank=("avg_quality_rank", "mean"), avg_all_rank=("avg_all_rank", "mean"),
        quality_dominates_count=("quality_dominates_count", "mean"), quality_dominated_by_count=("quality_dominated_by_count", "mean"),
        n=("method", "count")
    ).reset_index()
    summary["pareto_quality_rate"] *= 100.0
    summary["pareto_efficiency_rate"] *= 100.0
    summary = summary.sort_values(["pareto_quality_rate", "avg_quality_rank"], ascending=[False, True])

    difficulty_summary = long_df.groupby(["pair_group", "method", "q", "energy_rule"], dropna=False).agg(
        pareto_quality_rate=("pareto_quality", "mean"), pareto_efficiency_rate=("pareto_efficiency", "mean"),
        avg_quality_rank=("avg_quality_rank", "mean"), W2=("W2", "mean"), CD=("CD", "mean"), FScore=("FScore", "mean"), NC=("NC", "mean"), HD=("HD", "mean"), Runtime=("Runtime", "mean"), n=("method", "count")
    ).reset_index()
    difficulty_summary["pareto_quality_rate"] *= 100.0
    difficulty_summary["pareto_efficiency_rate"] *= 100.0

    factor_effects = compute_factor_effects(long_df)

    excel_path = pair_named_path(out_dir, "GEBSW_framework_design_space_analysis.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        long_df.to_excel(writer, sheet_name="long_pair_repeat_method", index=False)
        summary.to_excel(writer, sheet_name="design_space_summary", index=False)
        difficulty_summary.to_excel(writer, sheet_name="difficulty_summary", index=False)
        factor_effects.to_excel(writer, sheet_name="factor_effect_eta2", index=False)
        long_df[long_df["pareto_quality"]].to_excel(writer, sheet_name="quality_pareto_points", index=False)
        long_df[long_df["pareto_efficiency"]].to_excel(writer, sheet_name="efficiency_pareto_points", index=False)

    save_design_space_heatmaps(summary, f"{out_dir}/heatmaps")
    print(f"  ✓ 统一框架分析已保存: {excel_path}")


def aggregate_framework_analysis_from_dirs(base_dir="."):
    """批量模式结束后汇总15个pair的framework_analysis。"""
    files = []
    for root, dirs, names in os.walk(base_dir):
        for name in names:
            if name.endswith("GEBSW_framework_design_space_analysis.xlsx"):
                files.append(os.path.join(root, name))
    if not files:
        print("未找到可汇总的framework_analysis文件。")
        return
    all_long = []
    for f in files:
        try:
            df = pd.read_excel(f, sheet_name="long_pair_repeat_method")
            df["source_file"] = f
            all_long.append(df)
        except Exception as e:
            print(f"  跳过 {f}: {e}")
    if not all_long:
        return
    combined = pd.concat(all_long, ignore_index=True)
    out_dir = "PCR-ModelNet40-GEBSW-Framework-ExternalBaselines-Aggregate" if INCLUDE_EXTERNAL_BASELINES else "PCR-ModelNet40-GEBSW-Framework-Aggregate"
    os.makedirs(out_dir, exist_ok=True)
    quality_specs = [("W2", True), ("CD", True), ("FScore", False), ("NC", False), ("HD", True)]
    efficiency_specs = quality_specs + [("Runtime", True)]
    combined_q = compute_pareto_flags(combined, quality_specs, group_cols=("pair_index", "repeat"))
    combined_e = compute_pareto_flags(combined, efficiency_specs, group_cols=("pair_index", "repeat"))
    combined["pareto_quality_global"] = combined_q["pareto"].values
    combined["pareto_efficiency_global"] = combined_e["pareto"].values
    combined = add_average_ranks(combined)
    summary = combined.groupby(["method", "q", "energy_rule", "energy_type", "special_case", "family_role"], dropna=False).agg(
        W2=("W2", "mean"), CD=("CD", "mean"), FScore=("FScore", "mean"), NC=("NC", "mean"), HD=("HD", "mean"), Runtime=("Runtime", "mean"),
        pareto_quality_rate=("pareto_quality_global", "mean"), pareto_efficiency_rate=("pareto_efficiency_global", "mean"),
        avg_quality_rank=("avg_quality_rank", "mean"), avg_all_rank=("avg_all_rank", "mean"), n=("method", "count")
    ).reset_index()
    summary["pareto_quality_rate"] *= 100.0
    summary["pareto_efficiency_rate"] *= 100.0
    summary = summary.sort_values(["pareto_quality_rate", "avg_quality_rank"], ascending=[False, True])
    difficulty_summary = combined.groupby(["pair_group", "method", "q", "energy_rule"], dropna=False).agg(
        pareto_quality_rate=("pareto_quality_global", "mean"), pareto_efficiency_rate=("pareto_efficiency_global", "mean"), avg_quality_rank=("avg_quality_rank", "mean"), W2=("W2", "mean"), CD=("CD", "mean"), FScore=("FScore", "mean"), NC=("NC", "mean"), HD=("HD", "mean"), Runtime=("Runtime", "mean"), n=("method", "count")
    ).reset_index()
    difficulty_summary["pareto_quality_rate"] *= 100.0
    difficulty_summary["pareto_efficiency_rate"] *= 100.0
    effects = compute_factor_effects(combined)
    out_xlsx = os.path.join(out_dir, "ModelNet40_all_pairs_framework_analysis.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        combined.to_excel(writer, sheet_name="all_long_results", index=False)
        summary.to_excel(writer, sheet_name="overall_design_summary", index=False)
        difficulty_summary.to_excel(writer, sheet_name="difficulty_summary", index=False)
        effects.to_excel(writer, sheet_name="factor_effect_eta2", index=False)
    save_design_space_heatmaps(summary, os.path.join(out_dir, "heatmaps"))
    print(f"\n✅ ModelNet40统一框架总分析已保存: {out_xlsx}")

def point_cloud_reconstruction_experiment_final():
    print("加载点云数据...")
    source_pc = read_off_file_robust(source_path)
    target_pc = read_off_file_robust(target_path)
    source_pc = normalize_point_cloud(source_pc)
    target_pc = normalize_point_cloud(target_pc)

    target_tensor = torch.tensor(target_pc, dtype=torch.float32, device=device)

    raw_results = []
    all_results_by_dist = {}

    temp_funcs = get_all_distance_functions(2024, 0, 0, NUM_STEPS)
    all_distance_names = list(temp_funcs.keys())
    external_note = " + 2 个外部 sliced-OT baseline" if INCLUDE_EXTERNAL_BASELINES else ""
    print(f"共有 {len(all_distance_names)} 种距离变体{external_note}: {all_distance_names}")

    # 【新增】为组合图准备：记录每个变体的索引映射
    variant_name_to_idx = {name: idx for idx, name in enumerate(all_distance_names)}

    for repeat in range(REPEAT_TIMES):
        print(f"\n===== 重复实验 {repeat + 1}/{REPEAT_TIMES} =====")
        base_seed = 2024 + repeat * 100000
        proj_seed = base_seed + 1000
        perturbation_seed = base_seed + 5000

        distance_functions = get_all_distance_functions(proj_seed, repeat, current_step=0)

        for dist_name in distance_functions.keys():
            if dist_name not in all_results_by_dist:
                all_results_by_dist[dist_name] = {}

        for dist_idx, (dist_name, dist_func) in enumerate(tqdm(distance_functions.items(),
                                                               desc=f"Rep{repeat + 1}")):
            try:
                np.random.seed(perturbation_seed)
                torch.manual_seed(perturbation_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(perturbation_seed)

                source_tensor = torch.tensor(source_pc, dtype=torch.float32, device=device, requires_grad=True)
                perturbation = torch.randn_like(source_tensor) * INITIAL_PERTURBATION
                source_tensor.data = source_tensor.data + perturbation

                optimizer = torch.optim.Adam([source_tensor], lr=LR, betas=(MOMENTUM, 0.999), weight_decay=1e-5)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_STEPS, eta_min=LR * 0.01)
                start_time = time.time()

                single_run_record = {
                    "distance_name": dist_name, "repeat": repeat + 1,
                    "base_seed": base_seed, "proj_seed": proj_seed, "total_time": 0.0
                }

                # 【新增】仅在第1次重复时，收集该变体的所有step点云用于组合图
                variant_step_pcs = {} if repeat == 0 else None

                for step in range(NUM_STEPS + 1):
                    optimizer.zero_grad()

                    # 统一框架实验：所有配置每一步都重新通过同一函数工厂取得当前距离，
                    # 避免某些energy-weighted配置使用退火温度而uniform配置不更新导致流程不一致。
                    current_dist_funcs = get_all_distance_functions(proj_seed, repeat, step, NUM_STEPS)
                    current_dist_func = current_dist_funcs[dist_name]

                    distance = current_dist_func(source_tensor, target_tensor)
                    distance.backward()
                    torch.nn.utils.clip_grad_norm_([source_tensor], max_norm=1.0)
                    optimizer.step()
                    scheduler.step()

                    # 【关键修复】所有重复实验都记录统计指标
                    if step in RECORD_STEPS:
                        with torch.no_grad():
                            current_source_pc_gpu = source_tensor.detach()

                            dist_val = distance.item()
                            cd_val = chamfer_distance(current_source_pc_gpu, target_tensor)
                            fscore_val = compute_f_score(current_source_pc_gpu, target_tensor)
                            hd_val = compute_hausdorff_distance(current_source_pc_gpu, target_tensor)

                            nc_val = compute_normal_consistency(current_source_pc_gpu.cpu().numpy(), target_pc, seed=base_seed + step)

                            single_run_record[f"step_{step}_distance"] = dist_val
                            single_run_record[f"step_{step}_cd"] = cd_val
                            single_run_record[f"step_{step}_fscore"] = fscore_val
                            single_run_record[f"step_{step}_normal_consistency"] = nc_val
                            single_run_record[f"step_{step}_hausdorff"] = hd_val

                            for metric_en, val in [("distance", dist_val), ("cd", cd_val),
                                                   ("fscore", fscore_val), ("normal_consistency", nc_val),
                                                   ("hausdorff", hd_val)]:
                                key = f"step_{step}_{metric_en}"
                                if key not in all_results_by_dist[dist_name]:
                                    all_results_by_dist[dist_name][key] = []
                                all_results_by_dist[dist_name][key].append(val)

                            # 【修改】第1次重复时，仅在 VISUAL_STEPS 中存储点云数据用于后续组合图。
                            # 数值结果仍按 RECORD_STEPS 完整记录，二者互不干扰。
                            if repeat == 0 and variant_step_pcs is not None and step in VISUAL_STEPS:
                                current_source_pc_np = current_source_pc_gpu.cpu().numpy()
                                variant_step_pcs[step] = current_source_pc_np

                # 【新增】第1次重复结束后，保存该变体的组合图
                if repeat == 0 and variant_step_pcs and len(variant_step_pcs) > 0:
                    variant_idx = variant_name_to_idx[dist_name]
                    plot_variant_combined_figure(
                        source_pc, target_pc, variant_step_pcs,
                        dist_name, variant_idx,
                        f"{result_dir}/snapshots", VISUAL_STEPS
                    )

                total_time = time.time() - start_time
                single_run_record["total_time"] = total_time
                raw_results.append(single_run_record)

                time_key = "total_time"
                if time_key not in all_results_by_dist[dist_name]:
                    all_results_by_dist[dist_name][time_key] = []
                all_results_by_dist[dist_name][time_key].append(total_time)

            except Exception as e:
                print(f"\n错误：处理 {dist_name} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue

    # 保存原始数据（现在包含所有10次重复的详细指标）
    raw_df = pd.DataFrame(raw_results)
    raw_df.to_excel(pair_named_path(os.path.join(result_dir, "debug"), "raw_results.xlsx"), index=False)
    print(f"\n原始数据已保存（包含 {REPEAT_TIMES} 次重复的完整指标）")

    # 生成指标Excel和统计检验（完整保留）
    metrics_config = [
        ("距离值W2", "distance", True), ("倒角距离CD", "cd", True),
        ("FScore", "fscore", False), ("法向量一致性NC", "normal_consistency", False),
        ("豪斯多夫距离HD", "hausdorff", True), ("总耗时(秒)", "total_time", True)
    ]

    final_step = RECORD_STEPS[-1]
    all_p_values = []
    statistical_results = {}

    print("\n生成指标文件...")

    # 完整基线映射
    baseline_mapping = {
        "GEBSW-f^*_e-Proj-poly(q=1) (EBSW-Baseline)": "GSW-Proj-poly(q=1) (SW-Baseline)",
        f"GEBSW-f^*_{{{ENERGY_ORDER_SYMBOL}=1}}-Proj-poly(q=1) (EBSW-Baseline)": "GSW-Proj-poly(q=1) (SW-Baseline)",
        f"GEBSW-f^*_{{{ENERGY_ORDER_SYMBOL}=2}}-Proj-poly(q=1) (EBSW-Baseline)": "GSW-Proj-poly(q=1) (SW-Baseline)",
        f"GEBSW-f^*_{{{ENERGY_ORDER_SYMBOL}=3}}-Proj-poly(q=1) (EBSW-Baseline)": "GSW-Proj-poly(q=1) (SW-Baseline)",
        f"GEBSW-f^*_{{{ENERGY_ORDER_SYMBOL}=4}}-Proj-poly(q=1) (EBSW-Baseline)": "GSW-Proj-poly(q=1) (SW-Baseline)",
        "GEBSW-f^*_e-Proj-poly(q=3)": "GSW-Proj-poly(q=3) (GSW-Baseline)",
        f"GEBSW-f^*_{{{ENERGY_ORDER_SYMBOL}=1}}-Proj-poly(q=3)": "GSW-Proj-poly(q=3) (GSW-Baseline)",
        f"GEBSW-f^*_{{{ENERGY_ORDER_SYMBOL}=2}}-Proj-poly(q=3)": "GSW-Proj-poly(q=3) (GSW-Baseline)",
        f"GEBSW-f^*_{{{ENERGY_ORDER_SYMBOL}=3}}-Proj-poly(q=3)": "GSW-Proj-poly(q=3) (GSW-Baseline)",
        f"GEBSW-f^*_{{{ENERGY_ORDER_SYMBOL}=4}}-Proj-poly(q=3)": "GSW-Proj-poly(q=3) (GSW-Baseline)",
        "GEBSW-f^*_e-Proj-poly(q=5)": "GSW-Proj-poly(q=5) (GSW-Baseline)",
        f"GEBSW-f^*_{{{ENERGY_ORDER_SYMBOL}=1}}-Proj-poly(q=5)": "GSW-Proj-poly(q=5) (GSW-Baseline)",
        f"GEBSW-f^*_{{{ENERGY_ORDER_SYMBOL}=2}}-Proj-poly(q=5)": "GSW-Proj-poly(q=5) (GSW-Baseline)",
        f"GEBSW-f^*_{{{ENERGY_ORDER_SYMBOL}=3}}-Proj-poly(q=5)": "GSW-Proj-poly(q=5) (GSW-Baseline)",
        f"GEBSW-f^*_{{{ENERGY_ORDER_SYMBOL}=4}}-Proj-poly(q=5)": "GSW-Proj-poly(q=5) (GSW-Baseline)",
    }

    gebsw_keys = [k for k in all_results_by_dist.keys() if ("GEBSW" in k or "EBSW" in k)]
    sw_key = "GSW-Proj-poly(q=1) (SW-Baseline)"

    # 生成各指标Excel
    for metric_cn, metric_en, is_lower_better in metrics_config:
        print(f"\n处理指标: {metric_cn}")
        rows = []
        for dist_name in all_results_by_dist.keys():
            row = {"距离变体名称": dist_name}
            if metric_en == "total_time":
                key = "total_time"
                if key in all_results_by_dist[dist_name]:
                    values = all_results_by_dist[dist_name][key]
                    row["总耗时_Mean"] = round(np.mean(values), PRECISION_DIGITS)
                    row["总耗时_Std"] = round(np.std(values, ddof=1), PRECISION_DIGITS)
                    row["总耗时_Raw"] = str([round(v, PRECISION_DIGITS) for v in values])
            else:
                for step in RECORD_STEPS:
                    key = f"step_{step}_{metric_en}"
                    if key in all_results_by_dist[dist_name]:
                        values = all_results_by_dist[dist_name][key]
                        row[f"Step{step}_Mean"] = round(np.mean(values), PRECISION_DIGITS)
                        row[f"Step{step}_Std"] = round(np.std(values, ddof=1), PRECISION_DIGITS)
                        row[f"Step{step}_Raw"] = str([round(v, PRECISION_DIGITS) for v in values])
            rows.append(row)
        metric_df = pd.DataFrame(rows)
        excel_filename = f"{metric_cn}.xlsx" if metric_cn != "总耗时(秒)" else "总耗时.xlsx"
        metric_df.to_excel(pair_named_path(os.path.join(result_dir, "metrics"), excel_filename), index=False)
        print(f"  ✓ 已保存: {pair_named_path(os.path.join(result_dir, 'metrics'), excel_filename)} ({len(rows)}行)")

    # 统计显著性检验
    print(f"\n{'=' * 70}")
    print(f"开始统计显著性检验 (REPEAT_TIMES={REPEAT_TIMES})")
    print(f"{'=' * 70}")

    if REPEAT_TIMES >= 2:
        test_metrics = [("倒角距离CD", "cd", True), ("FScore", "fscore", False)]
        success_count = 0

        for metric_cn, metric_en, is_lower_better in test_metrics:
            print(f"\n【指标: {metric_cn}】")
            step_key = f"step_{final_step}_{metric_en}"

            for gebsw_key in gebsw_keys:
                print(f"\n  变体: {gebsw_key[:45]}...")
                gebsw_vals = all_results_by_dist.get(gebsw_key, {}).get(step_key, [])
                if not gebsw_vals or len(gebsw_vals) < 2:
                    print(f"    ❌ 数据不足")
                    continue

                gsw_key = baseline_mapping.get(gebsw_key)
                gsw_vals = all_results_by_dist.get(gsw_key, {}).get(step_key, []) if gsw_key else []
                sw_vals = all_results_by_dist.get(sw_key, {}).get(step_key, [])

                test_results = statistical_significance_test(gebsw_vals, gsw_vals, sw_vals)

                if test_results:
                    test_key = f"{gebsw_key}_vs_Baselines_{metric_en}"
                    statistical_results[test_key] = test_results
                    success_count += 1

                    if 'GEBSW_vs_GSW' in test_results:
                        all_p_values.append({
                            'test_name': f"{test_key}_vs_GSW",
                            'metric': metric_cn,
                            'p_value': test_results['GEBSW_vs_GSW']['p_value'],
                            'details': test_results['GEBSW_vs_GSW']
                        })
                    if 'GEBSW_vs_SW' in test_results:
                        all_p_values.append({
                            'test_name': f"{test_key}_vs_SW",
                            'metric': metric_cn,
                            'p_value': test_results['GEBSW_vs_SW']['p_value'],
                            'details': test_results['GEBSW_vs_SW']
                        })

        print(f"\n✓ 成功生成 {success_count} 个统计检验结果")

        # 保存统计检验详情
        if statistical_results:
            print(f"\n正在保存统计检验结果...")
            try:
                with pd.ExcelWriter(pair_named_path(os.path.join(result_dir, "metrics"), "统计显著性检验_详细结果.xlsx"), engine='openpyxl') as writer:
                    for test_name, test_results in statistical_results.items():
                        rows = []
                        for comparison, values in test_results.items():
                            rows.append({
                                "对比组": comparison,
                                "样本量(n)": values.get('sample_size', REPEAT_TIMES),
                                "t统计量": values.get('t_statistic'),
                                "p值": values.get('p_value'),
                                "显著性(α=0.05)": "是" if values.get('significant_05') else "否",
                                "显著性(α=0.01)": "是" if values.get('significant_01') else "否",
                                "Cohens_d": values.get('cohens_d'),
                                "效应量": values.get('effect_magnitude'),
                                "均值差异": values.get('mean_diff'),
                                "备注": values.get('note', '')
                            })
                        if rows:
                            df = pd.DataFrame(rows)
                            safe_name = test_name[:28] + "..." if len(test_name) > 31 else test_name
                            df.to_excel(writer, sheet_name=safe_name, index=False)
                print(f"✅ 统计检验详情已保存")
            except Exception as e:
                print(f"❌ 保存失败: {e}")

        # 保存FDR校正
        if all_p_values:
            pvals = [item['p_value'] for item in all_p_values]
            reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

            pd.DataFrame({
                '检验名称': [item['test_name'] for item in all_p_values],
                '指标': [item['metric'] for item in all_p_values],
                '原始p值': pvals,
                'FDR校正后p值': pvals_corrected,
                '显著性(FDR)': reject,
                'Cohens_d': [item['details'].get('cohens_d') for item in all_p_values]
            }).to_excel(pair_named_path(os.path.join(result_dir, "metrics"), "多重比较校正_FDR.xlsx"), index=False)
            print(f"✅ FDR校正已保存 ({len(pvals)}个检验)")
    else:
        print(f"跳过统计检验（REPEAT_TIMES={REPEAT_TIMES} < 2）")

    # 生成汇总表
    summary_rows = []
    for dist_name in all_results_by_dist.keys():
        summary_row = {"距离变体": dist_name}
        for metric_cn, metric_en, is_lower_better in metrics_config:
            if metric_en == "total_time":
                key = "total_time"
                if key in all_results_by_dist[dist_name]:
                    values = all_results_by_dist[dist_name][key]
                    summary_row["总耗时(秒)_Mean"] = round(np.mean(values), REPORT_DIGITS)
                    summary_row["总耗时(秒)_Std"] = round(np.std(values, ddof=1), REPORT_DIGITS)
            else:
                key = f"step_{final_step}_{metric_en}"
                if key in all_results_by_dist[dist_name]:
                    values = all_results_by_dist[dist_name][key]
                    summary_row[f"{metric_cn}_Mean"] = round(np.mean(values), REPORT_DIGITS)
                    summary_row[f"{metric_cn}_Std"] = round(np.std(values, ddof=1), REPORT_DIGITS)
        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_excel(pair_named_path(os.path.join(result_dir, "metrics"), f"汇总_最终步骤_Step{final_step}_Complete.xlsx"), index=False)
    print(f"  ✓ 汇总表已保存")

    # 绘制收敛曲线
    print("\n绘制收敛曲线...")
    plot_convergence_curves(all_results_by_dist, list(all_results_by_dist.keys()),
                            [(m[0], m[1]) for m in metrics_config if m[1] != "total_time"])

    print(f"\n{'=' * 60}")
    print("主实验完成！")
    print(f"共处理 {len(all_results_by_dist)} 种距离变体")
    print(f"结果目录: {result_dir}/")
    print(f"{'=' * 60}")

    return source_pc, target_pc, all_results_by_dist




def _configure_worker_gpu(gpu_id=None):
    """在每个子进程内绑定 GPU，并刷新全局 device。"""
    global device
    if gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(int(gpu_id))
        device = torch.device(f"cuda:{int(gpu_id)}")
        print(f"[worker pid={os.getpid()}] 使用 GPU {gpu_id}: {torch.cuda.get_device_name(int(gpu_id))}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[worker pid={os.getpid()}] 使用设备: {device}")


def _run_modelnet_pair(test_idx, gpu_id=None):
    """运行一个 ModelNet40 pair；供单进程和多 GPU worker 复用。"""
    global source_path, target_path, result_dir
    global CURRENT_PAIR_INDEX, CURRENT_PAIR_GROUP, CURRENT_SOURCE_NAME, CURRENT_TARGET_NAME

    _configure_worker_gpu(gpu_id)

    if test_idx < 0 or test_idx >= len(PAIR_CONFIGS):
        raise ValueError(f"TEST_INDEX {test_idx} 超出范围 (0-{len(PAIR_CONFIGS) - 1})")

    src_file, tgt_file, group = PAIR_CONFIGS[test_idx]
    source_path = os.path.join(BASE_PATH, src_file)
    target_path = os.path.join(BASE_PATH, tgt_file)
    src_name = src_file.replace('.off', '')
    tgt_name = tgt_file.replace('.off', '')
    result_dir = f"PCR-{group}-{src_name}-{tgt_name}"
    CURRENT_PAIR_INDEX = test_idx
    CURRENT_PAIR_GROUP = group
    CURRENT_SOURCE_NAME = src_name
    CURRENT_TARGET_NAME = tgt_name

    print(f"\n{'=' * 70}")
    print(f"【GPU {gpu_id if gpu_id is not None else 'single'}】处理 ModelNet40 Pair {test_idx}: {src_name} → {tgt_name}")
    print(f"分组: {group}")
    print(f"结果目录: {result_dir}")
    print(f"{'-' * 70}\n")

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(f"{result_dir}/snapshots", exist_ok=True)
    os.makedirs(f"{result_dir}/metrics", exist_ok=True)
    os.makedirs(f"{result_dir}/curves", exist_ok=True)
    os.makedirs(f"{result_dir}/debug", exist_ok=True)
    os.makedirs(f"{result_dir}/sensitivity", exist_ok=True)

    try:
        source_pc, target_pc, all_results = point_cloud_reconstruction_experiment_final()
        if RUN_SENSITIVITY_ANALYSIS:
            print(f"\n{'=' * 60}")
            print("开始敏感性分析...")
            hyperparameter_sensitivity_analysis(source_pc, target_pc)
        print(f"\n✅ ModelNet40 Pair {test_idx} ({src_name}-{tgt_name}) 实验完成！")
        return (True, test_idx, src_name, tgt_name, group, "")
    except Exception as e:
        print(f"\n❌ 错误：处理 ModelNet40 Pair {test_idx} ({src_name}-{tgt_name}) 时发生异常: {e}")
        import traceback
        traceback.print_exc()
        return (False, test_idx, src_name, tgt_name, group, str(e))


def _run_modelnet_pair_chunk(pair_indices, gpu_id):
    """一个 GPU 子进程顺序处理多个 pair，避免频繁启动进程。"""
    results = []
    for test_idx in pair_indices:
        results.append(_run_modelnet_pair(int(test_idx), gpu_id=gpu_id))
    return results


def _select_gpu_ids():
    if not torch.cuda.is_available():
        return []
    visible_count = torch.cuda.device_count()
    if MULTI_GPU_IDS is None:
        return list(range(visible_count))
    return [int(g) for g in MULTI_GPU_IDS if int(g) < visible_count]


if __name__ == "__main__":
    if BATCH_MODE:
        total_groups = len(PAIR_CONFIGS)
        print(f"\n{'=' * 70}")
        print(f"【ModelNet40 批量运行模式】共 {total_groups} 组实验待执行")
        print(f"能量函数阶数符号: {ENERGY_ORDER_SYMBOL}")
        print(f"外部 baseline: {'开启' if INCLUDE_EXTERNAL_BASELINES else '关闭'}")
        print(f"{'=' * 70}\n")

        success_groups, failed_groups = [], []
        gpu_ids = _select_gpu_ids()
        use_multi = USE_MULTI_GPU and len(gpu_ids) >= 2 and total_groups >= MULTI_GPU_MIN_TASKS

        if use_multi:
            print(f"[Multi-GPU] 启用 pair 级并行，GPU 列表: {gpu_ids}")
            chunks = [[] for _ in gpu_ids]
            for i, pair_idx in enumerate(range(total_groups)):
                chunks[i % len(gpu_ids)].append(pair_idx)
            for gpu, chunk in zip(gpu_ids, chunks):
                print(f"  GPU {gpu} -> pair indices: {chunk}")
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=len(gpu_ids)) as pool:
                nested = pool.starmap(_run_modelnet_pair_chunk, [(chunk, gpu) for gpu, chunk in zip(gpu_ids, chunks) if chunk])
            flat_results = [item for sub in nested for item in sub]
        else:
            print("[Single-GPU] 未启用多 GPU：可能只有一张 GPU、BATCH_MODE=False，或任务数过少。")
            flat_results = [_run_modelnet_pair(i, gpu_id=(gpu_ids[0] if gpu_ids else None)) for i in range(total_groups)]

        for ok, idx, src, tgt, grp, err in flat_results:
            if ok:
                success_groups.append((idx, src, tgt, grp))
            else:
                failed_groups.append((idx, src, tgt, grp, err))

        print(f"\n{'=' * 70}")
        print("【ModelNet40 批量运行完成总结】")
        print(f"总组数: {total_groups}")
        print(f"成功: {len(success_groups)} 组")
        print(f"失败: {len(failed_groups)} 组")
        if success_groups:
            print("\n成功完成的组:")
            for idx, src, tgt, grp in sorted(success_groups):
                print(f"  [{idx}] {grp}: {src} → {tgt}")
        if failed_groups:
            print("\n失败的组:")
            for idx, src, tgt, grp, err in sorted(failed_groups):
                print(f"  [{idx}] {grp}: {src} → {tgt} | 错误: {err[:80]}...")
        try:
            aggregate_framework_analysis_from_dirs(base_dir=".")
        except Exception as e:
            print(f"[Warning] 汇总 ModelNet40 framework analysis 失败: {e}")
        print(f"\n{'=' * 70}")
        print("所有实验执行完毕！各组结果保存在独立目录中。")
        print(f"能量函数阶数符号: {ENERGY_ORDER_SYMBOL}")
        print(f"{'=' * 70}\n")
    else:
        gpu_ids = _select_gpu_ids()
        result = _run_modelnet_pair(TEST_INDEX, gpu_id=(gpu_ids[0] if gpu_ids else None))
        ok, idx, src, tgt, grp, err = result
        if not ok:
            raise RuntimeError(f"ModelNet40 Pair {idx} 失败: {err}")
