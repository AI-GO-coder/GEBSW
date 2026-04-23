import os
import cv2
import time
import threading
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
import torch.nn.functional as F
import lpips
from PIL import Image
from scipy.stats import wasserstein_distance, ttest_rel
from scipy import stats
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import warnings

warnings.filterwarnings('ignore')

# ---------------------- 全局配置与设备管理 ----------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

ENERGY_ORDER_SYMBOL = 'p'  # DTD使用'p'作为符号

# 【双GPU】自动检测可用GPU数量，上限为2块
NUM_GPUS = min(torch.cuda.device_count(), 2)
GPU_IDS = list(range(NUM_GPUS))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"使用设备: {DEVICE}")
if torch.cuda.is_available():
    for gid in GPU_IDS:
        print(f"GPU {gid}: {torch.cuda.get_device_name(gid)}")
        print(f"  显存总量: {torch.cuda.get_device_properties(gid).total_memory / 1024 ** 3:.2f} GB")

DPI = 600
BASE_DIR = "/home/zhouyan/EBGSW/Paired-DTD"
OUTPUT_BASE_DIR = "DTD_gradient-50L-10rep-350stepOpt-full"
N_PROJ = 50
N_REPEATS = 10
OPTIMIZATION_STEPS = 350

for subdir in ["images", "single_metrics", "combined_metrics", "stats", "gradient_flows"]:
    os.makedirs(os.path.join(OUTPUT_BASE_DIR, subdir), exist_ok=True)

# 注意：lpips_model 不再全局创建，改为每个GPU局部实例化以支持多卡并行

# ---------------------- 数据集配对配置 ----------------------
DTD_PAIRS = [
    ("1-braided_0182.jpg", "11-blotchy_0102.jpg", "T1_pair_1_to_11"),
    ("2-braided_0156.jpg", "22-blotchy_0042.jpg", "T1_pair_2_to_22"),
    ("3-braided_0075.jpg", "33-blotchy_0063.jpg", "T1_pair_3_to_33"),
    ("4-waffled_0109.jpg", "44-chequered_0055.jpg", "T2_pair_4_to_44"),
    ("5-waffled_0183.jpg", "55-chequered_0208.jpg", "T2_pair_5_to_55"),
    ("6-waffled_0110.jpg", "66-chequered_0179.jpg", "T2_pair_6_to_66"),
    ("7-bubbly_0101.jpg", "77-bumpy_0160.jpg", "T3_pair_7_to_77"),
    ("8-bubbly_0117.jpg", "88-bumpy_0127.jpg", "T3_pair_8_to_88"),
    ("9-bubbly_0061.jpg", "99-bumpy_0137.jpg", "T3_pair_9_to_99"),
    ("111-cobwebbed_0129.jpg", "1111-fibrous_0150.jpg", "T4_pair_111_to_1111"),
    ("222-cobwebbed_0139.jpg", "2222-fibrous_0184.jpg", "T4_pair_222_to_2222"),
    ("333-cobwebbed_0060.jpg", "3333-fibrous_0114.jpg", "T4_pair_333_to_3333"),
]

TYPE_INDEX = -1
TYPE_RANGES = [
    (0, 3),
    (3, 6),
    (6, 9),
    (9, 12),
]
USE_CUSTOM_PAIRS = False
CUSTOM_PAIRS = []
# USE_CUSTOM_PAIRS = True
# CUSTOM_PAIRS = [DTD_PAIRS[0]]


# ---------------------- GPU距离函数 ----------------------

def polynomial_projection_controlled_torch(X, degree=1, max_dim=256):
    batch_size, dim = X.shape
    if degree == 1:
        return X
    projections = [X]
    current_dim = dim
    for d in range(2, degree + 1):
        new_terms = dim
        if dim > 1:
            new_terms += dim * (dim - 1) // 2
        if current_dim + new_terms > max_dim and dim > 3:
            proj_mat = torch.randn(dim, max_dim // degree, device=X.device, dtype=X.dtype)
            proj_mat = F.normalize(proj_mat, p=2, dim=0)
            X_reduced = torch.matmul(X, proj_mat)
            return polynomial_projection_controlled_torch(X_reduced, degree, max_dim)
        projections.append(torch.pow(X, d))
        if dim > 1:
            for i in range(min(dim, 5)):
                for j in range(i + 1, min(dim, 5)):
                    cross_term = (torch.pow(X[:, i:i + 1], d // 2)) * (torch.pow(X[:, j:j + 1], d - d // 2))
                    projections.append(cross_term)
                    current_dim += 1
        current_dim += dim
    result = torch.cat(projections, dim=1)
    if result.shape[1] > max_dim:
        proj_mat = torch.randn(result.shape[1], max_dim, device=X.device, dtype=X.dtype)
        proj_mat = F.normalize(proj_mat, p=2, dim=0)
        result = torch.matmul(result, proj_mat)
    if degree > 1:
        result = (result - result.mean(dim=0, keepdim=True)) / (result.std(dim=0, keepdim=True) + 1e-6)
    return result


def rand_projections_torch(dim, num_projections, seed=None, device=DEVICE):
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        projections = torch.randn((num_projections, dim), device=device, generator=gen, dtype=torch.float32)
    else:
        projections = torch.randn((num_projections, dim), device=device, dtype=torch.float32)
    projections = F.normalize(projections, p=2, dim=1)
    return projections


def one_dimensional_wasserstein_p2_torch(X_proj, Y_proj, eps=1e-8):
    X_sorted, _ = torch.sort(X_proj, dim=0)
    Y_sorted, _ = torch.sort(Y_proj, dim=0)
    diff = torch.abs(X_sorted - Y_sorted)
    mse = torch.mean(torch.pow(diff, 2))
    return torch.pow(mse + eps, 1 / 2)


def compute_energy_weights_torch(wd_tensor, energy_type, energy_q, step_ratio=0.5):
    temp_start = 2.0
    temp_end = 0.1
    current_temp = temp_start * (temp_end / temp_start) ** step_ratio
    wd_tensor = torch.clamp(wd_tensor, min=1e-12)
    if energy_type == "exp":
        logits = wd_tensor / (wd_tensor.mean() + 1e-12) / current_temp
        weights = F.softmax(logits, dim=0)
    elif energy_type == "poly":
        logits = torch.pow(wd_tensor, energy_q) / current_temp
        weights = F.softmax(logits, dim=0)
    else:
        weights = torch.ones_like(wd_tensor) / len(wd_tensor)
    weights = torch.clamp(weights, max=0.3)
    weights = weights / weights.sum()
    return weights


class DistanceFunctionGPU:
    def __init__(self, name, degree=1, energy_type=None, energy_q=1,
                 n_projections=50, proj_seed=42, device=DEVICE):
        self.name = name
        self.degree = degree
        self.energy_type = energy_type
        self.energy_q = energy_q
        self.n_projections = n_projections
        self.proj_seed = proj_seed
        self.device = device
        self.max_dim = 64 if degree == 1 else (128 if degree == 3 else 256)
        self.theta_list = None

    def _get_projections(self, input_dim):
        if self.theta_list is None or self.theta_list.shape[1] != input_dim:
            n_proj = self.n_projections if self.degree <= 3 else self.n_projections // 2
            self.theta_list = rand_projections_torch(
                input_dim, n_proj, self.proj_seed, self.device
            )
        return self.theta_list

    def __call__(self, source_img, target_img):
        if isinstance(source_img, np.ndarray):
            source_samples = torch.from_numpy(
                source_img.reshape(-1, 3).astype(np.float32)
            ).to(self.device) / 255.0
        else:
            source_samples = source_img.reshape(-1, 3).float().to(self.device) / 255.0

        if isinstance(target_img, np.ndarray):
            target_samples = torch.from_numpy(
                target_img.reshape(-1, 3).astype(np.float32)
            ).to(self.device) / 255.0
        else:
            target_samples = target_img.reshape(-1, 3).float().to(self.device) / 255.0

        X_proj = polynomial_projection_controlled_torch(
            source_samples, self.degree, self.max_dim
        )
        Y_proj = polynomial_projection_controlled_torch(
            target_samples, self.degree, self.max_dim
        )
        theta = self._get_projections(X_proj.shape[1])
        X_proj_1d = torch.matmul(X_proj, theta.T)
        Y_proj_1d = torch.matmul(Y_proj, theta.T)
        wd_list = []
        for i in range(X_proj_1d.shape[1]):
            wd = one_dimensional_wasserstein_p2_torch(
                X_proj_1d[:, i], Y_proj_1d[:, i]
            )
            wd_list.append(wd)
        wd_tensor = torch.stack(wd_list)
        if self.energy_type is not None:
            weights = compute_energy_weights_torch(
                wd_tensor, self.energy_type, self.energy_q
            )
            distance = torch.sum(wd_tensor * weights)
        else:
            distance = torch.mean(wd_tensor)
        return distance


def create_distance_functions(base_seed=2024, device=None):
    """
    【双GPU】接受device参数，支持在指定GPU上创建距离函数实例。
    """
    if device is None:
        device = DEVICE
    distances = {}
    sym = ENERGY_ORDER_SYMBOL

    distances["GSW-Proj-poly(q=1) (SW-Baseline)"] = DistanceFunctionGPU(
        "GSW-Proj-poly(q=1) (SW-Baseline)",
        degree=1, energy_type=None, n_projections=N_PROJ, proj_seed=base_seed + 1,
        device=device
    )
    distances["GSW-Proj-poly(q=3) (GSW-Baseline)"] = DistanceFunctionGPU(
        "GSW-Proj-poly(q=3) (GSW-Baseline)",
        degree=3, energy_type=None, n_projections=N_PROJ, proj_seed=base_seed + 2,
        device=device
    )
    distances["GSW-Proj-poly(q=5) (GSW-Baseline)"] = DistanceFunctionGPU(
        "GSW-Proj-poly(q=5) (GSW-Baseline)",
        degree=5, energy_type=None, n_projections=N_PROJ // 2, proj_seed=base_seed + 3,
        device=device
    )
    distances["GEBSW-f^*_e-Proj-poly(q=1) (EBSW-Baseline)"] = DistanceFunctionGPU(
        "GEBSW-f^*_e-Proj-poly(q=1) (EBSW-Baseline)",
        degree=1, energy_type='exp', energy_q=1, n_projections=N_PROJ, proj_seed=base_seed + 100,
        device=device
    )
    distances["GEBSW-f^*_e-Proj-poly(q=3)"] = DistanceFunctionGPU(
        "GEBSW-f^*_e-Proj-poly(q=3)",
        degree=3, energy_type='exp', energy_q=1, n_projections=N_PROJ, proj_seed=base_seed + 101,
        device=device
    )
    distances["GEBSW-f^*_e-Proj-poly(q=5)"] = DistanceFunctionGPU(
        "GEBSW-f^*_e-Proj-poly(q=5)",
        degree=5, energy_type='exp', energy_q=1, n_projections=N_PROJ // 2, proj_seed=base_seed + 102,
        device=device
    )
    for p in [1, 2, 3, 4]:
        is_baseline = " (EBSW-Baseline)"
        key_name = f"GEBSW-f^*_{sym}={p}-Proj-poly(q=1){is_baseline}"
        distances[key_name] = DistanceFunctionGPU(
            key_name,
            degree=1, energy_type='poly', energy_q=p, n_projections=N_PROJ, proj_seed=base_seed + 200 + p,
            device=device
        )
    for p in [1, 2, 3, 4]:
        key_name = f"GEBSW-f^*_{sym}={p}-Proj-poly(q=3)"
        distances[key_name] = DistanceFunctionGPU(
            key_name,
            degree=3, energy_type='poly', energy_q=p, n_projections=N_PROJ, proj_seed=base_seed + 300 + p,
            device=device
        )
    for p in [1, 2, 3, 4]:
        key_name = f"GEBSW-f^*_{sym}={p}-Proj-poly(q=5)"
        distances[key_name] = DistanceFunctionGPU(
            key_name,
            degree=5, energy_type='poly', energy_q=p, n_projections=N_PROJ // 2, proj_seed=base_seed + 400 + p,
            device=device
        )
    return distances


# ---------------------- 颜色迁移与评估 ----------------------

def to_scalar(val):
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().item()
    elif isinstance(val, np.ndarray):
        return val.item() if val.size == 1 else float(val)
    elif isinstance(val, (float, int)):
        return float(val)
    else:
        return float(val)


def color_transfer_optimization(source_img, target_img, distance_func,
                                n_steps=OPTIMIZATION_STEPS, lr=0.01,
                                track_gradient=True, device=None):
    """
    基于自动微分的颜色迁移优化。
    【双GPU】新增device参数，确保张量创建在指定GPU上。
    【梯度流】仅记录 Step 与 Loss，用于绘制收敛曲线。
    """
    if device is None:
        device = DEVICE

    current_rgb = torch.from_numpy(source_img.astype(np.float32)).to(device).requires_grad_(True)
    target_torch = torch.from_numpy(target_img.astype(np.float32)).to(device)

    optimizer = torch.optim.Adam([current_rgb], lr=lr)
    best_loss = float('inf')
    best_result = current_rgb.detach().clone()

    gradient_flow = []

    for step in range(n_steps):
        optimizer.zero_grad()

        rgb_clamped = torch.clamp(current_rgb, 0, 255)
        loss = distance_func(rgb_clamped, target_torch)
        loss_val = loss.item()

        if loss_val < best_loss:
            best_loss = loss_val
            best_result = rgb_clamped.detach().clone()

        loss.backward()

        if track_gradient:
            gradient_flow.append({
                'Step': step,
                'Loss': loss_val,
            })

        if step % 50 == 0:
            print(f"  Step {step:3d}: loss={loss_val:.4f}")

        optimizer.step()

        with torch.no_grad():
            current_rgb.clamp_(0, 255)

    result_rgb = best_result.cpu().numpy()
    result_rgb = np.clip(result_rgb, 0, 255).astype(np.uint8)
    final_w2 = to_scalar(distance_func(result_rgb, target_img))

    if track_gradient:
        return result_rgb, final_w2, gradient_flow
    return result_rgb, final_w2


def calculate_lpips(img1, img2, lpips_model, device=None):
    """
    【双GPU】接受局部lpips_model与device参数，支持多卡并行评估。
    """
    if device is None:
        device = DEVICE
    try:
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        img1_tensor = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0 * 2 - 1
        img2_tensor = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0 * 2 - 1
        with torch.no_grad():
            dist = lpips_model(img1_tensor.to(device), img2_tensor.to(device))
        return to_scalar(dist)
    except Exception as e:
        print(f"LPIPS计算错误: {e}")
        return np.nan


def compute_color_histogram_distance(img1, img2):
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    return wasserstein_distance(hist1, hist2)


# ---------------------- 统计检验功能 ----------------------

def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)
    if pooled_std == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std


def interpret_cohen_d(d):
    abs_d = abs(d)
    if abs_d < 0.2:
        return "可忽略(Negligible)"
    elif abs_d < 0.5:
        return "小(Small)"
    elif abs_d < 0.8:
        return "中(Medium)"
    else:
        return "大(Large)"


def paired_t_test_stats(values, baseline_values):
    if len(values) != len(baseline_values) or len(values) < 2:
        return {
            't_statistic': np.nan, 'p_value': np.nan, 'significant_05': False,
            'significant_01': False, 'mean_diff': np.nan, 'std_diff': np.nan,
            'cohen_d': np.nan, 'effect_size': 'N/A', 'percent_improvement': np.nan
        }
    t_stat, p_val = ttest_rel(values, baseline_values)
    diff = np.array(values) - np.array(baseline_values)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    cohen_dz = mean_diff / std_diff if std_diff > 0 else 0.0
    percent_improve = ((np.mean(baseline_values) - np.mean(values)) / np.mean(baseline_values) * 100) if np.mean(
        baseline_values) != 0 else 0.0
    return {
        't_statistic': t_stat, 'p_value': p_val, 'significant_05': p_val < 0.05,
        'significant_01': p_val < 0.01, 'mean_diff': mean_diff, 'std_diff': std_diff,
        'cohen_d': cohen_dz, 'effect_size': interpret_cohen_d(cohen_dz),
        'percent_improvement': percent_improve
    }


def compute_descriptive_stats(values):
    if not values or len(values) == 0:
        return {
            'Mean': np.nan, 'Std': np.nan, 'Min': np.nan, 'Max': np.nan,
            'Median': np.nan, 'Q1': np.nan, 'Q3': np.nan, 'IQR': np.nan, 'CV': np.nan
        }
    arr = np.array(values)
    q1, median, q3 = np.percentile(arr, [25, 50, 75])
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    return {
        'Mean': mean, 'Std': std, 'Min': np.min(arr), 'Max': np.max(arr),
        'Median': median, 'Q1': q1, 'Q3': q3, 'IQR': q3 - q1,
        'CV': (std / mean * 100) if mean != 0 else 0.0
    }


# ---------------------- 梯度流保存功能 ----------------------

def save_gradient_flows(pair_id, gradient_flows, repeat_idx=0):
    """
    将单个配对上全部距离变体的梯度流合并保存到一个 Excel 文件的一个 Sheet 中。
    宽格式：第一列为 Step，后续每列为一个变体的 Loss 轨迹。
    """
    gf_dir = os.path.join(OUTPUT_BASE_DIR, "gradient_flows")
    os.makedirs(gf_dir, exist_ok=True)
    out_path = os.path.join(gf_dir, f"{pair_id}_gradient_flow_Round{repeat_idx + 1}.xlsx")

    df_merged = None
    for dist_name, records in gradient_flows.items():
        if not records:
            continue
        df_temp = pd.DataFrame(records)[['Step', 'Loss']]
        df_temp = df_temp.rename(columns={'Loss': dist_name})
        if df_merged is None:
            df_merged = df_temp
        else:
            df_merged = df_merged.merge(df_temp, on='Step', how='outer')

    if df_merged is not None and not df_merged.empty:
        df_merged = df_merged.sort_values('Step').reset_index(drop=True)
        df_merged.to_excel(out_path, sheet_name="Gradient_Flow", index=False)
        print(f"  [梯度流] 已保存: {out_path}")
    else:
        print(f"  [梯度流] 无数据，跳过保存: {pair_id}")


# ---------------------- 【双GPU】主实验流程 ----------------------

def run_experiment_for_pair(source_name, target_name, pair_id, repeat_idx=0):
    source_path = os.path.join(BASE_DIR, source_name)
    target_path = os.path.join(BASE_DIR, target_name)
    print(f"\n{'=' * 60}")
    print(f"处理配对: {pair_id} | 第 {repeat_idx + 1} 轮")
    print(f"源图像: {source_name}")
    print(f"目标图像: {target_name}")
    print(f"{'=' * 60}")

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"源图像不存在: {source_path}")
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"目标图像不存在: {target_path}")

    source_img = cv2.imread(source_path)
    target_img = cv2.imread(target_path)
    if source_img is None or target_img is None:
        raise ValueError("图像加载失败")
    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    h, w = source_img.shape[:2]
    target_img = cv2.resize(target_img, (w, h))

    round_seed = 2024 + repeat_idx * 1000

    # 【双GPU】为每块GPU创建独立的距离函数实例（绑定到对应设备）
    gpu_dist_funcs = {}
    for gid in GPU_IDS:
        dev = torch.device(f"cuda:{gid}")
        gpu_dist_funcs[gid] = create_distance_functions(base_seed=round_seed, device=dev)

    # 收集全部变体并按数量平均分配到各GPU
    all_variant_items = []
    for gid in GPU_IDS:
        for name, dist_func in gpu_dist_funcs[gid].items():
            all_variant_items.append((gid, name, dist_func))

    n_total = len(all_variant_items)
    n_per_gpu = (n_total + NUM_GPUS - 1) // NUM_GPUS
    gpu_tasks = {gid: [] for gid in GPU_IDS}
    for idx, (gid, name, dist_func) in enumerate(all_variant_items):
        target_gid = GPU_IDS[idx // n_per_gpu] if NUM_GPUS > 1 else GPU_IDS[0]
        gpu_tasks[target_gid].append((name, dist_func))

    # 共享结果容器与线程锁
    results = {}
    gradient_flows = {}
    lock = threading.Lock()

    def process_subset(gpu_id, subset):
        """
        在指定GPU上处理分配到的变体子集。
        """
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        local_lpips = lpips.LPIPS(net='alex').to(device)
        local_lpips.eval()

        local_results = {}
        local_gf = {}

        for idx, (name, dist_func) in enumerate(subset):
            try:
                start_time = time.time()
                transferred_img, final_w2, gf_records = color_transfer_optimization(
                    source_img, target_img, dist_func,
                    track_gradient=True, device=device
                )
                elapsed = time.time() - start_time
                psnr_val = psnr(target_img, transferred_img, data_range=255)
                ssim_val = ssim(target_img, transferred_img, data_range=255, channel_axis=2)
                lpips_val = calculate_lpips(target_img, transferred_img, local_lpips, device)
                hist_dist = compute_color_histogram_distance(transferred_img, target_img)

                local_results[name] = {
                    'W_2': to_scalar(final_w2),
                    'PSNR': to_scalar(psnr_val),
                    'SSIM': to_scalar(ssim_val),
                    'LPIPS': to_scalar(lpips_val),
                    'Hist_W2': to_scalar(hist_dist),
                    'Time': to_scalar(elapsed),
                    'Image': transferred_img
                }
                local_gf[name] = gf_records

                print(f"[GPU{gpu_id}] {name:45s} | W2: {final_w2:.4f} | PSNR: {psnr_val:.2f} | "
                      f"SSIM: {ssim_val:.4f} | Time: {elapsed:.2f}s")

                if idx % 5 == 0:
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"[GPU{gpu_id}] 错误处理 {name}: {e}")
                import traceback
                traceback.print_exc()
                local_results[name] = {
                    'W_2': np.nan, 'PSNR': np.nan, 'SSIM': np.nan,
                    'LPIPS': np.nan, 'Hist_W2': np.nan, 'Time': np.nan, 'Image': None
                }
                local_gf[name] = []

        # 线程安全地合并结果
        with lock:
            results.update(local_results)
            gradient_flows.update(local_gf)

    # 【双GPU】启动多线程并行处理；单GPU时退化为串行
    if NUM_GPUS > 1:
        threads = []
        for gid in GPU_IDS:
            t = threading.Thread(target=process_subset, args=(gid, gpu_tasks[gid]))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
    else:
        process_subset(GPU_IDS[0], gpu_tasks[GPU_IDS[0]])

    # 串行保存图像，避免磁盘写入竞争
    for name, res in results.items():
        if res.get('Image') is not None:
            safe_name = name.replace("*", "_").replace("{", "").replace("}", "").replace("=", "-")
            save_dir = os.path.join(OUTPUT_BASE_DIR, "images", pair_id, f"Round_{repeat_idx + 1}")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{safe_name}.png")
            Image.fromarray(res['Image']).save(save_path, dpi=(DPI, DPI))

    save_gradient_flows(pair_id, gradient_flows, repeat_idx)
    return results, source_img, target_img


def safe_is_nan(val):
    if isinstance(val, torch.Tensor):
        return torch.isnan(val).any().item()
    try:
        return np.isnan(val)
    except (TypeError, ValueError):
        return False


def safe_to_float(val):
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().item()
    elif isinstance(val, np.ndarray):
        return val.item() if val.size == 1 else float(val)
    elif isinstance(val, (int, float)):
        return float(val)
    return np.nan


def save_all_results(all_results):
    metrics = ['W_2', 'PSNR', 'SSIM', 'LPIPS', 'Hist_W2', 'Time']
    sym = ENERGY_ORDER_SYMBOL
    baseline_mapping = {
        'SW_Baseline': "GSW-Proj-poly(q=1) (SW-Baseline)",
        'GSW3_Baseline': "GSW-Proj-poly(q=3) (GSW-Baseline)",
        'GSW5_Baseline': "GSW-Proj-poly(q=5) (GSW-Baseline)",
        'EBSW_Exp_Baseline': "GEBSW-f^*_e-Proj-poly(q=1) (EBSW-Baseline)",
        'EBSW_Poly1_Baseline': f"GEBSW-f^*_{sym}=1-Proj-poly(q=1) (EBSW-Baseline)"
    }

    summary_data = []
    for pair_id, pair_data in all_results.items():
        for metric in metrics:
            for dist_name in pair_data['Round_1'].keys():
                row = {'Pair_ID': pair_id, 'Distance_Function': dist_name, 'Metric': metric}
                values = []
                for round_idx in range(N_REPEATS):
                    round_key = f'Round_{round_idx + 1}'
                    if round_key in pair_data and dist_name in pair_data[round_key]:
                        raw_val = pair_data[round_key][dist_name][metric]
                        val = safe_to_float(raw_val)
                        row[f'Round_{round_idx + 1}'] = val
                        if not safe_is_nan(val):
                            values.append(val)
                if metric != 'Time' and len(values) > 0:
                    row['Mean'] = np.nanmean(values)
                    row['Std'] = np.nanstd(values, ddof=1)
                summary_data.append(row)

    df_summary = pd.DataFrame(summary_data)
    excel_path = os.path.join(OUTPUT_BASE_DIR, "combined_metrics", "all_pairs_all_metrics.xlsx")
    df_summary.to_excel(excel_path, index=False)
    print(f"\n已保存汇总结果: {excel_path}")

    print(f"\n{'=' * 60}")
    print("正在生成详细统计分析报告...")
    print(f"{'=' * 60}")
    stats_dir = os.path.join(OUTPUT_BASE_DIR, "stats")

    desc_stats_all = []
    ranking_data = {metric: [] for metric in metrics if metric != 'Time'}
    for pair_id, pair_data in all_results.items():
        for dist_name in pair_data['Round_1'].keys():
            row_base = {'Pair_ID': pair_id, 'Distance_Function': dist_name}
            for metric in metrics:
                values = []
                for round_idx in range(N_REPEATS):
                    round_key = f'Round_{round_idx + 1}'
                    if round_key in pair_data and dist_name in pair_data[round_key]:
                        raw_val = pair_data[round_key][dist_name][metric]
                        val = safe_to_float(raw_val)
                        if not safe_is_nan(val):
                            values.append(val)
                if len(values) > 0:
                    desc_stats = compute_descriptive_stats(values)
                    for stat_name, stat_val in desc_stats.items():
                        row_base[f"{metric}_{stat_name}"] = stat_val
                    if metric != 'Time':
                        ranking_data[metric].append({
                            'Pair_ID': pair_id, 'Distance_Function': dist_name,
                            'Mean': np.mean(values), 'Std': np.std(values, ddof=1)
                        })
                else:
                    for stat_name in ['Mean', 'Std', 'Min', 'Max', 'Median', 'Q1', 'Q3', 'IQR', 'CV']:
                        row_base[f"{metric}_{stat_name}"] = np.nan
            desc_stats_all.append(row_base)

    df_desc = pd.DataFrame(desc_stats_all)
    desc_path = os.path.join(stats_dir, "01_描述性统计_Descriptive_Statistics.xlsx")
    df_desc.to_excel(desc_path, index=False)
    print(f"✓ 描述性统计已保存: {desc_path}")

    with pd.ExcelWriter(os.path.join(stats_dir, "02_指标排名_Rankings.xlsx"), engine='openpyxl') as writer:
        for metric, data_list in ranking_data.items():
            if not data_list:
                continue
            df_rank = pd.DataFrame(data_list)
            ascending = metric in ['W_2', 'LPIPS', 'Hist_W2', 'Time']
            df_rank = df_rank.sort_values('Mean', ascending=ascending)
            df_rank['Rank'] = range(1, len(df_rank) + 1)
            best_value = df_rank['Mean'].iloc[0] if ascending else df_rank['Mean'].iloc[-1]
            df_rank['Gap_from_Best_%'] = ((df_rank['Mean'] - best_value) / best_value * 100).abs()
            safe_metric = metric.replace('/', '_').replace('\\', '_')
            df_rank.to_excel(writer, sheet_name=safe_metric[:31], index=False)
    print(f"✓ 指标排名已保存")

    paired_data = {}
    for pair_id, pair_data in all_results.items():
        paired_data[pair_id] = {}
        for dist_name in pair_data['Round_1'].keys():
            paired_data[pair_id][dist_name] = {}
            for metric in metrics:
                values = []
                for round_idx in range(N_REPEATS):
                    round_key = f'Round_{round_idx + 1}'
                    if round_key in pair_data and dist_name in pair_data[round_key]:
                        raw_val = pair_data[round_key][dist_name][metric]
                        val = safe_to_float(raw_val)
                        if not safe_is_nan(val):
                            values.append(val)
                paired_data[pair_id][dist_name][metric] = values

    t_test_results = []
    comparison_bases = {
        'vs_SW_Baseline': baseline_mapping['SW_Baseline'],
        'vs_GSW3_Baseline': baseline_mapping['GSW3_Baseline'],
        'vs_EBSW_Exp_Baseline': baseline_mapping['EBSW_Exp_Baseline']
    }
    for pair_id in paired_data.keys():
        for dist_name in paired_data[pair_id].keys():
            if dist_name in baseline_mapping.values():
                continue
            for metric in metrics:
                if metric == 'Time':
                    continue
                values = paired_data[pair_id][dist_name].get(metric, [])
                if len(values) < 2:
                    continue
                row_test = {
                    'Pair_ID': pair_id, 'Distance_Function': dist_name,
                    'Metric': metric, 'N_Samples': len(values)
                }
                for comp_name, baseline_name in comparison_bases.items():
                    if baseline_name not in paired_data[pair_id]:
                        continue
                    baseline_values = paired_data[pair_id][baseline_name].get(metric, [])
                    if len(baseline_values) != len(values):
                        continue
                    stats_result = paired_t_test_stats(values, baseline_values)
                    prefix = comp_name
                    row_test[f"{prefix}_t_stat"] = stats_result['t_statistic']
                    row_test[f"{prefix}_p_value"] = stats_result['p_value']
                    row_test[f"{prefix}_sig_05"] = stats_result['significant_05']
                    row_test[f"{prefix}_sig_01"] = stats_result['significant_01']
                    row_test[f"{prefix}_mean_diff"] = stats_result['mean_diff']
                    row_test[f"{prefix}_cohen_d"] = stats_result['cohen_d']
                    row_test[f"{prefix}_effect_size"] = stats_result['effect_size']
                    row_test[f"{prefix}_improve_%"] = stats_result['percent_improvement']
                t_test_results.append(row_test)

    if t_test_results:
        df_ttest = pd.DataFrame(t_test_results)
        ttest_path = os.path.join(stats_dir, "03_配对t检验_Paired_TTest_vs_Baselines.xlsx")
        df_ttest.to_excel(ttest_path, index=False)
        print(f"✓ 配对t检验结果已保存")

    aggregate_stats = {}
    for pair_id in paired_data.keys():
        for dist_name in paired_data[pair_id].keys():
            if dist_name not in aggregate_stats:
                aggregate_stats[dist_name] = {metric: [] for metric in metrics if metric != 'Time'}
            for metric in metrics:
                if metric == 'Time':
                    continue
                values = paired_data[pair_id][dist_name].get(metric, [])
                aggregate_stats[dist_name][metric].extend(values)

    aggregate_rows = []
    for dist_name, metric_data in aggregate_stats.items():
        row = {'Distance_Function': dist_name, 'Total_Samples': N_REPEATS * len(all_results)}
        for metric, all_values in metric_data.items():
            if len(all_values) > 0:
                desc = compute_descriptive_stats(all_values)
                for stat_name, stat_val in desc.items():
                    row[f"{metric}_{stat_name}"] = stat_val
        aggregate_rows.append(row)

    df_aggregate = pd.DataFrame(aggregate_rows)
    agg_path = os.path.join(stats_dir, "04_跨配对汇总统计_Aggregate_Statistics.xlsx")
    df_aggregate.to_excel(agg_path, index=False)
    print(f"✓ 跨配对汇总统计已保存")

    best_variants = {metric: {} for metric in ranking_data.keys()}
    for pair_id in all_results.keys():
        for metric in best_variants.keys():
            best_val = float('inf') if metric in ['W_2', 'LPIPS', 'Hist_W2', 'Time'] else float('-inf')
            best_dist = None
            for dist_name in paired_data[pair_id].keys():
                values = paired_data[pair_id][dist_name].get(metric, [])
                if len(values) == 0:
                    continue
                mean_val = np.mean(values)
                if metric in ['W_2', 'LPIPS', 'Hist_W2', 'Time']:
                    if mean_val < best_val:
                        best_val = mean_val
                        best_dist = dist_name
                else:
                    if mean_val > best_val:
                        best_val = mean_val
                        best_dist = dist_name
            best_variants[metric][pair_id] = {'Best_Variant': best_dist, 'Best_Value': best_val}

    variant_wins = {metric: {} for metric in best_variants.keys()}
    for metric, pair_dict in best_variants.items():
        for pair_id, info in pair_dict.items():
            variant = info['Best_Variant']
            if variant not in variant_wins[metric]:
                variant_wins[metric][variant] = 0
            variant_wins[metric][variant] += 1

    win_rows = []
    for metric, variant_dict in variant_wins.items():
        for variant, count in variant_dict.items():
            total_pairs = len(all_results)
            win_rows.append({
                'Metric': metric, 'Distance_Function': variant, 'Win_Count': count,
                'Win_Rate_%': (count / total_pairs * 100), 'Total_Pairs': total_pairs
            })

    df_wins = pd.DataFrame(win_rows)
    if len(df_wins) > 0:
        df_wins = df_wins.sort_values(['Metric', 'Win_Count'], ascending=[True, False])
        wins_path = os.path.join(stats_dir, "05_最佳变体统计_Best_Variant_Wins.xlsx")
        df_wins.to_excel(wins_path, index=False)
        print(f"✓ 最佳变体统计已保存")

    print(f"\n{'=' * 60}")
    print(f"统计分析报告生成完成！保存在: {stats_dir}")
    print(f"{'=' * 60}")

    for pair_id in all_results.keys():
        pair_metrics = {}
        for metric in metrics:
            metric_data = []
            for dist_name in all_results[pair_id]['Round_1'].keys():
                row = {'Distance_Function': dist_name}
                values = []
                for round_idx in range(N_REPEATS):
                    round_key = f'Round_{round_idx + 1}'
                    raw_val = all_results[pair_id][round_key][dist_name][metric]
                    val = safe_to_float(raw_val)
                    row[f'Round_{round_idx + 1}'] = val
                    if not safe_is_nan(val):
                        values.append(val)
                if metric != 'Time' and len(values) > 0:
                    row['Mean'] = np.nanmean(values)
                    row['Std'] = np.nanstd(values, ddof=1)
                metric_data.append(row)
            pair_metrics[metric] = pd.DataFrame(metric_data)
        excel_path = os.path.join(OUTPUT_BASE_DIR, "single_metrics", f"{pair_id}_results.xlsx")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for metric, df in pair_metrics.items():
                df.to_excel(writer, sheet_name=metric, index=False)
        print(f"已保存配对结果: {excel_path}")


if __name__ == "__main__":
    if USE_CUSTOM_PAIRS:
        selected_pairs = CUSTOM_PAIRS
        type_name = "Custom"
    else:
        if TYPE_INDEX == -1:
            selected_pairs = DTD_PAIRS
            type_name = "All_Types"
        elif TYPE_INDEX < 0 or TYPE_INDEX >= len(TYPE_RANGES):
            print(f"错误：TYPE_INDEX {TYPE_INDEX} 超出范围")
            exit(1)
        else:
            start_idx, end_idx = TYPE_RANGES[TYPE_INDEX]
            selected_pairs = DTD_PAIRS[start_idx:end_idx]
            type_names = ["T1_banded_to_porous", "T2_chequered_to_waffled",
                          "T3_fibrous_to_crystalline", "T4_smeared_to_perforated"]
            type_name = type_names[TYPE_INDEX]

    print("开始DTD数据集颜色迁移实验【双GPU并行加速版】")
    print(f"当前运行类型: {type_name} (索引: {TYPE_INDEX})")
    print(f"该类型包含 {len(selected_pairs)} 对图像，每对重复 {N_REPEATS} 次")
    print(f"并行策略: 每Pair内部的{len(create_distance_functions())}个变体平均分配到 {NUM_GPUS} 块GPU")

    all_results = {}
    for source_name, target_name, pair_id in selected_pairs:
        all_results[pair_id] = {}
        for i in range(N_REPEATS):
            results, src_img, tgt_img = run_experiment_for_pair(
                source_name, target_name, pair_id, i
            )
            all_results[pair_id][f'Round_{i + 1}'] = results
            if i == 0:
                ref_dir = os.path.join(OUTPUT_BASE_DIR, "images", pair_id, "reference")
                os.makedirs(ref_dir, exist_ok=True)
                Image.fromarray(src_img).save(os.path.join(ref_dir, "source.png"), dpi=(DPI, DPI))
                Image.fromarray(tgt_img).save(os.path.join(ref_dir, "target.png"), dpi=(DPI, DPI))
            # 【双GPU】每轮结束后清空所有GPU缓存
            for gid in GPU_IDS:
                with torch.cuda.device(gid):
                    torch.cuda.empty_cache()

    save_all_results(all_results)

    print("\n" + "=" * 60)
    print("实验完成！结果已保存到:", OUTPUT_BASE_DIR)
    print(f"运行类型: {type_name}")
    for gid in GPU_IDS:
        with torch.cuda.device(gid):
            print(f"GPU{gid}最终显存使用: {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB")
    print("目录结构：")
    print(f"  - {OUTPUT_BASE_DIR}/images/          : 迁移后的图像")
    print(f"  - {OUTPUT_BASE_DIR}/single_metrics/  : 每个配对的详细指标")
    print(f"  - {OUTPUT_BASE_DIR}/combined_metrics/: 所有配对的汇总指标")
    print(f"  - {OUTPUT_BASE_DIR}/stats/           : 统计分析报告（5个Excel文件）")
    print(f"  - {OUTPUT_BASE_DIR}/gradient_flows/  : 各配对的梯度流记录（单Sheet宽格式）")
    print("=" * 60)