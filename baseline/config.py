"""
FAS-RIS基线系统配置参数
"""
import math

# ==================== 系统参数 ====================
# 用户数量
NUM_USERS = 3  # 1个主用户 + 2个干扰用户
NUM_PRIMARY_USERS = 1
NUM_INTERFERING_USERS = 2

# FAS（流体天线系统）参数
NUM_FAS = 1  # 基站处FAS天线数量
FAS_AREA_SIZE = 1.0  # FAS可移动区域：1m x 1m正方形
FAS_MIN_DISTANCE = 0.1  # FAS天线间最小距离 (m)

# RIS（可重构智能表面）参数
NUM_RIS_ELEMENTS = 64  # RIS反射单元数量
RIS_PHASE_BITS = 3  # 相位量化比特数 (2^3 = 8个量化级)
RIS_PHASE_LEVELS = 2 ** RIS_PHASE_BITS

# ==================== 信道参数 ====================
# 频率和波长
CARRIER_FREQUENCY = 1e9  # 1 GHz
SPEED_OF_LIGHT = 3e8  # m/s
WAVELENGTH = SPEED_OF_LIGHT / CARRIER_FREQUENCY

# 路径损耗参数
PATH_LOSS_EXPONENT = 3  # 路径损耗指数
PATH_LOSS_CONSTANT = (WAVELENGTH / (4 * math.pi)) ** 2  # 自由空间路径损耗常数

# Nakagami衰落参数
NAKAGAMI_M = 2.0  # Nakagami m参数 (m=1为Rayleigh, m>1更好)
NAKAGAMI_OMEGA = 1.0  # Nakagami omega参数 (平均功率)

# Rician衰落参数（用户到RIS链路）
RICIAN_K_FACTOR = 3.0  # Rician K因子 (LOS到NLOS功率比)

# ==================== 传输参数 ====================
TRANSMIT_POWER = 1.0  # 发射功率 (W)
NOISE_POWER = 1e-12  # 噪声功率 (W) - 1 pW
NOISE_POWER_DBM = 10 * math.log10(NOISE_POWER * 1000)  # 噪声功率 (dBm)

# ==================== 优化参数 ====================
# FAS位置优化
FAS_GRADIENT_STEP_SIZE = 0.01  # FAS位置梯度下降步长
FAS_CONVERGENCE_THRESHOLD = 1e-4  # FAS优化收敛阈值

# 交替优化（基于凸近似）
MAX_ITERATIONS = 100  # 交替优化最大迭代次数
CONVERGENCE_THRESHOLD = 1e-3  # 总体收敛阈值
CONVEX_APPROXIMATION_ENABLED = True  # 启用凸近似优化

# ==================== 仿真参数 ====================
# 蒙特卡洛仿真配置
USE_MONTE_CARLO = False  # 是否使用蒙特卡洛方法 (True: 多次试验, False: 单次试验)
NUM_MONTE_CARLO_TRIALS = 20  # 蒙特卡洛试验次数 (仅当USE_MONTE_CARLO=True时有效)
RANDOM_SEED = 42  # 随机种子

# ==================== 几何参数 ====================
# 基站位置（固定在原点）
BS_POSITION = [0.0, 0.0, 0.0]  # [x, y, z] 单位：米

# RIS位置（固定，以基站为中心半径10m的圆上）
RIS_POSITION = [10.0, 0.0, 3.0]  # [x, y, z] 单位：米

# 用户位置（随机生成，距离基站至少20m）
USER_AREA_SIZE = 50.0  # 用户部署区域：50m x 50m（足够大以满足距离要求）
USER_MIN_DISTANCE_FROM_BS = 20.0  # 用户距离基站的最小距离：20m

# ==================== 输出参数 ====================
SAVE_RESULTS = True
RESULTS_DIR = "results"
PLOT_RESULTS = True

