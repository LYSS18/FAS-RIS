"""
FAS-RIS基线系统主仿真脚本
基于凸近似的交替优化
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from config import *
from channel_model import MultiUserChannelModel
from convex_optimization import ConvexOptimizer


def run_single_trial(trial_id):
    """运行单次仿真试验"""
    num_trials = NUM_MONTE_CARLO_TRIALS if USE_MONTE_CARLO else 1
    print(f"\n{'='*60}")
    print(f"试验 {trial_id + 1}/{num_trials}")
    print(f"{'='*60}")

    # 初始化信道模型和优化器
    channel_model = MultiUserChannelModel()
    optimizer = ConvexOptimizer()
    print(f"优化方法: 基于凸近似的交替优化 (Convex Approximation)")

    # 生成用户位置
    user_positions = channel_model.generate_user_positions(NUM_USERS, USER_AREA_SIZE)
    primary_user_pos = user_positions[0]
    interfering_users_pos = user_positions[1:]

    print(f"主用户位置: {primary_user_pos}")
    print(f"干扰用户位置: {interfering_users_pos}")
    print(f"RIS位置: {RIS_POSITION}")
    print(f"基站位置: {BS_POSITION}")

    # 运行基于凸近似的交替优化
    fas_pos, ris_phases, history = optimizer.alternating_optimization_convex(
        primary_user_pos, interfering_users_pos, RIS_POSITION, BS_POSITION)

    # 计算最终性能
    final_sinr, final_primary_power, final_interference_power = optimizer.calculate_objective(
        fas_pos, ris_phases, primary_user_pos, interfering_users_pos, RIS_POSITION, BS_POSITION)

    final_capacity = channel_model.channel_model.calculate_capacity(final_sinr)

    print(f"\n优化结果:")
    print(f"最终FAS位置: {fas_pos}")
    print(f"最终SINR: {final_sinr:.4f} ({10*np.log10(final_sinr):.2f} dB)")
    print(f"最终容量: {final_capacity:.4f} bits/s/Hz")
    print(f"主用户功率: {final_primary_power:.6e}")
    print(f"干扰功率: {final_interference_power:.6e}")

    return {
        'trial_id': trial_id,
        'fas_position': fas_pos,
        'ris_phases': ris_phases,
        'final_sinr': final_sinr,
        'final_capacity': final_capacity,
        'primary_power': final_primary_power,
        'interference_power': final_interference_power,
        'history': history,
        'user_positions': user_positions
    }


def run_monte_carlo_simulation():
    """运行蒙特卡洛仿真或单次仿真"""
    print(f"\n{'='*60}")
    print(f"FAS-RIS基线系统仿真")
    print(f"{'='*60}")
    print(f"配置参数:")
    print(f"  用户数量: {NUM_USERS}")
    print(f"  FAS天线数量: {NUM_FAS}")
    print(f"  RIS单元数量: {NUM_RIS_ELEMENTS}")
    print(f"  FAS区域大小: {FAS_AREA_SIZE}m x {FAS_AREA_SIZE}m")
    print(f"  发射功率: {TRANSMIT_POWER} W")
    print(f"  噪声功率: {NOISE_POWER} W ({NOISE_POWER_DBM:.2f} dBm)")

    # 根据配置选择仿真模式
    if USE_MONTE_CARLO:
        print(f"  仿真模式: 蒙特卡洛方法")
        print(f"  试验次数: {NUM_MONTE_CARLO_TRIALS}")
        num_trials = NUM_MONTE_CARLO_TRIALS
    else:
        print(f"  仿真模式: 单次仿真")
        print(f"  试验次数: 1")
        num_trials = 1

    print(f"{'='*60}\n")

    # 设置随机种子
    np.random.seed(RANDOM_SEED)

    # 运行试验
    results = []
    for trial_id in range(num_trials):
        trial_result = run_single_trial(trial_id)
        results.append(trial_result)

    return results


def analyze_results(results):
    """分析和打印仿真结果"""
    print(f"\n{'='*60}")
    print(f"仿真结果统计")
    print(f"{'='*60}\n")

    sinr_values = np.array([r['final_sinr'] for r in results])
    capacity_values = np.array([r['final_capacity'] for r in results])

    if USE_MONTE_CARLO and len(results) > 1:
        # 蒙特卡洛模式：显示统计信息
        print(f"SINR统计 (基于{len(results)}次试验):")
        print(f"  平均值: {np.mean(sinr_values):.4f} ({10*np.log10(np.mean(sinr_values)):.2f} dB)")
        print(f"  标准差: {np.std(sinr_values):.4f}")
        print(f"  最小值: {np.min(sinr_values):.4f} ({10*np.log10(np.min(sinr_values)):.2f} dB)")
        print(f"  最大值: {np.max(sinr_values):.4f} ({10*np.log10(np.max(sinr_values)):.2f} dB)")

        print(f"\n容量统计 (bits/s/Hz):")
        print(f"  平均值: {np.mean(capacity_values):.4f}")
        print(f"  标准差: {np.std(capacity_values):.4f}")
        print(f"  最小值: {np.min(capacity_values):.4f}")
        print(f"  最大值: {np.max(capacity_values):.4f}")
    else:
        # 单次仿真模式：显示单个结果
        print(f"SINR结果:")
        print(f"  值: {sinr_values[0]:.4f} ({10*np.log10(sinr_values[0]):.2f} dB)")

        print(f"\n容量结果 (bits/s/Hz):")
        print(f"  值: {capacity_values[0]:.4f}")

    return sinr_values, capacity_values


def plot_results(results, sinr_values, capacity_values):
    """绘制仿真结果"""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 仅在蒙特卡洛模式且有多个试验时绘制分布图
    if USE_MONTE_CARLO and len(results) > 1:
        # 绘图1: SINR分布
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.hist(sinr_values, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('SINR (Linear)')
        plt.ylabel('Frequency')
        plt.title('SINR Distribution')
        plt.grid(True, alpha=0.3)

        # 绘图2: SINR (dB)
        plt.subplot(1, 3, 2)
        sinr_db = 10 * np.log10(sinr_values)
        plt.hist(sinr_db, bins=20, edgecolor='black', alpha=0.7, color='orange')
        plt.xlabel('SINR (dB)')
        plt.ylabel('Frequency')
        plt.title('SINR Distribution (dB)')
        plt.grid(True, alpha=0.3)

        # 绘图3: 容量分布
        plt.subplot(1, 3, 3)
        plt.hist(capacity_values, bins=20, edgecolor='black', alpha=0.7, color='green')
        plt.xlabel('Capacity (bits/s/Hz)')
        plt.ylabel('Frequency')
        plt.title('Capacity Distribution')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'performance_distribution.png'), dpi=150)
        print(f"\n保存图表: {os.path.join(RESULTS_DIR, 'performance_distribution.png')}")
    else:
        print(f"\n(单次仿真模式，跳过分布图绘制)")

    # 绘图4: 第一次试验的收敛曲线
    if len(results) > 0:
        history = results[0]['history']
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history['sinr_values'], marker='o', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('SINR (Linear)')
        plt.title('SINR Convergence Curve (Trial 1)')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        sinr_db_conv = 10 * np.log10(np.array(history['sinr_values']))
        plt.plot(sinr_db_conv, marker='o', linewidth=2, color='orange')
        plt.xlabel('Iteration')
        plt.ylabel('SINR (dB)')
        plt.title('SINR Convergence Curve (dB) (Trial 1)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'convergence.png'), dpi=150)
        print(f"保存图表: {os.path.join(RESULTS_DIR, 'convergence.png')}")


def save_results(results, sinr_values, capacity_values):
    """保存结果到文件"""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(os.path.join(RESULTS_DIR, 'results_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("FAS-RIS基线系统仿真结果\n")
        f.write("="*60 + "\n\n")

        f.write(f"配置参数:\n")
        f.write(f"  用户数量: {NUM_USERS}\n")
        f.write(f"  FAS天线数量: {NUM_FAS}\n")
        f.write(f"  RIS单元数量: {NUM_RIS_ELEMENTS}\n")

        if USE_MONTE_CARLO:
            f.write(f"  仿真模式: 蒙特卡洛方法\n")
            f.write(f"  试验次数: {NUM_MONTE_CARLO_TRIALS}\n\n")

            f.write(f"SINR统计:\n")
            f.write(f"  平均值: {np.mean(sinr_values):.4f} ({10*np.log10(np.mean(sinr_values)):.2f} dB)\n")
            f.write(f"  标准差: {np.std(sinr_values):.4f}\n")
            f.write(f"  最小值: {np.min(sinr_values):.4f}\n")
            f.write(f"  最大值: {np.max(sinr_values):.4f}\n\n")

            f.write(f"容量统计:\n")
            f.write(f"  平均值: {np.mean(capacity_values):.4f} bits/s/Hz\n")
            f.write(f"  标准差: {np.std(capacity_values):.4f}\n")
            f.write(f"  最小值: {np.min(capacity_values):.4f}\n")
            f.write(f"  最大值: {np.max(capacity_values):.4f}\n")
        else:
            f.write(f"  仿真模式: 单次仿真\n\n")

            f.write(f"SINR结果:\n")
            f.write(f"  值: {sinr_values[0]:.4f} ({10*np.log10(sinr_values[0]):.2f} dB)\n\n")

            f.write(f"容量结果:\n")
            f.write(f"  值: {capacity_values[0]:.4f} bits/s/Hz\n")

    print(f"保存结果: {os.path.join(RESULTS_DIR, 'results_summary.txt')}")


if __name__ == "__main__":
    # 运行仿真
    results = run_monte_carlo_simulation()

    # 分析结果
    sinr_values, capacity_values = analyze_results(results)

    # 绘制结果
    if PLOT_RESULTS:
        plot_results(results, sinr_values, capacity_values)

    # 保存结果
    if SAVE_RESULTS:
        save_results(results, sinr_values, capacity_values)

    print(f"\n{'='*60}")
    print(f"仿真完成!")
    print(f"{'='*60}\n")

