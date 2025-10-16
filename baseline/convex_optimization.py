"""
基于论文FDAO方法的FAS位置和RIS相位交替优化
与论文"基于GS的FAS-RIS系统优化"中的基线方法一致
"""
import numpy as np
from config import *
from channel_model import MultiUserChannelModel


class ConvexOptimizer:
    """基于论文FDAO方法的优化器"""

    def __init__(self):
        self.channel_model = MultiUserChannelModel()
        self.max_iterations = MAX_ITERATIONS
        self.convergence_threshold = CONVERGENCE_THRESHOLD
        self.fas_step_size = FAS_GRADIENT_STEP_SIZE  # 固定步长（论文方法）
        self.gradient_delta = 1e-4  # 数值梯度计算的微小扰动

    def initialize_fas_position(self):
        """在可移动区域中心初始化FAS位置"""
        return np.array([FAS_AREA_SIZE / 2, FAS_AREA_SIZE / 2, 0.0])

    def initialize_ris_phases(self):
        """均匀初始化RIS相位"""
        phases = np.zeros(NUM_RIS_ELEMENTS, dtype=complex)
        for n in range(NUM_RIS_ELEMENTS):
            phase_angle = 2 * np.pi * n / NUM_RIS_ELEMENTS
            phases[n] = np.exp(1j * phase_angle)
        return phases

    def calculate_objective(self, fas_pos, ris_phases, primary_user_pos,
                           interfering_users_pos, ris_pos, bs_pos):
        """计算目标函数（SINR）"""
        primary_power, interference_power = self.channel_model.calculate_received_signal(
            primary_user_pos, interfering_users_pos, fas_pos, ris_pos, ris_phases, use_fading=True)

        sinr = self.channel_model.channel_model.calculate_sinr(
            primary_power, interference_power, NOISE_POWER)

        return sinr, primary_power, interference_power

    def optimize_fas_position_convex(self, fas_pos, ris_phases, primary_user_pos,
                                     interfering_users_pos, ris_pos, bs_pos,
                                     reference_point=None):
        """
        使用论文FDAO方法优化FAS位置

        论文方法:
        1. 计算SINR关于FAS位置的数值梯度
        2. 梯度归一化
        3. 使用固定步长进行梯度上升
        4. 投影到可行域

        更新公式:
        t_m^(q+1) = Π_St(t_m^(q) + η_t * ∇SINR / ||∇SINR||)

        其中:
        - η_t: 固定步长
        - Π_St: 投影到可行域
        - ||∇SINR||: 梯度的二范数（归一化）
        """
        if reference_point is None:
            reference_point = fas_pos.copy()

        # 计算参考点的SINR
        ref_sinr, _, _ = self.calculate_objective(
            reference_point, ris_phases, primary_user_pos, interfering_users_pos, ris_pos, bs_pos)

        # 计算数值梯度（在参考点处）
        gradient = np.zeros(3)

        for dim in range(3):
            fas_pos_plus = reference_point.copy()
            fas_pos_plus[dim] += self.gradient_delta

            sinr_plus, _, _ = self.calculate_objective(
                fas_pos_plus, ris_phases, primary_user_pos, interfering_users_pos, ris_pos, bs_pos)

            gradient[dim] = (sinr_plus - ref_sinr) / self.gradient_delta

        # 梯度归一化（论文方法）
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm > 1e-10:
            gradient_normalized = gradient / gradient_norm
        else:
            gradient_normalized = gradient

        # 使用固定步长进行梯度上升（论文方法）
        new_fas_pos = reference_point + self.fas_step_size * gradient_normalized

        # 投影到可行域
        new_fas_pos[0] = np.clip(new_fas_pos[0], 0, FAS_AREA_SIZE)
        new_fas_pos[1] = np.clip(new_fas_pos[1], 0, FAS_AREA_SIZE)
        new_fas_pos[2] = 0.0

        return new_fas_pos

    def optimize_ris_phases_convex(self, fas_pos, ris_phases, primary_user_pos,
                                   interfering_users_pos, ris_pos, bs_pos):
        """
        使用论文FDAO方法优化RIS相位

        论文方法:
        1. 对每个RIS单元，计算功率增益 ΔP_n(θ_n)
        2. 从码本中选择使功率增益最大的相位
        3. 贪心地更新每个单元的相位

        功率增益定义:
        ΔP_n(θ_n) = |E_{-n}(t_m) + β_n*exp(jθ_n)*G_n(t_m)|^2 - |E_{-n}(t_m)|^2

        其中:
        - E_{-n}(t_m): 除去第n个单元的复场
        - G_n(t_m): 第n个单元的高斯基元
        - β_n: 第n个单元的幅度

        为简化实现，使用SINR作为功率增益的代理
        """
        best_ris_phases = ris_phases.copy()
        best_sinr, _, _ = self.calculate_objective(
            fas_pos, best_ris_phases, primary_user_pos, interfering_users_pos, ris_pos, bs_pos)

        # 对每个RIS单元进行优化（贪心方法）
        for n in range(NUM_RIS_ELEMENTS):
            best_phase_for_n = best_ris_phases[n]
            best_sinr_for_n = best_sinr

            # 尝试所有码本条目（穷举搜索）
            for phase_idx in range(RIS_PHASE_LEVELS):
                phase_angle = 2 * np.pi * phase_idx / RIS_PHASE_LEVELS
                test_phases = best_ris_phases.copy()
                test_phases[n] = np.exp(1j * phase_angle)

                # 计算该相位下的SINR（作为功率增益的代理）
                test_sinr, _, _ = self.calculate_objective(
                    fas_pos, test_phases, primary_user_pos, interfering_users_pos, ris_pos, bs_pos)

                # 选择使SINR最大的相位
                if test_sinr > best_sinr_for_n:
                    best_sinr_for_n = test_sinr
                    best_phase_for_n = np.exp(1j * phase_angle)

            # 更新该单元的相位（贪心更新）
            best_ris_phases[n] = best_phase_for_n
            best_sinr = best_sinr_for_n

        return best_ris_phases

    def alternating_optimization_convex(self, primary_user_pos, interfering_users_pos,
                                       ris_pos, bs_pos):
        """
        论文FDAO方法的交替优化

        算法流程（与论文一致）:
        1. 初始化FAS位置和RIS相位
        2. 循环迭代 (q = 0, 1, ..., Q_max):
           a. 固定Θ^(q)，优化FAS位置 t^(q+1)
              使用梯度上升: t_m^(q+1) = Π_St(t_m^(q) + η_t * ∇SINR / ||∇SINR||)
           b. 固定t^(q+1)，优化RIS相位 Θ^(q+1)
              对每个单元穷举搜索最优相位
           c. 计算新的功率谱 P_rad^(q+1)
           d. 检查收敛条件: |P_rad^(q+1) - P_rad^(q)| < ε
        3. 返回最优解

        注意: 论文返回最后一次迭代的结果，但我们保存全局最优值以获得更好的性能
        """
        fas_pos = self.initialize_fas_position()
        ris_phases = self.initialize_ris_phases()

        # 计算初始功率谱（用于收敛判据）
        prev_sinr, _, _ = self.calculate_objective(
            fas_pos, ris_phases, primary_user_pos, interfering_users_pos, ris_pos, bs_pos)

        # 保存全局最优值
        best_sinr = prev_sinr
        best_fas_pos = fas_pos.copy()
        best_ris_phases = ris_phases.copy()

        history = {
            'fas_positions': [fas_pos.copy()],
            'ris_phases': [ris_phases.copy()],
            'sinr_values': [prev_sinr],
            'primary_power': [],
            'interference_power': [],
            'best_sinr_values': [best_sinr]
        }

        for iteration in range(self.max_iterations):
            # 步骤1: 固定RIS相位，使用梯度上升优化FAS位置（论文方法）
            new_fas_pos = self.optimize_fas_position_convex(
                fas_pos, ris_phases, primary_user_pos, interfering_users_pos,
                ris_pos, bs_pos, reference_point=fas_pos)

            # 步骤2: 固定FAS位置，使用穷举搜索优化RIS相位（论文方法）
            new_ris_phases = self.optimize_ris_phases_convex(
                new_fas_pos, ris_phases, primary_user_pos, interfering_users_pos, ris_pos, bs_pos)

            # 计算新的功率谱（SINR）
            sinr, primary_power, interference_power = self.calculate_objective(
                new_fas_pos, new_ris_phases, primary_user_pos, interfering_users_pos, ris_pos, bs_pos)

            # 保存全局最优值
            if sinr > best_sinr:
                best_sinr = sinr
                best_fas_pos = new_fas_pos.copy()
                best_ris_phases = new_ris_phases.copy()

            history['fas_positions'].append(new_fas_pos.copy())
            history['ris_phases'].append(new_ris_phases.copy())
            history['sinr_values'].append(sinr)
            history['primary_power'].append(primary_power)
            history['interference_power'].append(interference_power)
            history['best_sinr_values'].append(best_sinr)

            # 检查收敛条件（论文方法：基于功率谱变化）
            sinr_change = abs(sinr - prev_sinr)
            if sinr_change < self.convergence_threshold:
                print(f"第{iteration}次迭代收敛 (SINR变化: {sinr_change:.6f})")
                break

            fas_pos = new_fas_pos
            ris_phases = new_ris_phases
            prev_sinr = sinr

            if (iteration + 1) % 10 == 0:
                print(f"第{iteration + 1}次迭代: SINR = {sinr:.4f} ({10*np.log10(sinr):.2f} dB), "
                      f"全局最优 = {best_sinr:.4f} ({10*np.log10(best_sinr):.2f} dB)")

        return best_fas_pos, best_ris_phases, history

