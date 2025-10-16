"""
FAS-RIS系统信道建模（包含Nakagami衰落）
"""
import numpy as np
from config import *


class ChannelModel:
    """FAS-RIS系统信道模型"""

    def __init__(self):
        self.wavelength = WAVELENGTH
        self.path_loss_constant = PATH_LOSS_CONSTANT
        self.path_loss_exponent = PATH_LOSS_EXPONENT

    def calculate_distance(self, pos1, pos2):
        """计算两个位置之间的欧几里得距离"""
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def nakagami_fading(self, m=NAKAGAMI_M, omega=NAKAGAMI_OMEGA):
        """
        生成Nakagami衰落系数
        m: 形状参数 (m=1为Rayleigh衰落)
        omega: 平均功率
        """
        # 使用Gamma分布生成Nakagami衰落
        # Nakagami: h ~ sqrt(Gamma(m, omega/m))
        gamma_sample = np.random.gamma(m, omega / m)
        return np.sqrt(gamma_sample)

    def rician_fading(self, k_factor=RICIAN_K_FACTOR):
        """
        生成Rician衰落系数
        k_factor: K因子 (LOS到NLOS功率比)
        """
        # Rician衰落: h = sqrt(k/(k+1)) * LOS + sqrt(1/(k+1)) * NLOS
        los_component = np.sqrt(k_factor / (k_factor + 1))
        nlos_component = np.sqrt(1 / (k_factor + 1))

        # LOS分量（确定性）
        los = los_component

        # NLOS分量（Rayleigh）
        nlos_real = nlos_component * np.random.randn()
        nlos_imag = nlos_component * np.random.randn()
        nlos = np.sqrt(nlos_real**2 + nlos_imag**2)

        return los + nlos

    def path_loss(self, distance):
        """
        计算路径损耗（自由空间模型）
        PL(d) = C_L * d^(-alpha)
        """
        if distance < 0.1:  # 避免极小距离处的奇点
            distance = 0.1
        return self.path_loss_constant / (distance ** self.path_loss_exponent)

    def direct_link_channel(self, tx_pos, rx_pos, use_fading=True):
        """
        计算直射链路信道（用户到RIS或用户到BS）
        使用Nakagami衰落
        """
        distance = self.calculate_distance(tx_pos, rx_pos)
        path_loss = self.path_loss(distance)

        if use_fading:
            fading = self.nakagami_fading()
        else:
            fading = 1.0

        # 复信道系数
        phase = np.random.uniform(0, 2 * np.pi)
        channel = np.sqrt(path_loss * fading) * np.exp(1j * phase)

        return channel, distance

    def ris_reflected_link_channel(self, user_pos, ris_pos, fas_pos, ris_phase, use_fading=True):
        """
        计算RIS反射链路信道
        路径: 用户 -> RIS -> FAS
        用户到RIS: Nakagami衰落
        RIS到FAS: 无衰落 (LOS)
        """
        # 用户到RIS距离
        dist_user_ris = self.calculate_distance(user_pos, ris_pos)

        # RIS到FAS距离
        dist_ris_fas = self.calculate_distance(ris_pos, fas_pos)

        # 用户到RIS信道（Nakagami衰落）
        if use_fading:
            fading_user_ris = self.nakagami_fading()
        else:
            fading_user_ris = 1.0

        path_loss_user_ris = self.path_loss(dist_user_ris)
        phase_user_ris = np.random.uniform(0, 2 * np.pi)

        # RIS到FAS信道（LOS，无衰落）
        path_loss_ris_fas = self.path_loss(dist_ris_fas)
        phase_ris_fas = 2 * np.pi / self.wavelength * dist_ris_fas

        # 总反射信道
        channel = (np.sqrt(path_loss_user_ris * fading_user_ris) * np.exp(1j * phase_user_ris) *
                   ris_phase *
                   np.sqrt(path_loss_ris_fas) * np.exp(1j * phase_ris_fas))

        return channel, dist_user_ris, dist_ris_fas

    def calculate_sinr(self, signal_power, interference_power, noise_power):
        """计算SINR"""
        sinr = signal_power / (interference_power + noise_power)
        return sinr

    def calculate_capacity(self, sinr):
        """计算Shannon容量"""
        return np.log2(1 + sinr)

    def calculate_angle_factor(self, pos1, pos2, pos3):
        """
        计算三个位置之间的夹角因子
        使用余弦定理计算pos1-pos2-pos3的夹角
        返回角度因子：角度越小，因子越大（干扰越强）
        """
        # 计算距离
        dist_12 = self.calculate_distance(pos1, pos2)
        dist_23 = self.calculate_distance(pos2, pos3)
        dist_13 = self.calculate_distance(pos1, pos3)

        # 使用余弦定理计算夹角
        # cos(θ) = (a² + b² - c²) / (2ab)
        cos_theta = (dist_12**2 + dist_23**2 - dist_13**2) / (2 * dist_12 * dist_23 + 1e-6)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        # 角度因子：角度越小，因子越大
        # 当θ接近0时，因子接近无穷大（干扰最强）
        # 当θ接近π时，因子接近0（干扰最弱）
        angle_factor = 1.0 / (theta + 0.1)  # 加0.1避免除以0

        return angle_factor, theta


class MultiUserChannelModel:
    """FAS-RIS系统多用户信道模型"""

    def __init__(self):
        self.channel_model = ChannelModel()

    def generate_user_positions(self, num_users, area_size=USER_AREA_SIZE,
                               min_distance=None):
        """
        在部署区域内生成随机用户位置
        确保用户距离基站至少min_distance米
        """
        if min_distance is None:
            min_distance = USER_MIN_DISTANCE_FROM_BS

        positions = []
        bs_pos = np.array(BS_POSITION)

        for _ in range(num_users):
            # 生成满足最小距离约束的用户位置
            while True:
                x = np.random.uniform(-area_size/2, area_size/2)
                y = np.random.uniform(-area_size/2, area_size/2)
                z = 1.5  # 用户高度 (1.5m)

                # 计算用户到基站的距离
                user_pos = np.array([x, y, z])
                distance_to_bs = np.linalg.norm(user_pos[:2] - bs_pos[:2])

                # 如果满足最小距离约束，则接受该位置
                if distance_to_bs >= min_distance:
                    positions.append([x, y, z])
                    break

        return positions

    def calculate_received_signal(self, primary_user_pos, interfering_users_pos,
                                 fas_pos, ris_pos, ris_phases, use_fading=True):
        """
        计算FAS处接收信号
        包括直射链路和RIS反射链路
        """
        # 主用户信号（直射+反射）
        direct_channel, _ = self.channel_model.direct_link_channel(
            primary_user_pos, fas_pos, use_fading=use_fading)

        # 主用户RIS反射信号
        reflected_signal = 0
        for n in range(len(ris_phases)):
            # 简化：假设单个RIS单元贡献
            reflected_channel, _, _ = self.channel_model.ris_reflected_link_channel(
                primary_user_pos, ris_pos, fas_pos, ris_phases[n], use_fading=use_fading)
            reflected_signal += reflected_channel

        # 主用户总信号 = 直射 + 反射（两个信号相加）
        primary_signal = direct_channel + reflected_signal
        primary_power = np.abs(primary_signal) ** 2

        # 其他用户的干扰（考虑角度因子）
        interference_power = 0
        for interfering_pos in interfering_users_pos:
            # 直射干扰（不考虑角度因子，直接传播）
            direct_interference, _ = self.channel_model.direct_link_channel(
                interfering_pos, fas_pos, use_fading=use_fading)
            interference_power += np.abs(direct_interference) ** 2

            # RIS反射干扰（考虑角度因子）
            # 计算干扰用户与主用户的夹角因子
            # 夹角：干扰用户 - FAS - 主用户
            angle_factor, angle = self.channel_model.calculate_angle_factor(
                interfering_pos, fas_pos, primary_user_pos)

            for n in range(len(ris_phases)):
                reflected_interference, _, _ = self.channel_model.ris_reflected_link_channel(
                    interfering_pos, ris_pos, fas_pos, ris_phases[n], use_fading=use_fading)
                # RIS反射干扰乘以角度因子（夹角越小，干扰越强）
                interference_power += np.abs(reflected_interference) ** 2 * angle_factor

        return primary_power, interference_power

