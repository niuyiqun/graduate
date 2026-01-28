# c3/diffusion.py

import numpy as np


class AnisotropicDiffusion:
    """
    Step 2: 神经符号门控的各向异性扩散
    """

    def __init__(self, graph_storage):
        self.graph_storage = graph_storage
        self.firing_threshold = 5.0  # 激发阈值 (Concept -> Event)

    def diffuse_step(self, current_energy_map):
        """
        执行一步能量扩散
        current_energy_map: {node_id: energy_value}
        """
        next_energy_map = current_energy_map.copy()

        for node_id, energy in current_energy_map.items():
            if energy <= 0.1: continue  # 忽略微小能量

            neighbors = self.graph_storage.get_neighbors(node_id)

            for nbr_id, edge_data in neighbors.items():
                edge_type = edge_data.get("type")
                transfer_energy = 0.0

                # 路径 1: 语义共振 (Concept <-> Concept)
                # 沿着 ABSTRACT 和 IMPLICIT 边快速传播
                if edge_type in ["ABSTRACT", "IMPLICIT"]:
                    transfer_energy = energy * 0.8  # 低阻力

                # 路径 2: 情景回溯 (Concept -> Event)
                # 门控机制: 只有能量超过阈值才“倒灌”
                elif edge_type == "SEMANTIC":  # 假设 SEMANTIC 连接 Concept 和 Event
                    if energy > self.firing_threshold:
                        transfer_energy = energy * 0.6
                    else:
                        transfer_energy = 0.0  # 门控关闭

                # 路径 3: 时序推演 (Event -> Event)
                # 沿着 TEMPORAL 边，受对数阻尼影响
                elif edge_type == "TEMPORAL":
                    # 模拟对数时序阻尼，假设 edge_data 包含时间差 delta_t
                    delta_t = edge_data.get("delta_t", 1.0)
                    damping = 1.0 / (1.0 + np.log(1 + delta_t))
                    transfer_energy = energy * 0.5 * damping

                # 累加能量 (简化版，实际可用矩阵运算加速)
                if nbr_id not in next_energy_map:
                    next_energy_map[nbr_id] = 0.0
                next_energy_map[nbr_id] += transfer_energy

        return next_energy_map