from ..c2.graph_storage import MemoryGraph
from ..c2.definitions import EdgeType
from .physics import PotentialField
import collections


class StimulusDiffuser:
    """Chapter 3 Step 2: 神经符号门控的各向异性扩散"""

    def __init__(self, graph: MemoryGraph):
        self.graph = graph
        self.firing_threshold = 0.5  # 激发阈值

    def diffuse(self, start_nodes: list, steps=2) -> dict:
        # 能量缓存: {node_id: energy}
        energy_map = collections.defaultdict(float)

        # 1. 初始注入 (Injection)
        for node in start_nodes:
            energy_map[node.id] = 1.0  # 初始刺激能量

        # 2. 迭代扩散 (Propagation)
        for step in range(steps):
            new_energy = energy_map.copy()
            print(f"  > Diffusion Step {step + 1}...")

            for node_id, current_energy in energy_map.items():
                if current_energy < 0.1: continue  # 能量过低不传导

                # 获取邻居
                neighbors = self.graph.get_neighbors(node_id)

                for neighbor_id, edge_type, weight in neighbors:
                    transmission_rate = self._get_transmission_rate(edge_type)

                    # 门控机制：如果是 SEMANTIC (Concept -> Event)，需要超过阈值才“倒灌”
                    if edge_type == EdgeType.SEMANTIC and current_energy < self.firing_threshold:
                        transmission_rate = 0.0

                    flow = current_energy * transmission_rate * weight
                    new_energy[neighbor_id] += flow

            energy_map = new_energy

        return energy_map

    def _get_transmission_rate(self, edge_type: EdgeType) -> float:
        # 差异化传导率
        rates = {
            EdgeType.ABSTRACT: 0.9,  # 语义共振 (最快)
            EdgeType.IMPLICIT: 0.8,  # 隐式推理 (快)
            EdgeType.TEMPORAL: 0.4,  # 时序推演 (受阻尼)
            EdgeType.SEMANTIC: 0.6,  # 情景回溯 (中等)
            EdgeType.VERSION: 0.1  # 旧版本 (极低)
        }
        return rates.get(edge_type, 0.5)