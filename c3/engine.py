# c3/engine.py

from .potential import PotentialField
from .diffusion import AnisotropicDiffusion
from .prediction import VectorEvolution, TrajectoryPredictor
from .pruning import AdaptivePruner


class AssociativeMemoryEngine:
    def __init__(self, graph_storage, embedding_index):
        self.graph_storage = graph_storage

        # 初始化各组件
        self.potential_field = PotentialField(graph_storage)
        self.diffusion = AnisotropicDiffusion(graph_storage)
        self.predictor = VectorEvolution(TrajectoryPredictor(), embedding_index)
        self.pruner = AdaptivePruner()

    def active_inference(self, current_active_nodes):
        """
        执行一次完整的主动推理/预取周期
        """
        # Step 1: 势能建模
        energy_map = {}
        for node_id in current_active_nodes:
            atom_data = self.graph_storage.get_node_data(node_id)
            mass, decay = self.potential_field.calculate_initial_mass(node_id, atom_data)
            energy_map[node_id] = mass  # 初始能量 = 质量

        # Step 2: 扩散 (运行 T 步)
        for _ in range(3):  # 扩散3步，模拟思维涟漪
            energy_map = self.diffusion.diffuse_step(energy_map)

        # 提取高能量子图作为"当前思维状态"
        active_subgraph = [n for n, e in energy_map.items() if e > 0.5]

        # Step 3: 轨迹预测 (The Jump)
        candidates, scores = self.predictor.predict_next_nodes(active_subgraph, energy_map)

        # Step 4: 自适应剪枝 (The Control)
        prefetch_nodes = self.pruner.prune(candidates, scores)

        return prefetch_nodes
