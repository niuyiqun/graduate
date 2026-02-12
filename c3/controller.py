import logging
import time
from typing import List
from .potential import PotentialModeler
from .diffusion import GatedDiffuser
from .trajectory import TrajectoryPredictor
from .pruner import EntropyPruner
from ..c2.graph_kernel import GraphKernel


class SDAAController:
    """
    SDAA 完整在线推理流程控制核心。
    对应论文 4.3.1 总体框架描述的五个阶段。
    """

    def __init__(self, kernel: GraphKernel):
        self.kernel = kernel
        self.potential_mod = PotentialModeler(kernel)
        self.diffuser = GatedDiffuser(kernel.graph)
        self.predictor = TrajectoryPredictor(kernel.get_all_nodes_data())
        self.pruner = EntropyPruner()

        # 配置日志：无时间戳格式
        self.logger = logging.getLogger("SDAA_Engine")
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler("activation_diffusion.log", mode='w', encoding='utf-8')
        fh.setFormatter(logging.Formatter('[%(levelname)s] - %(message)s'))
        self.logger.addHandler(fh)

    def trigger_activation(self, active_node_ids: List[str]):
        """
        执行“状态感知—势能注入—受控扩散—轨迹外推—熵减筛选”全流程
        """
        self.logger.info(f"--- SDAA Activation Triggered by {len(active_node_ids)} nodes ---")
        t_now = time.time()

        # Phase 1: 势能注入 (Potential Injection)
        initial_energies = {}
        for nid in active_node_ids:
            e0 = self.potential_mod.compute_initial_potential(nid, t_now)
            initial_energies[nid] = e0
        self.logger.info(
            f"[Phase 1] Energy injected into seeds. Mean E0: {np.mean(list(initial_energies.values())):.3f}")

        # Phase 2: 受控扩散 (Gated Diffusion)
        diffused_energies = self.diffuser.run_k_step_diffusion(initial_energies)
        exp_nodes = [self.kernel.get_node_data(nid) for nid in diffused_energies.keys()]
        self.logger.info(f"[Phase 2] Diffusion complete. Explicit candidates (S_exp): {len(exp_nodes)}")

        # Phase 3: 轨迹外推 (Manifold Projection)
        imp_nodes = self.predictor.predict_jump(exp_nodes, diffused_energies)
        self.logger.info(f"[Phase 3] Manifold Jump successful. Implicit candidates (S_imp): {len(imp_nodes)}")

        # Phase 4: 熵减筛选 (Entropy Pruning)
        all_candidates = list(set(exp_nodes + imp_nodes))
        # 融合能量分布 (隐式节点赋予基础搜索分)
        merged_energies = diffused_energies.copy()
        for n in imp_nodes:
            if n.node_id not in merged_energies: merged_energies[n.node_id] = 0.5

        final_subgraph, h_val = self.pruner.prune(all_candidates, merged_energies)

        self.logger.info(
            f"[Phase 4] Adaptive Pruning. Entropy H: {h_val:.4f} | Final Prefetch Size: {len(final_subgraph)}")
        self.logger.info(f"[Output] G_next ready for Working Memory loading.")

        return final_subgraph