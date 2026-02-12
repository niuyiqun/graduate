import logging
from typing import List, Dict, Set
import numpy as np
from .potential import PotentialModeler
from .diffusion import GatedDiffuser
from .trajectory import TrajectoryPredictor
from .pruner import EntropyPruner


class SDAAController:
    """
    4.3.1 总体框架 (SDAA Controller)
    实现从“已知当前态”到“预测未来态”的完整推演流水线。
    """

    def __init__(self, graph_kernel):
        self.kernel = graph_kernel

        # 初始化四大核心模块
        self.potential_modeler = PotentialModeler(graph_kernel)
        self.diffuser = GatedDiffuser(graph_kernel.graph)
        self.predictor = TrajectoryPredictor(graph_kernel.get_all_nodes_data())
        self.pruner = EntropyPruner()

        # 配置系统日志
        self.logger = logging.getLogger("SDAA_Controller")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            fh = logging.FileHandler("activation_diffusion.log", mode='w', encoding='utf-8')
            fh.setFormatter(logging.Formatter('[%(levelname)s] - %(message)s'))
            self.logger.addHandler(fh)

    def run_inference_cycle(self, active_node_ids: List[str], t_now: float):
        """
        执行完整在线推理流程：
        状态感知 -> 势能注入 -> 受控扩散 -> 轨迹外推 -> 熵减筛选
        """
        self.logger.info(f"--- 启动 SDAA 联想激活周期 (输入原子数: {len(active_node_ids)}) ---")

        # 1. 势能注入与能量初始化 (Potential Injection)
        # 对应 4.3.2(1) 定义节点初始势能并映射为激活值 A0
        initial_energies = {}
        for node_id in active_node_ids:
            a0 = self.potential_modeler.compute_initial_potential(node_id, t_now)
            initial_energies[node_id] = a0

        self.logger.info(
            f"[Step 1] 势能场建模完成，激活种子节点能量均值: {np.mean(list(initial_energies.values())):.4f}")

        # 2. 神经符号门控扩散 (Neuro-Symbolic Gated Diffusion)
        # 对应 4.3.2(2) 沿语义边和时序边执行 K 步受约束扩散
        diffused_energies = self.diffuser.run_k_step_diffusion(initial_energies)
        explicit_candidates = [self.kernel.get_node_data(nid) for nid in diffused_energies.keys()]

        self.logger.info(f"[Step 2] 显式扩散完成，激活显式候选集 S_exp 大小: {len(explicit_candidates)}")

        # 3. 流形轨迹预测 (Manifold Trajectory Prediction)
        # 对应 4.3.2(3) 计算语义重心并推演下一时刻落点 h_next
        implicit_candidates = self.predictor.predict_implicit_nodes(explicit_candidates, diffused_energies)

        self.logger.info(f"[Step 3] 流形轨迹外推成功，召回隐式候选集 S_imp 大l: {len(implicit_candidates)}")

        # 4. 熵减自适应剪枝 (Entropy-based Adaptive Pruning)
        # 对应 4.3.2(4) 评估聚焦程度 H(C) 并动态确定预取窗口
        hybrid_set = list(set(explicit_candidates + implicit_candidates))
        # 能量融合
        final_scores = diffused_energies.copy()
        for node in implicit_candidates:
            if node.node_id not in final_scores:
                final_scores[node.node_id] = 0.5  # 为隐式召回赋予基础势能

        prefetch_subgraph, h_val = self.pruner.prune(hybrid_set, final_scores)

        self.logger.info(f"[Step 4] 熵减筛选完成，当前分布熵 H: {h_val:.4f}，最终预取节点数: {len(prefetch_subgraph)}")
        self.logger.info(f"--- 联想激活周期结束，预取子图 G_next 已加载至工作记忆 ---")

        return prefetch_subgraph, h_val