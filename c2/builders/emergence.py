# -*- coding: UTF-8 -*-
# c2/builders/emergence.py

import logging
import uuid
import networkx as nx
from typing import List
from c2.builders.base import BaseGraphBuilder
from c2.definitions import NodeType, AtomCategory, EdgeType, MemoryNode
from general.model import QwenChat

logger = logging.getLogger(__name__)


class EmergenceBuilder(BaseGraphBuilder):
    """
    [THESIS] Phase 5: 结构诱导的概念涌现 (Structure-Induced Emergence)
    逻辑: Bottom-up Abstraction
    1. 结构聚类: 在图上发现紧密连接的事件社区 (Community)
    2. 神经抽象: 利用 LLM 总结社区的共性，生成新的 Concept
    3. 图谱回写: 将新 Concept 写入图谱，并建立 ABSTRACT 边
    """

    def __init__(self, llm_client: QwenChat):
        self.llm = llm_client

    def process(self, new_nodes, graph):
        """
        注意：涌现通常是对全图或较大的子图进行的，而不仅仅是新节点。
        这里我们对全图的 Event 层进行分析。
        """
        self.build_emergence(graph)

    def build_emergence(self, graph):
        logger.info("  [Emergence] 正在启动概念涌现流程...")

        # 1. 转换图结构
        # [SIMPLIFIED] 这里使用 NetworkX 进行内存计算。
        # 对于千万级节点的大图，应使用 GraphScope 或 Neo4j 的图算法引擎。
        nx_graph = graph.get_nx_graph()
        if not nx_graph:
            logger.warning("  [Emergence] 无法获取 NetworkX 图对象，跳过。")
            return

        # 只提取 Episodic 层的节点进行聚类 (我们只关心从行为中涌现规律)
        activity_nodes = [n for n, attr in nx_graph.nodes(data=True)
                          if attr.get('type') == NodeType.EPISODIC.value]

        # [SIMPLIFIED] 数据太少时不聚类，阈值设为 5
        if len(activity_nodes) < 5:
            logger.info("  [Emergence] 节点数量不足，跳过涌现。")
            return

        # 构建子图 (无向图)
        subgraph = nx_graph.subgraph(activity_nodes).to_undirected()

        # 2. 执行社区发现算法 (Community Detection)
        # [THESIS] 论文提及使用 Leiden 算法 (效果优于 Louvain)。
        # [SIMPLIFIED] 为了无需编译 C++ 依赖，优先尝试 Louvain，降级使用 Label Propagation。
        try:
            # 需要 pip install python-louvain 或 networkx>=2.7
            communities = nx.community.louvain_communities(subgraph)
        except AttributeError:
            logger.info("  [Emergence] Louvain 算法不可用，降级使用 Label Propagation。")
            communities = nx.community.label_propagation_communities(subgraph)
        except Exception as e:
            logger.warning(f"  [Emergence] 聚类算法失败: {e}")
            return

        logger.info(f"  [Emergence] 发现了 {len(communities)} 个潜在的语义簇。")

        # 3. 神经抽象 (Neuro-Abstraction)
        for comm in communities:
            # [SIMPLIFIED] 忽略太小的簇 (噪声)
            if len(comm) < 3: continue

            # 收集簇内内容
            cluster_contents = []
            cluster_ids = []
            for node_id in comm:
                node = graph.get_node(node_id)
                if node:
                    cluster_contents.append(f"- {node.content}")
                    cluster_ids.append(node_id)

            # [SIMPLIFIED] 截断 Token，防止 Context Window 溢出
            context_str = "\n".join(cluster_contents[:20])

            # Prompt: 让 LLM 充当“新皮层”进行归纳
            prompt = f"""
            以下是一组用户的具体行为记忆片段，它们在图结构上紧密关联：
            {context_str}

            任务：请分析这些行为背后的共同模式、用户性格特质或高层抽象概念。
            输出：生成一条简短的“用户画像（Profile）”或“一般性知识（Knowledge）”。
            要求：仅输出结论，不要解释。不要包含"根据..."等字样。
            """

            # 调用 Qwen
            res = self.llm.chat([{"role": "user", "content": prompt}])
            abstract_content = res.get('content', '') if isinstance(res, dict) else str(res)

            # 简单的有效性检查
            if len(abstract_content) < 2 or "无法" in abstract_content: continue

            # 4. 图谱回写 (Write Back)
            new_concept_id = f"concept_emerged_{uuid.uuid4().hex[:8]}"
            new_node = MemoryNode(
                node_id=new_concept_id,
                content=abstract_content,
                category=AtomCategory.PROFILE,  # 默认为 Profile，也可以让 LLM 分类
                node_type=NodeType.CONCEPTUAL,
                meta={"source": "emergence_phase", "algorithm": "community_detection"}
            )

            graph.add_node(new_node)
            logger.info(f"    ✨ [Emerged] 涌现出新概念: '{abstract_content[:20]}...'")

            # 建立 ABSTRACT 边 (Concept -> Events)
            # 这体现了“解释力”：高层概念解释了底层的行为簇
            for event_id in cluster_ids:
                graph.add_edge(new_concept_id, event_id, EdgeType.ABSTRACT)