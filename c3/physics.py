import math
from ..c2.definitions import AtomType, GraphNode


class PotentialField:
    """定义记忆空间的物理法则"""

    @staticmethod
    def calculate_decay(node: GraphNode, current_time: float) -> float:
        """
        Chapter 3 Step 1: 双原子差异化衰减
        Concept: 低衰减 (Attractor Basins)
        Event: 高衰减 (Particles)
        """
        age = current_time - node.atom.timestamp
        atom_type = node.atom.atom_type

        if atom_type in [AtomType.PROFILE, AtomType.KNOWLEDGE]:
            # 概念原子：对数级缓慢衰减
            decay = 1.0 / (math.log(age + 2))
        else:
            # 情景原子：指数级快速衰减 (除非 info_weight 很高)
            decay = math.exp(-0.1 * age) * node.atom.info_weight

        return max(0.01, decay)