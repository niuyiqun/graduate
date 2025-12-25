# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：grpo.py
@Author  ：niu
@Date    ：2025/12/3 10:15
@Desc    ：
"""

import numpy as np

def grpo(cost_matrix):
    """
    基于GRPO (Greedy Randomized Probabilistic Optimization) 的指派算法实现
    :param cost_matrix: numpy array, n x n 的成本矩阵
    :return: 最优分配（任务分派列表），与总成本
    """
    n = cost_matrix.shape[0]
    best_solution = None
    best_cost = float("inf")
    iterations = 100

    for _ in range(iterations):
        available_rows = list(range(n))
        available_cols = list(range(n))
        solution = []
        cost = 0

        for step in range(n):
            sub_matrix = cost_matrix[np.ix_(available_rows, available_cols)]
            # 贪婪选择top k的指派（通常取k=2~5），增加一定的随机性
            k = min(3, sub_matrix.size)
            flat_indices = sub_matrix.flatten().argsort()[:k]
            chosen_idx = np.random.choice(flat_indices)
            row_idx, col_idx = np.unravel_index(chosen_idx, sub_matrix.shape)

            actual_row = available_rows[row_idx]
            actual_col = available_cols[col_idx]
            solution.append((actual_row, actual_col))
            cost += cost_matrix[actual_row, actual_col]

            # 删除已分配的行和列
            del available_rows[row_idx]
            del available_cols[col_idx]

        if cost < best_cost:
            best_cost = cost
            best_solution = solution

    return best_solution, best_cost


