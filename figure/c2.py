import matplotlib.pyplot as plt
import numpy as np

# ================= 配置：通用字体 =================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ================= 数据准备 =================
# 变体名称 (英文换行)
variants = [
    'NSCS\n(Full)',
    'w/o\nSymbolic',
    'w/o\nEvolution',
    'w/o\nConcept'
]

# Metric 1: LOCOMO Multi-Hop F1
# 逻辑：NSCS最高(45.6)，Symbolic缺失导致推理断裂(35.2)，Evolution缺失影响较小(43.5)，Concept缺失中度影响(40.8)
locomo_f1 = [45.60, 35.20, 43.50, 40.80]

# Metric 2: MSC Consistency
# 逻辑：NSCS最高(94.5)，Symbolic缺失影响较小(91.2)，Evolution缺失导致冲突(78.5)，Concept缺失中度影响(87.0)
msc_consistency = [94.50, 91.20, 78.50, 87.00]

# ================= 绘图设置 =================
x = np.arange(len(variants))  # 标签位置
width = 0.35  # 柱状图宽度

# 创建画布
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

# --- 绘制左侧 Y 轴 (LOCOMO Multi-Hop F1) ---
color1 = '#4E79A7'  # 学术蓝
rects1 = ax1.bar(x - width/2, locomo_f1, width, label='LOCOMO Multi-Hop F1', color=color1, alpha=0.9, edgecolor='black', linewidth=1)
ax1.set_ylabel('LOCOMO Multi-Hop F1', color=color1, fontsize=12, fontweight='bold')
ax1.set_ylim(0, 55)  # 设置Y轴范围
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(axis='y', linestyle='--', alpha=0.3)

# --- 绘制右侧 Y 轴 (MSC Consistency) ---
ax2 = ax1.twinx()  # 共享X轴
color2 = '#F28E2B'  # 学术橙
rects2 = ax2.bar(x + width/2, msc_consistency, width, label='MSC Consistency', color=color2, alpha=0.9, edgecolor='black', linewidth=1)
ax2.set_ylabel('MSC Consistency (0-100)', color=color2, fontsize=12, fontweight='bold')
ax2.set_ylim(60, 100)  # 设置Y轴范围以突显差异
ax2.tick_params(axis='y', labelcolor=color2)

# --- 设置 X 轴标签 ---
ax1.set_xlabel('Ablation Variants', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(variants, fontsize=11)

# --- 合并图例 (放在底部) ---
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=11)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('c2_ablation_study.png')
print("Chart generated successfully: c2_ablation_study.png")
plt.show()