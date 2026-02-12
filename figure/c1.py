import matplotlib.pyplot as plt
import numpy as np

# ================= 配置：通用字体以避免报错 =================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ================= 数据准备 =================
# 变体名称 (使用英文换行，保持整洁)
variants = [
    'DS-ORC\n(Full)',
    'w/o\nDual-Stream',
    'w/o\nGRPO',
    'w/o\nTruth Gate',
    'w/o\nLogic Comp.'
]

# LOCOMO Avg F1 Score 数据
locomo_f1 = [0.68, 0.55, 0.62, 0.65, 0.66]

# MSC BERTScore 数据
msc_bert = [57.1, 48.5, 52.3, 53.8, 54.5]

# ================= 绘图设置 =================
x = np.arange(len(variants))  # 标签位置
width = 0.35  # 柱状图宽度

# 创建画布
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

# --- 绘制左侧 Y 轴 (LOCOMO Avg F1) ---
color1 = '#4E79A7'  # 学术蓝
rects1 = ax1.bar(x - width/2, locomo_f1, width, label='LOCOMO Avg F1', color=color1, alpha=0.9, edgecolor='black', linewidth=1)
ax1.set_ylabel('LOCOMO Avg F1 Score', color=color1, fontsize=12, fontweight='bold')
ax1.set_ylim(0.4, 0.75)  # 设置Y轴范围以突显差异
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(axis='y', linestyle='--', alpha=0.3)

# --- 绘制右侧 Y 轴 (MSC BERTScore) ---
ax2 = ax1.twinx()  # 共享X轴
color2 = '#F28E2B'  # 学术橙
rects2 = ax2.bar(x + width/2, msc_bert, width, label='MSC BERTScore (%)', color=color2, alpha=0.9, edgecolor='black', linewidth=1)
ax2.set_ylabel('MSC BERTScore (%)', color=color2, fontsize=12, fontweight='bold')
ax2.set_ylim(45, 60)  # 设置Y轴范围以突显差异
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
plt.savefig('c1_ablation_study_clean.png')
print("Chart generated successfully: c1_ablation_study_clean.png")
plt.show()