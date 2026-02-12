import matplotlib.pyplot as plt
import numpy as np

# ================= 配置：通用字体 =================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False 

# ================= 数据准备 =================
# 变体名称 (英文换行以适应图表)
variants = [
    'S-Diffusion\n(Full)',
    'w/o\nDiffusion',
    'w/o\nTrajectory',
    'w/o\nPruning'
]

# 数据 1: Prefetch Hit Rate (%) - 左轴
# 逻辑: Full(86.4), w/o Diff(大幅降), w/o Traj(微降), w/o Prun(微升)
hit_rates = [86.4, 55.0, 82.0, 89.0]

# 数据 2: Retrieval Latency (ms) - 右轴
# 逻辑: Full(83), w/o Diff(升), w/o Traj(微升), w/o Prun(激增)
latencies = [83, 185, 95, 210]

# ================= 绘图设置 =================
x = np.arange(len(variants))  # 标签位置
width = 0.35  # 柱状图宽度

# 创建画布
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

# --- 绘制左侧 Y 轴 (Hit Rate) ---
color1 = '#4E79A7'  # 学术蓝
rects1 = ax1.bar(x - width/2, hit_rates, width, label='Prefetch Hit Rate (%)', color=color1, alpha=0.9, edgecolor='black', linewidth=1)
ax1.set_ylabel('Prefetch Hit Rate (%)', color=color1, fontsize=12, fontweight='bold')
ax1.set_ylim(0, 100)  # Hit Rate 0-100%
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(axis='y', linestyle='--', alpha=0.3)

# --- 绘制右侧 Y 轴 (Latency) ---
ax2 = ax1.twinx()  # 共享X轴
color2 = '#F28E2B'  # 学术橙
rects2 = ax2.bar(x + width/2, latencies, width, label='Retrieval Latency (ms)', color=color2, alpha=0.9, edgecolor='black', linewidth=1)
ax2.set_ylabel('Retrieval Latency (ms)', color=color2, fontsize=12, fontweight='bold')
ax2.set_ylim(0, 250)  # Latency 范围
ax2.tick_params(axis='y', labelcolor=color2)

# --- 设置 X 轴标签 ---
ax1.set_xlabel('Ablation Variants', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(variants, fontsize=11)

# --- 添加数值标签 (可选) ---
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1, ax1)
autolabel(rects2, ax2)

# --- 合并图例 (放在底部) ---
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=11)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('c3_ablation_study.png')
print("Chart generated successfully: c3_ablation_study.png")
plt.show()