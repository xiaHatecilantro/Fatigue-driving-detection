"""生成项目对比分析图表"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── 全局中文设置 ──
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# =====================================================
# 数据定义
# =====================================================
models = [
    "本项目\n融合方案",
    "Pure\nMobileNetV3",
    "YOLOv8",
    "ResNet18",
    "Dlib+OpenCV\n规则法",
    "EffRes-\nDrowsyNet",
]

accuracy = [97.9, 93.1, 95.6, 96.5, 87.0, 97.7]
fps_cpu = [25, 27, 20, 15, 18, 12]
params_m = [2.5, 2.5, 25, 11.7, 0, 28]
explainability = [95, 25, 20, 20, 80, 15]     # 可解释性评分
deploy_ease = [90, 85, 55, 50, 90, 40]         # 部署便捷性评分
robustness = [92, 78, 82, 78, 70, 85]          # 环境鲁棒性评分
colors = ["#E63946", "#457B9D", "#2A9D8F", "#E76F51", "#264653", "#8B5CF6"]

# =====================================================
# 图 1：六维雷达图
# =====================================================
fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(polar=True))

categories = ["准确率", "推理速度", "模型轻量", "可解释性", "部署便捷", "鲁棒性"]
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# 归一化各维度到 0-100
radar_data = {
    "本项目\n融合方案": [accuracy[0], fps_cpu[0]/30*100, (1-params_m[0]/30)*100, explainability[0], deploy_ease[0], robustness[0]],
    "Pure\nMobileNetV3": [accuracy[1], fps_cpu[1]/30*100, (1-params_m[1]/30)*100, explainability[1], deploy_ease[1], robustness[1]],
    "YOLOv8":           [accuracy[2], fps_cpu[2]/30*100, (1-params_m[2]/30)*100, explainability[2], deploy_ease[2], robustness[2]],
    "ResNet18":         [accuracy[3], fps_cpu[3]/30*100, (1-params_m[3]/30)*100, explainability[3], deploy_ease[3], robustness[3]],
    "Dlib+OpenCV\n规则法": [accuracy[4], fps_cpu[4]/30*100, (1-params_m[4]/30)*100, explainability[4], deploy_ease[4], robustness[4]],
}

for (label, vals), c in zip(radar_data.items(), colors):
    vals += vals[:1]
    ax.fill(angles, vals, alpha=0.08, color=c)
    ax.plot(angles, vals, "o--", linewidth=2, color=c, label=label, markersize=5)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12, fontweight="bold")
ax.set_ylim(0, 105)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8, color="gray")
ax.set_title("驾驶员疲劳检测方案 — 六维能力对比雷达图", fontsize=16, fontweight="bold", pad=25)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=10, frameon=False)
plt.tight_layout()
plt.savefig(r"D:\疲劳驾驶检测\outputs\comparison_radar.png", dpi=200, bbox_inches="tight")
plt.close()
print("图1/4: 雷达图已生成")

# =====================================================
# 图 2：准确率 + 推理速度 双轴柱状图
# =====================================================
fig, ax1 = plt.subplots(figsize=(12, 6))

x = np.arange(len(models))
width = 0.35

bars1 = ax1.bar(x - width/2, accuracy, width, color=colors, alpha=0.85, edgecolor="white", linewidth=0.8)
ax1.set_ylabel("准确率 (%)", fontsize=13, fontweight="bold")
ax1.set_ylim(75, 102)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=11)

# 标注准确率
for bar, val in zip(bars1, accuracy):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.4, f"{val}%",
             ha="center", va="bottom", fontsize=10, fontweight="bold")

ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, fps_cpu, width, color=["#F4A261"]*len(models), alpha=0.7, edgecolor="white", linewidth=0.8)
ax2.set_ylabel("推理速度 (FPS, CPU)", fontsize=13, fontweight="bold")
ax2.set_ylim(0, 38)

# 标注FPS
for bar, val in zip(bars2, fps_cpu):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, f"{val} FPS",
             ha="center", va="bottom", fontsize=9, color="#E76F51", fontweight="bold")

ax1.set_title("各方案准确率与推理速度对比", fontsize=16, fontweight="bold", pad=15)
ax1.legend([bars1, bars2], ["准确率 (Accuracy)", "推理速度 (FPS, CPU)"], loc="upper right", fontsize=10, frameon=False)

# 高亮本项目
bars1[0].set_edgecolor("#E63946")
bars1[0].set_linewidth(3)

plt.tight_layout()
plt.savefig(r"D:\疲劳驾驶检测\outputs\comparison_acc_fps.png", dpi=200, bbox_inches="tight")
plt.close()
print("图2/4: 准确率+FPS柱状图已生成")

# =====================================================
# 图 3：模型参数量 & 可解释性 气泡图
# =====================================================
fig, ax = plt.subplots(figsize=(12, 7))

bubble_sizes = [s*28 for s in accuracy]  # 气泡大小反映准确率

scatter = ax.scatter(params_m, explainability, s=bubble_sizes, c=colors,
                      alpha=0.75, edgecolors="white", linewidth=2, zorder=5)

for i, (m, sz) in enumerate(zip(models, bubble_sizes)):
    offset_y = 3 if i != 0 else -4
    offset_x = 0.8 if i == 0 else 0.5
    ax.annotate(m.replace("\n", " "), (params_m[i], explainability[i]),
                textcoords="offset points", xytext=(offset_x*6, offset_y*6),
                fontsize=11, fontweight="bold", color=colors[i],
                arrowprops=dict(arrowstyle="->", color=colors[i], lw=1.2))

ax.set_xlabel("模型参数量 (M)", fontsize=13, fontweight="bold")
ax.set_ylabel("可解释性评分", fontsize=13, fontweight="bold")
ax.set_title("模型轻量 vs 可解释性（气泡大小 = 准确率）", fontsize=16, fontweight="bold", pad=15)
ax.set_xlim(-3, 32)
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3, linestyle="--")

# 图例
for label, c, acc in zip(models, colors, accuracy):
    ax.scatter([], [], s=100, c=c, alpha=0.75, label=f"{label.replace(chr(10),' ')} ({acc}%)")
ax.legend(loc="lower left", fontsize=9, frameon=False, ncol=2)

plt.tight_layout()
plt.savefig(r"D:\疲劳驾驶检测\outputs\comparison_bubble.png", dpi=200, bbox_inches="tight")
plt.close()
print("图3/4: 气泡图已生成")

# =====================================================
# 图 4：综合评分横向柱状图
# =====================================================
fig, ax = plt.subplots(figsize=(12, 7))

# 综合评分 = 准确率*0.4 + FPS归一化*0.15 + 轻量归一化*0.1 + 可解释性*0.15 + 部署*0.1 + 鲁棒性*0.1
scores = []
for i in range(len(models)):
    s = (accuracy[i]/100 * 0.40 +
         fps_cpu[i]/30 * 0.15 +
         (1 - params_m[i]/30) * 0.10 +
         explainability[i]/100 * 0.15 +
         deploy_ease[i]/100 * 0.10 +
         robustness[i]/100 * 0.10) * 100
    scores.append(round(s, 1))

y_pos = np.arange(len(models))
bar_colors = ["#E63946" if i == 0 else "#6C757D" for i in range(len(models))]

bars = ax.barh(y_pos, scores, color=bar_colors, alpha=0.85, edgecolor="white", height=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels([m.replace("\n", " ") for m in models], fontsize=12, fontweight="bold")
ax.set_xlabel("综合评分 (加权)", fontsize=13, fontweight="bold")
ax.set_title("各方案综合能力评分\n(加权: 准确率40% + 速度15% + 轻量10% + 可解释性15% + 部署10% + 鲁棒性10%)",
             fontsize=14, fontweight="bold", pad=15)
ax.set_xlim(0, 100)

for bar, score in zip(bars, scores):
    ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height()/2., f"{score}",
            va="center", fontsize=12, fontweight="bold", color="#E63946")

# 图例
ax.legend([bars[0], bars[5]], ["本项目", "其他方案"], loc="lower right", fontsize=10, frameon=False)
plt.tight_layout()
plt.savefig(r"D:\疲劳驾驶检测\outputs\comparison_overall.png", dpi=200, bbox_inches="tight")
plt.close()
print("图4/4: 综合评分图已生成")
print("\n全部图表已保存到 D:\\疲劳驾驶检测\\outputs\\")
