# 导入 matplotlib 和 numpy 包
import matplotlib.pyplot as plt
import numpy as np
# 这里使用硬编码形式存储数据，需要保证数据列表长度相等
x = [1, 2, 3, 4, 5, 6, 7]
y1 = [0.31, 0.37, 0.42, 0.41, 0.35, 0.32, 0.30]
y2 = [0.27, 0.31, 0.38, 0.44, 0.45, 0.39, 0.36]

# 使用 numpy 生成一组正态分布的随机数，作为误差数据
np.random.seed(341126)
err1 = 0.02 * np.random.random(len(y1))
err2 = 0.02 * np.random.random(len(y1))
# 生成图形对象 fig 和 子图对象 ax，使用约束布局避免重叠
fig, ax = plt.subplots(constrained_layout=True)

# 两次调用坐标系对象 ax 的 errorbar 方法绘制误差棒图
# 参数详见 https://www.wolai.com/matplotlib/6QuMVNfhxsMMnMp7VSpkUk
ax.errorbar(
    x, y1, err1, c="orangered", marker="s", capsize=4, alpha=0.75, label="data: y1"
)
ax.errorbar(x, y2, err2, c="red", marker="s", capsize=4, alpha=0.75, label="data: y2")

# 设置轴标签
ax.set_xlabel("x (m)", fontweight="bold")
ax.set_ylabel(r"$\bf{y_1}$ and $\bf{y_2}$ (m)", fontweight="bold", labelpad=2)

# 添加图例
ax.legend(loc=0)

# 保存图片
# plt.savefig('误差折线图.png', dpi=300)

# 显示图片
# plt.show()
plt.savefig('./errorbar.png')