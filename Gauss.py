
#-*-coding:utf-8-*-
"""
python绘制标准正态分布曲线
"""
# ==============================================================
import numpy as np
import math
import matplotlib.pyplot as plt


def gd(x, mu=0, sigma=1):
  """根据公式，由自变量x计算因变量的值

  Argument:
    x: array
      输入数据（自变量）
    mu: float
      均值
    sigma: float
      方差
  """
  left = 1 / (np.sqrt(2 * math.pi) * np.sqrt(sigma))
  right = np.exp(-(x - mu)**2 / (2 * sigma))
  return left * right


if __name__ == '__main__':
  # 自变量
  x = np.arange(25, 38, 0.1)
  # 因变量（不同均值或方差）
  y_1 = gd(x, 30, 4)
  y_2 = gd(x, 32, 16)
  y_3 = gd(x, 30.4, 3.2)

  # 绘图
  plt.plot(x, y_1, color='green', label="z1")
  plt.plot(x, y_2, color='blue', label="z2")
  plt.plot(x, y_3, color='red', label="Z")
  plt.axvline(30, linestyle='--', c="green")
  plt.axvline(32, linestyle='--', c="blue")
  plt.axvline(30.4, linestyle='--', c="red")
  # 设置坐标系
  plt.xlim(25, 38)
  plt.ylim(-0.2, 0.5)

  ax = plt.gca()
  ax.spines['right'].set_color('none')
  ax.spines['top'].set_color('none')
  ax.xaxis.set_ticks_position('bottom')
  ax.spines['bottom'].set_position(('data', 0))
  ax.yaxis.set_ticks_position('left')
  ax.spines['left'].set_position(('data', 0))

  plt.legend(labels=['z1 $\mu = 30, \sigma^2=2.0$', 'z2 $\mu = 32, \sigma^2=4.0$', 'Z  $\mu = 30.4, \sigma^2=3.2$'])
  plt.show()
