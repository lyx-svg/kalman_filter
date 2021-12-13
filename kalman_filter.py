import numpy as np
import matplotlib.pyplot as plt

Q = np.row_stack(((1,0),(0,1)))
# Q = [[1 0]
#     [0 1]]
R = np.row_stack(((0.1, 0), (0, 0.1)))
# R = [[0.1 0. ]
#     [0.  0.1]]
# dan
A = np.row_stack(((1,0),(0,1)))
H = A
I = A
X0 = np.row_stack((0, 1))
W = np.random.rand(2, 1)
V = np.random.rand(2, 1)

X_true = []
Z_true = []
Z_true.append(X0)
X_true.append(X0)
X_bef = X0
# 计算真实值
for i in range(50):
    X_cur = np.dot(A, X_bef) + W
    Z_cur = np.dot(H, X_cur) + V
    X_true.append(X_cur)
    Z_true.append(Z_cur)
    X_bef = X_cur
    W = np.random.rand(2, 1)
    V = np.random.rand(2, 1)

# kalman filter 迭代
X_predict = []
X_predict.append(X0)
P0 = A
P_bef = P0
X_f = []
X_f.append(X0)

for i in range(50):
    X_xianyan = np.dot(A, X_bef)
    X_f.append(X_xianyan)
    P_xianyan = np.dot(np.dot(A, P_bef), np.transpose(A))+Q
    temp = R + np.dot(np.dot(H, P_xianyan), np.transpose(H))
    temp = np.linalg.inv(temp)
    Kk = np.dot(np.dot(P_xianyan, np.transpose(H)), temp)
    X_pre = X_xianyan + np.dot(Kk, (Z_true[i] - np.dot(H, X_xianyan)))
    X_bef = X_pre
    X_predict.append(X_pre)
    P_bef = np.dot((I - np.dot(Kk, H)), P_xianyan)


X_p1 = []
X_p2 = []
X_t1 = []
X_t2 = []
X_f1 = []
X_f2 = []
Z_t1 = []
Z_t2 = []

for i in range(50):
    X_p1.append(X_predict[i][0])
    X_p2.append(X_predict[i][1])

    X_t1.append(X_true[i][0])
    X_t2.append(X_true[i][1])

    X_f1.append(X_f[i][0])
    X_f2.append(X_f[i][1])

    Z_t1.append(Z_true[i][0])
    Z_t2.append(Z_true[i][1])

X_label = range(50)
'''
# 开启一个窗口，num设置子图数量，figsize设置窗口大小，dpi设置分辨率
fig1 = plt.figure(num=1, figsize=(16, 9),dpi=100)
# 直接用plt.plot画图，第一个参数是表示横轴的序列，第二个参数是表示纵轴的序列
plt.plot(X_label, X_p1, "g-", label="X_predict_1")
plt.plot(X_label, X_t1, "r-", label="X_true_1")
# plt.plot(X_label, X_f1, "r-.", label="X_front_1")
plt.plot(X_label, Z_t1, "b-", label="Z_true_")
# 绘制图例
plt.legend(loc='upper right')
# 显示绘图结果
plt.savefig("pose_result.png")
plt.show()

fig2 = plt.figure(num=1, figsize=(16, 9),dpi=100)
# 直接用plt.plot画图，第一个参数是表示横轴的序列，第二个参数是表示纵轴的序列
plt.plot(X_label, X_p2, "g-", label="X_predict_2")
plt.plot(X_label, X_t2, "r-", label="X_true_2")
# lt.plot(X_label, X_f2, "r-.", label="X_front_2")
plt.plot(X_label, Z_t2, "b-", label="Z_true_2")
# 绘制图例
plt.legend(loc='upper right')
# 显示绘图结果
plt.savefig("velocity_result.png")
plt.show()
'''

fig3 = plt.figure(num=1, figsize=(16, 9),dpi=100)
# 直接用plt.plot画图，第一个参数是表示横轴的序列，第二个参数是表示纵轴的序列
# plt.plot(X_label, X_p1, "g-", label="X_predict_2")
plt.plot(X_label, X_t1, "r-", label="X_true_2")
plt.plot(X_label, X_f1, "g-.", label="X_front_2")
# plt.plot(X_label, Z_t2, "b-", label="Z_true_2")
# 绘制图例
plt.legend(loc='upper right')
# 显示绘图结果
plt.savefig("pose_front_true.png")
plt.show()




