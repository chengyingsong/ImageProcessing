#  直方图均衡化
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("Image/img.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.subplot(1, 2, 1)
# plt.imshow(img, cmap="gray")


# 统计一下灰度值分布函数
N = np.array([0 for i in range(256)])
W, H = img.shape
for x in range(W):
    for y in range(H):
        N[img[x][y]] += 1
plt.subplot(1, 2, 1)
plt.bar(range(256), N)
N = N / (W * H)

# 映射函数计算
S = np.zeros((256, 1))
for i in range(256):
    sk = 255 * sum(N[:i + 1])
    S[i] = round(sk)

for x in range(W):
    for y in range(H):
        img[x][y] = int(S[img[x][y]])

#
# plt.subplot(1, 2, 2)
# plt.imshow(img, cmap="gray")


N = np.array([0 for i in range(256)])
W, H = img.shape
for x in range(W):
    for y in range(H):
        N[img[x][y]] += 1
plt.subplot(1, 2, 2)
plt.bar(range(256), N)

plt.show()