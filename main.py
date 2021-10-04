import cv2
import matplotlib.pyplot as plt
import numpy as np
from bokeh.palettes import mpl

img = cv2.imread("Image/img_2.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# print(img1.shape)
# img2 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
# print(img2.shape)
# plt.subplot(1,2,1)
# plt.imshow(img1)
# plt.title("gray")
# plt.subplot(1,2,2)
# plt.imshow(img2)
# plt.show()


r_x = 2  # x轴缩放率
r_y = 2  # y轴缩放率
NEAREST_NEIGHBOR = 1
BILINEAR_INTERPOLATION = 2


def resize(img, r_x, r_y, interpolation=None):
    src_w, src_h, _ = img.shape
    dst_w, dst_h = round(src_w * r_x), round(src_h * r_y)
    dst = np.zeros((dst_w, dst_h, 3))
    if interpolation == NEAREST_NEIGHBOR:
        for dst_x in range(dst_w):
            for dst_y in range(dst_h):
                src_x = round(dst_x * (src_w / dst_w))
                src_y = round(dst_y * (src_h / dst_h))
                if src_x == src_w:
                    src_x -= 1
                if src_y == src_h:
                    src_y -= 1
                dst[dst_x][dst_y] = img[src_x][src_y]
    elif interpolation == BILINEAR_INTERPOLATION:
        for dst_x in range(dst_w):
            for dst_y in range(dst_h):
                x = dst_x * (src_w / dst_w)
                y = dst_y * (src_h / dst_h)
                # 取整数部分和小数部分
                i = int(x)
                j = int(y)
                u = x - i
                v = y - j
                if i+1 < src_w and j+1 < src_h:
                    dst[dst_x][dst_y] = img[i][j] * (1 - u) * (1 - v) + img[i][j + 1] * (1 - u) * v + \
                                    img[i + 1][j] * u * (1 - v) + img[i + 1][j + 1] * u * v
                else:
                    dst[dst_x][dst_y] = img[i][j]

    dst = dst.astype(int)
    return dst


def display_image_in_actual_size(im_data,title):
    dpi = 100
    height, width, depth = im_data.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    plt.title(title)
    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data)


img1 = resize(img, r_x, r_y, NEAREST_NEIGHBOR)
img2 = resize(img,r_x,r_y,BILINEAR_INTERPOLATION)
display_image_in_actual_size(img,"origin")
display_image_in_actual_size(img1,"nearst")
display_image_in_actual_size(img2,"bilinear")
plt.show()
