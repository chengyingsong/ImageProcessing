import cv2
import matplotlib.pyplot as plt
import numpy as np
from bokeh.palettes import mpl

img = cv2.imread("Image/img.png")
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


r_x = 1.25  # x轴缩放率
r_y = 1.25  # y轴缩放率
NEAREST_NEIGHBOR = 1


def resize(img, r_x, r_y, interpolation=None):
    src_w, src_h, _ = img.shape
    dst_w, dst_h = round(src_w * r_x), round(src_h * r_y)
    dst = np.zeros((dst_w, dst_h, 3))
    if interpolation == NEAREST_NEIGHBOR:
        for dst_x in range(dst_w):
            for dst_y in range(dst_h):
                src_x = round(dst_x * (src_w / dst_w))
                src_y = round(dst_y * (src_h / dst_h))
                dst[dst_x][dst_y] = img[src_x][src_y]
    dst = dst.astype(int)
    return dst


def display_image_in_actual_size(im_data):
    dpi = 100
    height, width, depth = im_data.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data)


img1 = resize(img, r_x, r_y, NEAREST_NEIGHBOR)
display_image_in_actual_size(img)
display_image_in_actual_size(img1)
plt.show()
