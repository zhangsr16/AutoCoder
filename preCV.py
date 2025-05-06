import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

# Pre
color_threshold = 2
background_color = [0, 0, 0]  # 黑色
grid_red = [0, 0, 128]  # 暗红
bar_red = [50, 50, 255]
bar_blue = [252, 252, 84]
screen_tensor = torch.zeros(7, 4)

# Bar
bar_1step = 2
bar_step = 7
bar_lstep = 1308
barmid_1step = 4

# Grid
grid = 1
grid_1stepX = 1
grid_1stepY = 61
grid_stepX = 3
grid_stepY = 72
grid2_1stepX = 4
grid2_1stepY = 61
grid2_stepX = 6
grid2_stepY = 72


def screen_abstract(input_array):
    H, W, C = input_array.shape

    # Red cut
    for i in range(grid_1stepY, H - 1, grid_stepY):
        for j in range(grid_1stepX, W - 1, grid_stepX):
            # 确定池化窗口的范围
            h_start, h_end = i - grid, i + grid
            w_start, w_end = j - grid, j + grid

            # mask grid
            flag = 0
            for h in range(h_start, h_end):
                if flag != 0:
                    break
                for w in range(w_start, w_end):
                    if h != i and w != j:
                        background_distance = calculate_color_distance(input_array[h, w], background_color)
                        red_distance = calculate_color_distance(input_array[h, w], bar_red)
                        if red_distance < color_threshold or background_distance > color_threshold:
                            flag = 1
                            break
            if flag == 0:
                input_array[i, j] = background_color

    # Red_dark cut
    for i in range(grid2_1stepY, H - 1, grid2_stepY):
        for j in range(grid2_1stepX, W - 1, grid2_stepX):
            grid_distance = calculate_color_distance(input_array[i, j], grid_red)
            if grid_distance < color_threshold:
                input_array[i, j] = background_color

    return input_array


def screen_tensor(input_array):
    screen_abstracted = screen_abstract(input_array)
    H, W, C = screen_abstracted.shape
    # cv2.imwrite('screen_abstracted.jpg', screen_abstracted)
    screen_tensor = torch.zeros(int((bar_lstep - bar_1step) / bar_step), 4)
    # Bar abstract
    # avg
    axix = 0
    for i in range(bar_1step, H, bar_step):
        # avg_high
        cnt = 0
        for j in range(W):
            red_distance = calculate_color_distance(input_array[i, j], bar_red)
            bar_blue_distance = calculate_color_distance(input_array[i, j], bar_blue)
            if red_distance < color_threshold or bar_blue_distance < color_threshold:
                break
            cnt += 1
        screen_tensor[axix, 0] = cnt
        # avg_low
        cnt = 0
        for j in reversed(range(W)):
            red_distance = calculate_color_distance(input_array[i, j], bar_red)
            bar_blue_distance = calculate_color_distance(input_array[i, j], bar_blue)
            if red_distance < color_threshold or bar_blue_distance < color_threshold:
                break
            cnt += 1
        screen_tensor[axix, 1] = cnt
        axix += 1
    # est
    axix = 0
    for i in range(bar_1step, H, bar_step):
        # avg_high
        cnt = 0
        for j in range(W):
            red_distance = calculate_color_distance(input_array[i, j], bar_red)
            bar_blue_distance = calculate_color_distance(input_array[i, j], bar_blue)
            if red_distance < color_threshold or bar_blue_distance < color_threshold:
                break
            cnt += 1
        screen_tensor[axix, 2] = cnt
        # avg_low
        cnt = 0
        for j in reversed(range(W)):
            red_distance = calculate_color_distance(input_array[i, j], bar_red)
            bar_blue_distance = calculate_color_distance(input_array[i, j], bar_blue)
            if red_distance < color_threshold or bar_blue_distance < color_threshold:
                break
            cnt += 1
        screen_tensor[axix, 3] = cnt
        axix += 1
    return screen_tensor


def pool_min(input_array, pool_size, stride):
    """
    对输入的numpy数组进行池化操作，对最后一个维度分别取最小值，并支持设置步长。

    参数:
    - input_array: 输入的 numpy.ndarray, 形状为 (H, W, C)
    - pool_size: 池化窗口的大小 (pool_height, pool_width)
    - stride: 滑动步长 (stride_height, stride_width)

    返回:
    - pooled_array: 池化后的数组
    """
    pool_height, pool_width = pool_size
    stride_height, stride_width = stride
    H, W, C = input_array.shape

    # 计算池化后输出的高度和宽度
    out_height = (H - pool_height) // stride_height + 1
    out_width = (W - pool_width) // stride_width + 1

    # 初始化池化结果
    pooled_array = np.zeros((out_height, out_width, C))

    # 遍历每个通道
    for c in range(C):
        for i in range(out_height):
            for j in range(out_width):
                # 确定池化窗口的范围
                h_start, h_end = i * stride_height, i * stride_height + pool_height
                w_start, w_end = j * stride_width, j * stride_width + pool_width

                # 在池化窗口中取最小值
                pooled_array[i, j, c] = np.max(input_array[h_start:h_end, w_start:w_end, c])

    return pooled_array


def statis_col(y, h, x, image, background_color):
    cnt_c = 0
    for yy in range(y, h):
        background_distance = calculate_color_distance(background_color, image[yy, x])
        if background_distance > color_threshold:
            cnt_c += 1
    return cnt_c


def calculate_color_distance(color1, color2):
    """
    计算两个颜色之间的欧几里得距离。
    :param color1: 第一个颜色 [R, G, B]
    :param color2: 第二个颜色 [R, G, B]
    :return: 欧几里得距离
    """
    return np.sqrt(np.sum((np.array(color1) - np.array(color2)) ** 2))


def vertical_scaling(image, color_threshold, background_color):
    """
    对图像进行纵向缩放。
    :param image: 输入图像（BGR 格式）
    :param color_threshold: 颜色相似性的阈值
    :return: 纵向缩放后的图像
    """
    h, w, c = image.shape
    scaled_image = []
    cnt_bg = []
    cnt_col = []

    for x in range(w):  # 遍历每一列
        column = []
        current_region = [image[0, x].tolist()]  # 初始化区域
        cnt_b = 0
        cnt_c = 0
        for y in range(1, h):
            color_distance = calculate_color_distance(current_region[-1], image[y, x])
            background_distance = calculate_color_distance(background_color, image[y, x])
            if background_distance < color_threshold:
                cnt_b += 1  # col_h = All_h - bg_h
                continue
            if color_distance > color_threshold:
                # 保留当前区域并重新开始
                column.append(image[y, x])
                current_region = [image[y, x].tolist()]
                cnt_c = statis_col(y, h, x, image, background_color)
                continue
        cnt_bg.append(cnt_b)
        cnt_col.append(cnt_c)

        # 处理最后一个区域
        if current_region:
            column.append(image[-1, x])

        scaled_image.append(column)

    # 转换为 NumPy 数组，并转置为纵向结果
    max_height = max(len(col) for col in scaled_image)  # 找到最长列的长度
    vertical_result = np.zeros((max_height, w, c), dtype=np.uint8)

    for i, col in enumerate(scaled_image):
        for j, pixel in enumerate(col):
            vertical_result[j, i] = pixel

    return vertical_result[:len(scaled_image[0])], cnt_bg, cnt_col


def horizontal_scaling(image, color_threshold, cnt_bg, cnt_col, background_color):
    """
    对图像进行横向缩放。
    :param image: 输入图像（BGR 格式）
    :param color_threshold: 颜色相似性的阈值
    :return: 横向缩放后的图像
    """
    h, w, c = image.shape
    scaled_image = []
    cntpool = []

    for y in range(h):  # 遍历每一行
        row = []
        current_region = [image[y, 0].tolist()]  # 初始化区域
        pool_flag = 0

        for x in range(1, w):
            color_distance = calculate_color_distance(current_region[-1], image[y, x])

            # 跳过背景色像素
            background_distance = calculate_color_distance(background_color, image[y, x])
            if background_distance < color_threshold:
                pool_flag = 0
                continue
            if color_distance > color_threshold and cnt_col[x] > 2:
                # 保留当前区域并重新开始
                row.append(image[y, x])
                current_region = [image[y, x].tolist()]
                if pool_flag == 0:
                    cntpool.append([max(cnt_bg[x], cnt_bg[x + 1]), max(cnt_col[x], cnt_col[x + 1])])
                    pool_flag = 1

        scaled_image.append(row)

    # 转换为 NumPy 数组
    max_width = max(len(row) for row in scaled_image)  # 找到最长行的长度
    horizontal_result = np.zeros((h, max_width, c), dtype=np.uint8)

    for i, row in enumerate(scaled_image):
        for j, pixel in enumerate(row):
            horizontal_result[i, j] = pixel

    return horizontal_result[:, :len(scaled_image[0])], cntpool


def two_pass_scaling(input_image, color_threshold, background_color):
    """
    基于两次缩放（纵向+横向）的自适应图像缩放算法。
    :param image: 输入图像（BGR 格式）
    :param color_threshold: 颜色相似性的阈值
    :return: 缩放后的图像
    """
    # 调试中间结果
    vertical_scaled, cnt_bg, cnt_col = vertical_scaling(input_image, color_threshold, background_color)
    cv2.imwrite('debug_vertical_scaled.jpg', vertical_scaled)  # 保存纵向缩放结果
    print("Vertical scaled shape:", vertical_scaled.shape)

    final_scaled, cntpool = horizontal_scaling(vertical_scaled, color_threshold, cnt_bg, cnt_col, background_color)
    cv2.imwrite('debug_final_scaled.jpg', final_scaled)  # 保存横向缩放结果
    print("Final scaled shape:", final_scaled.shape)
    return final_scaled, cntpool


input_image = cv2.imread('colbar2.png')
screen_tensored = screen_tensor(input_image[:348])

input_image = pool_min(input_image, [2, 2], [2, 2])

# 执行两次缩放算法
output_image, cntpool = two_pass_scaling(input_image, color_threshold, background_color)

# 保存并显示结果
cv2.imwrite('scaled_image.jpg', output_image)
# 使用 matplotlib 显示图像
output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
plt.imshow(output_image_rgb)

plt.title('Scaled Image')
plt.show()
