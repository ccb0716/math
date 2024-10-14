import os

import pandas as pd
from rasterio.windows import Window
import numpy as np
import rasterio
from scipy.ndimage import sobel
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance
import heapq
import cupy as cp  # 使用CuPy进行GPU加速

# 里程加权系数
mileage_weights = {
    (1, 0): 1,
    (1, 45): 1.5,
    (1, 90): 2,
    (2, 0): np.sqrt(2),
    (2, 45): np.sqrt(2) + 0.5,
    (2, 90): np.sqrt(2) + 1,
}

# 读取TIFF文件
def read_map(file_path):
    with rasterio.open(file_path) as src:
        elevation_data = src.read(1)
    return elevation_data, src

# 计算坡度
def calculate_slope(elevation_data, cell_size=5):
    elevation_data = cp.asarray(elevation_data)
    dzdx = cp.gradient(elevation_data, axis=1) / cell_size
    dzdy = cp.gradient(elevation_data, axis=0) / cell_size
    slope = cp.arctan(cp.sqrt(dzdx ** 2 + dzdy ** 2)) * (180 / cp.pi)
    return slope.get()  # 将结果从GPU内存转回主机内存

# 定义速度模型
def speed_model(slope, speed_profile):
    for (min_slope, max_slope), speed in speed_profile.items():
        if min_slope <= slope < max_slope:
            return speed
    if slope >= 30:
        return float('inf')  # 无法通行
    return None

# 计算单个单元格的距离
def calculate_single_cell_distance(delta_x, delta_y, delta_theta, cell_size=5):
    delta_l = delta_x + delta_y

    if delta_theta == 0 or delta_theta == 90:
        delta_theta = delta_theta
    elif delta_theta < 45:
        delta_theta = 0
    elif 45 <= delta_theta < 90:
        delta_theta = 45
    else:
        delta_theta = 90

    distance = mileage_weights.get((delta_l, delta_theta), 0) * cell_size / 1000
    return distance

# 创建图
def create_graph_from_slope(slope_data, speed_profile, cell_size=5):
    rows, cols = slope_data.shape
    G = nx.DiGraph()

    for r in range(rows):
        for c in range(cols):
            G.add_node((r, c), slope=slope_data[r, c])

    for r in range(rows):
        for c in range(cols):
            if slope_data[r, c] < 30:
                # 只连接四个直接邻居
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if nr >= 0 and nr < rows and nc >= 0 and nc < cols and slope_data[nr, nc] < 30:
                        angle = 0 if dc == 0 else 90
                        G.add_edge((r, c), (nr, nc),
                                   weight=calculate_single_cell_distance(abs(dc), abs(dr), angle, cell_size))

    return G

# 启发式函数
def adjusted_heuristic(a, b, slope_data):
    # 计算两点之间的欧几里得距离
    euclidean_distance = distance.euclidean(a, b)

    # 获取两点的平均斜率
    avg_slope = (slope_data[a] + slope_data[b]) / 2

    # 根据平均斜率调整启发式值
    # 这里假设斜率越大，调整系数越大，即路径越困难
    adjustment_factor = 1 + avg_slope / 100  # 假设斜率最大为100%

    # 返回调整后的启发式值
    return euclidean_distance * adjustment_factor

# A*算法
def a_star_search(graph, start, goal, heuristic=None):
    if heuristic is None:
        heuristic = lambda x, y: 0

    frontier = []
    heapq.heappush(frontier, (heuristic(start, goal), start))
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while len(frontier) > 0:
        _, current = heapq.heappop(frontier)

        if current == goal:
            break

        for next_node in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph[current][next_node]['weight']
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost + heuristic(next_node, goal)
                heapq.heappush(frontier, (priority, next_node))
                came_from[next_node] = current

    return came_from, cost_so_far

# 构建路径
def reconstruct_path(came_from, start, goal):
    current = goal
    path = [current]
    while current != start:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

# 可视化结果
def visualize_results(slope_data, path, title):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(range(slope_data.shape[1]), range(slope_data.shape[0]))
    surf = ax.plot_surface(X, Y, slope_data, cmap='viridis', edgecolor='none')

    # 仅绘制路径中在裁剪区域内的点
    valid_path = [p for p in path if 0 <= p[0] < slope_data.shape[0] and 0 <= p[1] < slope_data.shape[1]]
    ax.plot3D([p[1] for p in valid_path], [p[0] for p in valid_path], [slope_data[p] for p in valid_path], color='r')

    ax.set_title(title)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Slope (%)')
    plt.show()

def crop_tiff_to_path_area(src, start, end):
    # 确保起点和终点在图像的边界内
    start_row = max(0, min(start[0], src.height - 1))
    start_col = max(0, min(start[1], src.width - 1))
    end_row = max(0, min(end[0], src.height - 1))
    end_col = max(0, min(end[1], src.width - 1))

    # 计算裁剪窗口的边界
    row_start = min(start_row, end_row)
    col_start = min(start_col, end_col)
    row_end = max(start_row, end_row)
    col_end = max(start_col, end_col)

    # 确保窗口尺寸非负
    window_width = max(0, row_end - row_start + 1)
    window_height = max(0, col_end - col_start + 1)

    # 创建裁剪窗口
    window = Window(col_start, row_start, window_width, window_height)

    # 裁剪TIFF文件
    cropped_data = src.read(1, window=window)

    # 返回裁剪后的数据和窗口信息
    return cropped_data, window
coordinates = {


    "Z9": {
        "start": (7054, 410),
        "end": [(5333, 8223)]
    }
}


def main():
    file_path = 'map.tif'
    all_paths = []  # 用于存储所有路径的列表
    for key, value in coordinates.items():
        start = value['start']
        # 现在end是一个坐标点列表，我们将对每个终点分别处理
        for end in value['end']:


            # 读取TIFF文件
            with rasterio.open(file_path) as src:
                # 裁剪TIFF文件到路径区域
                cropped_data, window = crop_tiff_to_path_area(src, start, end)

                # 计算斜率数据
                slope_data = calculate_slope(cropped_data)

                # 定义速度模型
                speed_profile = {
                    (0, 10): 30,
                    (10, 20): 20,
                    (20, 30): 10
                }

                # 创建图
                G = create_graph_from_slope(slope_data, speed_profile)

                # 调整起点和终点到裁剪区域的坐标
                start_in_cropped = (start[1] - window.col_off, start[0] - window.row_off)
                end_in_cropped = (end[1] - window.col_off, end[0] - window.row_off)

                # A* 搜索
                came_from, _ = a_star_search(G, start_in_cropped, end_in_cropped, heuristic=lambda a, b: adjusted_heuristic(a, b, slope_data))
                path = reconstruct_path(came_from, start_in_cropped, end_in_cropped)
                if start[0]<end[0]and start[1]>=end[1]:
                    revised_path = [(c + window.row_off, r + window.col_off) for (r, c) in path]
                # 修订路径坐标为原始TIFF文件中的坐标
                #revised_path = [(r + window.row_off, c + window.col_off) for (r, c) in path]
                else:
                    revised_path = [(r + window.row_off, c + window.col_off) for (r, c) in path]
                # 输出修订后的路径
                all_paths.append((key, revised_path))

# 将所有路径转换为DataFrame
    paths_df = pd.DataFrame(all_paths, columns=['Path_ID', 'Coordinates'])
    # 将坐标转换为两列：X和Y
    paths_df['X'] = paths_df['Coordinates'].apply(lambda x: [coord[1] for coord in x])
    paths_df['Y'] = paths_df['Coordinates'].apply(lambda x: [coord[0] for coord in x])
    # 保存为Excel文件
    paths_df.to_excel('paths.xlsx', index=False)
        # 可视化结果
        #visualize_results(cropped_data,revised_path, "Optimal Path on Slope Map using A* Algorithm")

if __name__ == '__main__':
    main()
