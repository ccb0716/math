import numpy as np
import rasterio
from scipy.ndimage import sobel
import networkx as nx
from rasterio.windows import Window
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance
import heapq
import cupy as cp  # 使用CuPy进行GPU加速
import os

# 里程加权系数
mileage_weights = {
    (1, 0): 1,
    (1, 45): 1.5,
    (1, 90): 2,
    (2, 0): np.sqrt(2),
    (2, 45): np.sqrt(2) + 0.5,
    (2, 90): np.sqrt(2) + 1,
}


def get_next_states_for_theta_0(r, c, theta):
    next_states = []
    if  theta==0:
    # 向前移动
        if (r - 1, c) in valid_nodes:
            next_states.append(((r - 1, c), 0))
            next_states.append(((r - 1, c), 45))
            next_states.append(((r - 1, c), 315))
        # 左前方
        if (r - 1, c -1) in valid_nodes:
            next_states.append(((r - 1, c - 1), 315))
            next_states.append(((r - 1, c - 1), 270))
        # 右前方
        if (r-1, c + 1) in valid_nodes:
            next_states.append(((r-1, c + 1), 90))
            next_states.append(((r - 1, c + 1), 45))
    if  theta==45:
    # 向前移动
        if (r - 1, c) in valid_nodes:
            next_states.append(((r - 1, c), 0))
            next_states.append(((r - 1, c), 315))
        # 右
        if (r , c+1) in valid_nodes:
            next_states.append(((r , c+1), 90))
            next_states.append(((r , c+1), 135))
        # 右前方
        if (r-1, c + 1) in valid_nodes:
            next_states.append(((r-1, c + 1), 90))
            next_states.append(((r - 1, c + 1), 45))
            next_states.append(((r - 1, c + 1), 0))

    if  theta==90:
    # 右前方
        if (r - 1, c+1) in valid_nodes:
            next_states.append(((r - 1, c+1), 0))
            next_states.append(((r - 1, c+1), 45))

        # 右
        if (r  ,c +1) in valid_nodes:
            next_states.append(((r , c + 1), 45))
            next_states.append(((r , c + 1), 90))
            next_states.append(((r, c + 1), 135))
        # 右后方
        if (r+1, c + 1) in valid_nodes:
            next_states.append(((r+1, c + 1), 135))
            next_states.append(((r+1, c + 1), 180))
    if  theta==135:
    # 向后移动
        if (r + 1, c) in valid_nodes:
            next_states.append(((r + 1, c), 180))
            next_states.append(((r + 1, c), 225))

        # 左前方
        if (r, c +1) in valid_nodes:
            next_states.append(((r, c +1), 45))
            next_states.append(((r, c +1), 90))
        # 右前方
        if (r+1, c + 1) in valid_nodes:
            next_states.append(((r+1, c + 1), 90))
            next_states.append(((r + 1, c + 1), 135))
            next_states.append(((r + 1, c + 1), 180))
    if theta == 180:
        # 向左后移动
        if (r + 1, c-1) in valid_nodes:
            next_states.append(((r + 1, c-1), 270))
            next_states.append(((r + 1, c-1), 225))

        # 后
        if (r + 1, c) in valid_nodes:
            next_states.append(((r + 1, c), 135))
            next_states.append(((r + 1, c), 180))
            next_states.append(((r + 1, c), 225))
        # 右后方
        if (r + 1, c + 1) in valid_nodes:
            next_states.append(((r + 1, c + 1), 90))
            next_states.append(((r + 1, c + 1), 135))
    if theta == 225:
        # 向左后移动
        if (r + 1, c-1) in valid_nodes:
            next_states.append(((r + 1, c-1), 180))
            next_states.append(((r + 1, c-1), 225))
            next_states.append(((r + 1, c - 1), 270))
        # 左方
        if (r, c - 1) in valid_nodes:
            next_states.append(((r, c - 1), 270))
            next_states.append(((r, c - 1), 315))
        # 后方
        if (r + 1, c ) in valid_nodes:
            next_states.append(((r + 1, c ), 135))
            next_states.append(((r + 1, c ), 180))
    if theta == 270:
        # 向左后移动
        if (r + 1, c-1) in valid_nodes:
            next_states.append(((r + 1, c-1), 180))
            next_states.append(((r + 1, c-1), 225))
        # 左方
        if (r, c - 1) in valid_nodes:
            next_states.append(((r, c - 1), 270))
            next_states.append(((r, c - 1), 315))
            next_states.append(((r, c - 1), 225))
        # 左前方
        if (r - 1, c-1 ) in valid_nodes:

            next_states.append(((r - 1, c-1 ), 0))
            next_states.append(((r - 1, c - 1), 315))
    if theta == 315:
        # 向左移动
        if (r , c - 1) in valid_nodes:
            next_states.append(((r , c - 1), 270))
            next_states.append(((r , c - 1), 225))
        # 左前方
        if (r-1, c - 1) in valid_nodes:
            next_states.append(((r, c - 1), 270))
            next_states.append(((r, c - 1), 315))
            next_states.append(((r, c - 1), 0))
        # 前方
        if (r , c - 1) in valid_nodes:
            next_states.append(((r - 1, c - 1), 0))
            next_states.append(((r - 1, c - 1), 45))
    return next_states


def create_graph_from_slope(slope_data, speed_profile, cell_size=5):
    global valid_nodes
    rows, cols = slope_data.shape
    G = nx.DiGraph()

    # 将数据移到GPU上处理
    slope_data_gpu = cp.array(slope_data)

    # 仅添加斜率小于一定阈值的节点
    valid_nodes = set([(r, c) for r in range(rows) for c in range(cols) if slope_data[r, c] < 30])

    # 创建带有车头朝向的节点
    for r in range(rows):
        for c in range(cols):
            if (r, c) in valid_nodes:
                for theta in [0, 45, 90, 135, 180, 225, 270, 315]:
                    G.add_node((r, c, theta), slope=slope_data[r, c])

    # 创建边
    for r, c in valid_nodes:
        for theta in [0, 45, 90, 135, 180, 225, 270, 315]:

            next_states = get_next_states_for_theta_0(r, c, theta)

            for ((nr, nc), ntheta) in next_states:
                distance = calculate_single_cell_distance(abs(nc - c), abs(nr - r), ntheta, cell_size)
                speed = speed_model(slope_data_gpu[nr, nc], speed_profile)
                if speed is not None:
                    G.add_edge((r, c, theta), (nr, nc, ntheta), weight=distance / speed)

    return G

# 使用CuPy处理大型数组
def calculate_slope(elevation_data, cell_size=5):
    elevation_data = cp.asarray(elevation_data)
    dzdx = cp.gradient(elevation_data, axis=1) / cell_size
    dzdy = cp.gradient(elevation_data, axis=0) / cell_size
    slope = cp.arctan(cp.sqrt(dzdx ** 2 + dzdy ** 2)) * (180 / cp.pi)
    return slope.get()  # 将结果从GPU内存转回主机内存

# 读取TIFF文件
def read_map(file_path):
    with rasterio.open(file_path) as src:
        elevation_data = src.read(1)
    return elevation_data, src

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

# 新的启发式函数
def adjusted_heuristic(current, goal, slope_data):
    euclidean_distance = distance.euclidean(current, goal)
    # 获取当前节点和目标节点之间的平均斜率

    # 距离目标越近，权重越大，这里使用距离的倒数作为权重因子
    proximity_weight = 1 / (euclidean_distance + 1)  # 防止除以零，加1
    # 将距离的倒数作为权重，乘以调整因子
    return euclidean_distance * proximity_weight

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
    ax.plot3D([p[1] for p in path], [p[0] for p in path], [slope_data[p] for p in path], color='r')
    ax.set_title(title)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Slope (%)')
    plt.show()

# 分割TIFF文件
def split_tiff(input_tiff, output_dir, block_size=(2048,2048)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with rasterio.open(input_tiff) as src:
        width, height = src.width, src.height
        num_blocks_w = int(width / block_size[0]) + 1
        num_blocks_h = int(height / block_size[1]) + 1

        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                w = Window(j * block_size[0], i * block_size[1], min(block_size[0], width - j * block_size[0]), min(block_size[1], height - i * block_size[1]))

                block = src.read(window=w)

                profile = src.profile.copy()
                profile.update({
                    'height': block.shape[1],
                    'width': block.shape[2],
                    'transform': rasterio.windows.transform(w, src.transform),
                })

                out_file = os.path.join(output_dir, f'block_{i}_{j}.tif')
                with rasterio.open(out_file, 'w', **profile) as dst:
                    dst.write(block)

# 主函数
def main():
    file_path = 'map.tif'
    output_dir = 'blocks'
    block_size = (256,256)

    # 分割TIFF文件
    split_tiff(file_path, output_dir, block_size)

    # 遍历每个分割后的TIFF文件
    for tiff_block in os.listdir(output_dir):
        if tiff_block.endswith('.tif'):
            # 读取分割后的TIFF文件块
            elevation_data, src = read_map(os.path.join(output_dir, tiff_block))

            # 计算块的位置
            i, j = map(int, tiff_block.replace('.tif', '').replace('block_', '').split('_'))
            block_row_start = i * block_size[0]
            block_col_start = j * block_size[1]

            # 计算斜率数据
            slope_data = calculate_slope(elevation_data)

            # 定义速度模型
            speed_profile = {
                (0, 10): 30,
                (10, 20): 20,
                (20, 30): 10
            }

            # 创建图
            G = create_graph_from_slope(slope_data, speed_profile)

            # 调整起点和终点
            start = (100- block_row_start, 100 - block_col_start)
            end = (200 - block_row_start, 200 - block_col_start)

            # 确保起点和终点在块内
            if start[0] < 0 or start[0] >= slope_data.shape[0] or start[1] < 0 or start[1] >= slope_data.shape[1]:
                start = (0, 0)  # 如果不在块内，则设置为块的左上角
            if end[0] < 0 or end[0] >= slope_data.shape[0] or end[1] < 0 or end[1] >= slope_data.shape[1]:
                end = (slope_data.shape[0] - 1, slope_data.shape[1] - 1)  # 如果不在块内，则设置为块的右下角

            # A* 搜索
            came_from, _ = a_star_search(G, start, end, heuristic=lambda a, b: adjusted_heuristic(a, b, slope_data))
            path = reconstruct_path(came_from, start, end)

            print(f"在文件 {tiff_block} 中找到的路径:", path)

            visualize_results(slope_data, path, f"Optimal Path on Slope Map using A* Algorithm ({tiff_block})")

if __name__ == '__main__':
    main()



