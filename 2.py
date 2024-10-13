import numpy as np
import pandas as pd
import rasterio
from heapq import heappop, heappush
def read_map(file_path):
    with rasterio.open(file_path) as src:
        map = src.read(1)
    return map
def get_elevation(map, x, y):
    [r, c] = (np.array([12499 - y, x])).astype(np.int16)
    return map[r, c]
def calculate_slope(map, x, y, cellsize=5,k=5):
    a, b, c, d, e, f, g, h, i = [
        get_elevation(map, x - 1, y + 1),
        get_elevation(map, x, y + 1),
        get_elevation(map, x + 1, y + 1),
        get_elevation(map, x - 1, y),
        get_elevation(map, x , y),
        get_elevation(map, x + 1, y),
        get_elevation(map, x - 1, y - 1),
        get_elevation(map, x, y - 1),
        get_elevation(map, x + 1, y - 1)
    ]

    dz_dx = k*(((c + 2 * f + i) - (a + 2 * d + g)) / (8 * cellsize))
    dz_dy = k*(((a + 2 * b + c) - (g + 2 * h + i)) / (8 * cellsize))
    if np.sqrt(dz_dx ** 2 + dz_dy ** 2) == 0:
        slope = 0
    else:
        slope = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)) * (180 / np.pi)
    return slope

# 定义速度与坡度的关系
speed_profile = {
    (0, 10): 30,  # [0°, 10°) 速度为 30km/h
    (10, 20): 20,  # [10°, 20°) 速度为 20km/h
    (20, 30): 10,  # [20°, 30°) 速度为 10km/h
    }

# 定义里程加权系数
mileage_weights = {
    (1, 0): 1,
    (1, 45): 1.5,
    (1, 90): 2,
    (2, 0): np.sqrt(2),
    (2, 45): np.sqrt(2) + 0.5,
    (2, 90): np.sqrt(2) + 1,
    }
#确定速度
def calculate_speed(map, x, y, cellsize=5, k=5):
    # 计算当前坐标的坡度
    slope = calculate_slope(map, x, y, cellsize, k)
    # 定义速度与坡度的关系
    speed_profile = {
        (0, 10): 30,  # [0°, 10°) 速度为 30km/h
        (10, 20): 20,  # [10°, 20°) 速度为 20km/h
        (20, 30): 10,  # [20°, 30°) 速度为 10km/h
    }
    # 默认速度为0，如果坡度不在定义的范围内
    speed = 0
    # 遍历速度与坡度的关系，找到匹配的坡度范围
    for key, value in speed_profile.items():
        if key[0] <= slope < key[1]:
            speed = value
            break
    return speed
#计算单个栅格内的行驶距离。(km)

def calculate_single_cell_distance(x_current, y_current, theta_current, x_previous, y_previous, theta_previous, cell_size=5):

    # 定义里程加权系数
    mileage_weights = {
        (1, 0): 1,
        (1, 45): 1.5,
        (1, 90): 2,
        (2, 0): np.sqrt(2),
        (2, 45): np.sqrt(2) + 0.5,
        (2, 90): np.sqrt(2) + 1,
    }

    # 计算坐标变化量和车头朝向变化量
    delta_x = abs(x_current - x_previous)
    delta_y = abs(y_current - y_previous)
    delta_theta = abs(theta_current - theta_previous)
    delta_l = delta_x + delta_y

    # 将角度变化量转换为最接近的常见角度
    if delta_theta == 0 or delta_theta == 90:
        delta_theta = delta_theta
    elif delta_theta < 45:
        delta_theta = 0
    elif 45 <= delta_theta < 90:
        delta_theta = 45
    else:
        delta_theta = 90
    # 计算里程
    if (delta_l, delta_theta) in mileage_weights:
        distance = mileage_weights[(delta_l, delta_theta)] * cell_size
        distance = distance / 1000  # 转换为千米
    else:
        distance = 0

    return distance
# 创建快速查找的不良区域数据结构
def create_hazardous_area_lookup(hazardous_areas):
    lookup = set()
    for _, row in hazardous_areas.iterrows():
        lookup.add((row['栅格x坐标'], row['栅格y坐标']))
    return lookup
# 读取不良区域数据
hazardous_areas1 = pd.read_excel('附件4：不良区域位置信息.xlsx',sheet_name="不良区域1")
hazardous_areas2 = pd.read_excel('附件4：不良区域位置信息.xlsx',sheet_name="不良区域2")
# 创建不良区域查找表
hazardous_area_lookup = create_hazardous_area_lookup(pd.concat([hazardous_areas1, hazardous_areas2]))



def heuristic(a, b):
    # 欧几里得距离作为启发式
    return np.linalg.norm(np.array(a) - np.array(b))

def a_star_search(start, goal, grid, speed_profile, mileage_weights):
    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in get_neighbors(current, grid):
            tentative_g_score = g_score[current] + distance(current, neighbor, grid, speed_profile, mileage_weights)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heappush(open_set, (f_score[neighbor], neighbor))

    return []

def get_neighbors(node, grid):
    # 获取邻居节点
    pass

def distance(current, neighbor, grid, speed_profile, mileage_weights):
    # 计算距离和时间
    pass

def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]

# 读取栅格数据
with rasterio.open('map.tif') as dataset:
    elevation_data = dataset.read(1)

# 计算坡度
slope_data = calculate_slope(elevation_data)

# 进行路径规划
start = (0, 0)
goal = (12499, 12499)
path = a_star_search(start, goal, slope_data, speed_profile, mileage_weights)

print("Optimal path:", path)

import numpy as np
from osgeo import gdal
from scipy.ndimage import filters
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance


# 读取TIFF文件
def read_tif(file_path):
    dataset = gdal.Open(file_path)
    band = dataset.GetRasterBand(1)
    elevation_data = band.ReadAsArray()
    return elevation_data, dataset


# 计算坡度
def calculate_slope(elevation_data, cell_size=30.0):
    # 计算梯度
    dzdx = filters.sobel(elevation_data, axis=1, mode='constant') / cell_size
    dzdy = filters.sobel(elevation_data, axis=0, mode='constant') / cell_size

    # 计算坡度
    slope = np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2)) * (180 / np.pi)
    return slope


# 定义速度模型
def speed_model(slope):
    if slope < 5:
        return 50  # km/h
    elif slope < 10:
        return 30  # km/h
    else:
        return 10  # km/h


# 创建图
def create_graph_from_slope(slope_data, speed_model):
    rows, cols = slope_data.shape
    G = nx.grid_2d_graph(rows, cols)

    # 添加边权重（距离除以速度）
    for u, v in G.edges():
        dist = np.linalg.norm(np.array(u) - np.array(v))
        speed = speed_model(slope_data[u])
        G[u][v]['weight'] = dist / speed

    return G


# 启发式函数
def heuristic(a, b):
    return distance.euclidean(a, b)


# A*算法
def a_star_search(graph, start, goal, heuristic=None):
    if heuristic is None:
        heuristic = lambda x, y: 0

    frontier = []
    nx.push(frontier, (heuristic(start, goal), start))
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while len(frontier) > 0:
        _, current = nx.pop(frontier, 0)

        if current == goal:
            break

        for next_node in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph[current][next_node]['weight']
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost + heuristic(next_node, goal)
                nx.push(frontier, (priority, next_node))
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


# 主函数
def main():
    # 文件路径
    file_path = 'path_to_your_elevation_file.tif'  # 替换为你的TIFF文件路径

    # 读取TIFF文件中的高程数据
    elevation_data, dataset = read_tif(file_path)

    # 计算坡度
    slope_data = calculate_slope(elevation_data)

    # 创建图
    G = create_graph_from_slope(slope_data, speed_model)

    # 设定起点和终点
    start = (0, 0)  # 起点
    end = (slope_data.shape[0] - 1, slope_data.shape[1] - 1)  # 终点

    # 应用A*算法找到最短路径
    came_from, _ = a_star_search(G, start, end, heuristic=heuristic)
    path = reconstruct_path(came_from, start, end)

    # 输出结果
    print("找到的路径:", path)

    # 可视化结果
    visualize_results(slope_data, path, "Optimal Path on Slope Map using A* Algorithm")


if __name__ == '__main__':
    main()