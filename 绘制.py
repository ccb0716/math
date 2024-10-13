import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from matplotlib.ticker import FuncFormatter

# 读取tif文件
def read_tif(file_path):
    with Image.open(file_path) as img:
        # 将图像转换为numpy数组
        elevation_data = np.array(img)
    return elevation_data

# 绘制3D地形图
def plot_3dterrain(elevation_data):
    # 创建一个图形和一个3D坐标轴
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 创建网格数据
    x = np.arange(elevation_data.shape[1])
    y = np.arange(elevation_data.shape[0])
    x, y = np.meshgrid(x, y)

    # 绘制3D表面图
    ax.plot_surface(x, y, elevation_data, cmap='terrain')

    # 设置坐标轴标签
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_zlabel('Elevation')

    # 放大x和y轴的比例，降低高程比例
    ax.set_box_aspect((10, 10, 3))  # 这里可以根据需要进行调整

    # 设置z轴的刻度格式，降低高程显示比例
    #ax.zaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/10:.0f}'))  # 这里假设将高程缩小10倍显示

    # 显示图形
    plt.show()

# 主程序
if __name__ == "__main__":
    # 假设map.tif文件在当前目录下
    file_path = 'map.tif'
    elevation_data = read_tif(file_path)
    plot_3dterrain(elevation_data)
