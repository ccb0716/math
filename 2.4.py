import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import Polygon, mapping
import numpy as np


def get_rectangle_tif_area(tif_path, point1, point2, output_path=None):
    """
    从包含地理空间信息的TIFF文件中提取指定长方形区域的数据。

    参数:
    tif_path : str
        TIFF文件的路径。
    point1 : tuple
        长方形左下角的坐标 (x, y)，这里x和y应该是地理坐标（例如经纬度）。
    point2 : tuple
        长方形右上角的坐标 (x, y)，这里x和y应该是地理坐标（例如经纬度）。
    output_path : str, optional
        输出TIFF文件的路径。如果不指定，则不会保存输出。
    """
    # 读取TIFF文件
    with rasterio.open(tif_path) as src:
        # 检查是否有地理变换矩阵
        if src.transform.is_identity:
            raise ValueError("TIFF文件缺少地理变换信息。")

        # 创建矩形区域
        rectangle = Polygon([point1, (point2[0], point1[1]), point2, (point1[0], point2[1])])

        # 将地理坐标转换为像素坐标
        left, bottom = src.index(point1[0], point1[1])
        right, top = src.index(point2[0], point2[1])

        # 计算窗口范围
        window = rasterio.windows.Window.from_slices((top, bottom), (left, right))

        # 提取窗口内的数据
        data = src.read(
            window=window,
            out_shape=(src.count, int(bottom - top), int(right - left))
        )

        # 如果需要输出到新文件
        if output_path:
            # 创建新的元数据
            new_meta = src.meta.copy()
            new_meta.update({
                'driver': 'GTiff',
                'height': data.shape[1],
                'width': data.shape[2],
                'transform': rasterio.windows.transform(window, src.transform),
            })

            # 写入新文件
            with rasterio.open(output_path, 'w', **new_meta) as dest:
                dest.write(data)

    return data


# 使用示例
tif_path = 'map.tif'
point1 = (2, 2)  # 左下角坐标
point2 = (2000, 2000)  # 右上角坐标
output_path = 'output_tif_file.tif'

data = get_rectangle_tif_area(tif_path, point1, point2, output_path)

print("Extracted data shape:", data.shape)