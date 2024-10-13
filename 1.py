import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取高程数据
def read_map(file_path):
    with rasterio.open(file_path) as src:
        map = src.read(1)
    return map

# 获取高程
def get_elevation(map, x, y):
    [r, c] = (np.array([12499 - y, x])).astype(np.int16)
    return map[r, c]
# 计算坡度和坡向
def calculate_slope_aspect(map, x, y, cellsize=5,k=5):
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
    if dz_dy==0 and dz_dx<0 :
        aspect=90
    elif dz_dx==0 and dz_dy>0:
        aspect=180
    elif dz_dy==0 and dz_dx>0:
        aspect=270
    elif dz_dx==0 and dz_dy<0:
        aspect=0
    elif dz_dy==0 and dz_dx==0:
        aspect=0
    else:
        aspect=270+(np.arctan(dz_dx/dz_dy)*(180/np.pi))-90*(dz_dy/abs(dz_dy))
    if aspect < 0:
        aspect += 360
    elif aspect >360:
        aspect -= 360

    return slope, aspect
# 计算法向量之间的夹角
def calculate_angle_between_normals(n1, n2):
    dot_product = np.dot(n1, n2)
    norm_n1 = np.linalg.norm(n1)
    norm_n2 = np.linalg.norm(n2)
    cos_angle = dot_product / (norm_n1 * norm_n2)
    # 限制 cos_angle 的范围在 -1 到 1 之间
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# 计算平稳性指标
def calculate_smoothness(df):
    epsilon = 0  # 平稳性指标初始化为0
    for i in range(1, len(df)):
        # 计算相邻两个栅格的平均坡度
        si_bar = (eval(df.loc[i-1, '坡度']) + eval(df.loc[i, '坡度'])) / 2
        # 计算相邻两个栅格法向量之间的夹角
        phi_i = calculate_angle_between_normals(df.loc[i-1, '法向量'], df.loc[i, '法向量'])
        # 累加到平稳性指标中
        epsilon += si_bar * phi_i
    return epsilon
# 创建快速查找的不良区域数据结构
def create_hazardous_area_lookup(hazardous_areas):
    lookup = set()
    for _, row in hazardous_areas.iterrows():
        lookup.add((row['栅格x坐标'], row['栅格y坐标']))
    return lookup
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
# 初始化路径质量评价指标
total_distance = 0
total_time = 0
elevations = []
slopes = []
aspects = []
times= []
distances=[]
speeds=[]
n=[]
phi=[]

# 读取路径数据
df = pd.read_excel('P1-P2.xlsx', sheet_name=0)
# 读取不良区域数据
hazardous_areas1 = pd.read_excel('附件4：不良区域位置信息.xlsx',sheet_name="不良区域1")
hazardous_areas2 = pd.read_excel('附件4：不良区域位置信息.xlsx',sheet_name="不良区域2")
# 创建不良区域查找表
hazardous_area_lookup = create_hazardous_area_lookup(pd.concat([hazardous_areas1, hazardous_areas2]))
# 计算高程、坡度和坡向
file_path = "D://CCB//Code//Python//math//map.tif"
map = read_map(file_path)

for i in range(len(df)):

    x = df.loc[i, '栅格x坐标']
    y = df.loc[i, '栅格y坐标']
    elevation = get_elevation(map, x, y)
    slope, aspect = calculate_slope_aspect(map, x, y)
    if slope == 0:
        n1 = np.array([0, 0, 1])  # 水平面的法向量为(0, 0, 1)
    else:
        n1 = np.array([
            np.sin(np.radians(slope)) * np.sin(np.radians(aspect)),
            np.sin(np.radians(slope)) * np.cos(np.radians(aspect)),
            np.cos(np.radians(slope))
        ])
    n.append(n1)
    elevations.append(str(elevation))
    slopes.append(str(slope))
    aspects.append(str(aspect))

df['高程'] = elevations
df['坡度'] = slopes
df['坡向'] = aspects
df['法向量'] = n

# 初始化路径质量评价指标
total_distance = 0
total_time = 0
safety_metric = 0
# 初始化前一个栅格的坐标和车头朝向
x_previous, y_previous, theta_previous ,n_previous= None, None, 0,np.array([0,0,1])

# 遍历路径上的每个栅格，从第二个栅格开始
for i in range(len(df) ):
    x_current, y_current, theta_current,n_current= df.loc[i, ['栅格x坐标', '栅格y坐标', '车头朝向（单位：度）','法向量']]

    if x_previous is not None and y_previous is not None:
        # 计算坐标变化量和车头朝向变化量
        delta_x = abs(x_current - x_previous)
        delta_y = abs(y_current - y_previous)
        delta_theta = abs(theta_previous-theta_current  )
        delta_l=delta_x + delta_y
        # 确定速度
        slope = df.loc[i, '坡度']  # 当前栅格的坡度
        speed = speed_profile.get((0, 10), 0)  # 默认速度为0，如果坡度不在定义的范围内
        for key, value in speed_profile.items():
            if key[0] <= eval(slope) < key[1]:
                speed = value
                break
        speeds.append(speed)
        # 计算里程
        cell_size = 5  # 栅格边长
        if (delta_l, delta_theta) in mileage_weights:
            distance = (mileage_weights[(delta_l, delta_theta)] )* cell_size
            distance=distance/1000
            #delta_thetas.append(delta_theta)
        else:
            distance = 0
        # 累加总里程
        distances.append(round(total_distance,4))
        total_distance += distance
        # 计算行驶时长
        time = distance / speed
        # 累加总行驶时长
        total_time += time
        times.append(round(total_time,4))
        # 计算phi
        phi_degree =  calculate_angle_between_normals(n_previous, n_current)
        phi.append(phi_degree)
        #计算不良区域
        if (x_current, y_current) in hazardous_area_lookup:
            time_current = distance/ speed
            safety_metric += time_current# 如果在不良区域内，计算行驶时间并累加到安全性指标

    # 在循环结束时更新前一个栅格的坐标和车头朝向

    x_previous, y_previous, theta_previous,n_previous = x_current, y_current, theta_current,n_current
distances.append(total_distance)
times.append(total_time)
speeds.append(speed)
#delta_thetas.append(theta_previous)
df['total_time'] = times
df['distance'] = distances
df['speed'] = speeds
#df['delta_theta']=delta_thetas
df.to_excel('path_data.xlsx', index=False)
smoothness = calculate_smoothness(df)

print("安全性指标（不良区域行驶时间）:", safety_metric)
print("该路径下平稳性指标ε:", smoothness)
print("该路径下时效性性指标:",round(total_time,5))


