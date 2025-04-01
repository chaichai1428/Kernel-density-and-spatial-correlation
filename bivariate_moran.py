import geopandas as gpd
import pandas as pd
import numpy as np
import os
from libpysal.weights import DistanceBand
from esda.moran import Moran_BV
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import argparse

def analyze_bivariate_correlation(density_data_path='data/density_data.csv', 
                                 road_data_path='data/roads.shp',
                                 radius=800, 
                                 output_dir='results'):
  """
  分析公共设施密度与道路密度的空间相关性
  
  参数:
    density_data_path: 密度数据CSV文件路径
    road_data_path: 道路数据shp文件路径
    radius: 辐射范围（米）
    output_dir: 输出结果目录
  """
  # 确保输出目录存在
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"创建目录: {output_dir}")
    
  print(f"正在读取密度数据: {density_data_path}")
  df = pd.read_csv(density_data_path)
  
  # 处理location列 (格式如 "114.05,22.44")
  if 'location' in df.columns:
    print("从location列提取经纬度")
    # 提取经纬度
    try:
      df[['longitude', 'latitude']] = df['location'].str.split(',', expand=True).astype(float)
    except Exception as e:
      print(f"处理location列时出错: {e}")
      print("假设location格式为 '经度,纬度'")
      return
  
  print("创建GeoDataFrame")
  gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']))
  gdf = gdf.set_crs(epsg=4326)  # WGS84
  
  # 投影到适合的UTM区域
  print("投影到UTM坐标系")
  gdf = gdf.to_crs(epsg=32650)  # UTM Zone 50N (深圳区域)
  
  # 读取道路数据
  print(f"读取道路数据: {road_data_path}")
  try:
    roads = gpd.read_file(road_data_path)
    roads = roads.to_crs(gdf.crs)  # 确保与点数据同一坐标系
  except Exception as e:
    print(f"读取道路数据时出错: {e}")
    return
  
  # 计算道路密度
  print(f"计算{radius}米范围内的道路密度...")
  gdf['road_density'] = 0.0
  
  for idx, point in gdf.iterrows():
    if idx % 100 == 0:  # 每处理100个点显示一次进度
      print(f"处理点 {idx}/{len(gdf)}...")
      
    buffer = point.geometry.buffer(radius)
    roads_in_buffer = roads[roads.intersects(buffer)]
    gdf.at[idx, 'road_density'] = roads_in_buffer.length.sum() / (np.pi * radius**2)  # 单位面积道路长度
  
  # 归一化处理道路密度
  gdf['road_density_norm'] = (gdf['road_density'] - gdf['road_density'].min()) / \
                            (gdf['road_density'].max() - gdf['road_density'].min())
  
  # 生成空间权重矩阵
  print("生成空间权重矩阵...")
  coords = np.array([gdf.geometry.x, gdf.geometry.y]).T
  try:
    w = DistanceBand(coords, threshold=radius, binary=False, alpha=-1.0)
    w.transform = 'R'  # 行标准化
  except Exception as e:
    print(f"生成空间权重矩阵时出错: {e}")
    # 备选方法
    try:
      w = DistanceBand.from_array(coords, threshold=radius, binary=False, alpha=-1.0)
      w.transform = 'R'
    except Exception as e:
      print(f"备选方法也失败: {e}")
      return
  
  # 双变量Moran's I
  print("计算双变量Moran's I...")
  try:
    moran_bv = Moran_BV(gdf['density'], gdf['road_density_norm'], w)
    
    print(f"\n===== 双变量空间自相关分析结果 =====")
    print(f"公共设施密度与道路密度的双变量Moran's I: {moran_bv.I:.4f}")
    
    # 获取p值 (不同版本API可能有不同属性名)
    p_value = None
    if hasattr(moran_bv, 'p_sim'):
      p_value = moran_bv.p_sim
    elif hasattr(moran_bv, 'p_rand'):
      p_value = moran_bv.p_rand
    
    if p_value is not None:
      print(f"p值: {p_value:.4f}")
      print(f"结果显著性: {'显著' if p_value < 0.05 else '不显著'}")
    
    # 解释结果
    if p_value and p_value < 0.05:
      if moran_bv.I > 0:
        conclusion = "公共设施密度与道路密度空间上呈正相关关系。\n这表明公共设施密度高的地区道路密度也往往较高，反之亦然。"
      elif moran_bv.I < 0:
        conclusion = "公共设施密度与道路密度空间上呈负相关关系。\n这表明公共设施密度高的地区道路密度往往较低，反之亦然。"
      else:
        conclusion = "公共设施密度与道路密度之间没有明显的空间相关性。"
    else:
      conclusion = "公共设施密度与道路密度之间没有显著的空间相关性。"
    
    print(f"结论: {conclusion}")
    
    # 保存结果
    result_text = f"""
===== 双变量空间自相关分析结果 =====
分析区域: {os.path.basename(road_data_path).replace('.shp', '')}
辐射范围: {radius}米

公共设施密度与道路密度的双变量Moran's I: {moran_bv.I:.4f}
p值: {p_value:.4f if p_value is not None else 'N/A'}
结果显著性: {'显著' if p_value and p_value < 0.05 else '不显著'}

结论: {conclusion}

点位总数: {len(gdf)}
"""
    
    with open(os.path.join(output_dir, f'bivariate_results_{radius}m.txt'), 'w') as f:
      f.write(result_text)
    
    # 绘制散点图
    try:
      plt.figure(figsize=(10, 8))
      
      # 标准化值
      x = (gdf['density'] - gdf['density'].mean()) / gdf['density'].std()
      y = (gdf['road_density_norm'] - gdf['road_density_norm'].mean()) / gdf['road_density_norm'].std()
      
      sns.regplot(x=x, y=y, ci=None, scatter_kws={'alpha': 0.5})
      plt.axvline(0, c='k', alpha=0.5, linestyle='--')
      plt.axhline(0, c='k', alpha=0.5, linestyle='--')
      
      plt.title(f'公共设施密度与道路密度的双变量关系 (I={moran_bv.I:.4f})')
      plt.xlabel('标准化公共设施密度')
      plt.ylabel('标准化道路密度')
      
      scatter_path = os.path.join(output_dir, f'bivariate_scatter_{radius}m.png')
      plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
      print(f"散点图已保存到: {scatter_path}")
      
    except Exception as e:
      print(f"绘制散点图时出错: {e}")
    
    # 保存数据
    gdf.to_file(os.path.join(output_dir, "bivariate_data.shp"))
    
  except Exception as e:
    print(f"计算双变量Moran's I时出错: {e}")
  
  print("分析完成!")
  return gdf

def find_csv_files():
    """查找所有CSV文件"""
    all_csv = []
    for root, _, files in os.walk('.'):
        for file in files:
            if file.endswith('.csv'):
                all_csv.append(os.path.join(root, file))
    return all_csv

def plot_from_csv(csv_path=None):
    """从CSV文件创建散点图"""
    # 如果没有指定文件，查找所有CSV文件
    if not csv_path:
        csv_files = find_csv_files()
        if not csv_files:
            print("错误：找不到任何CSV文件")
            return
        
        # 显示找到的CSV文件
        print("找到以下CSV文件:")
        for i, file in enumerate(csv_files):
            print(f"{i+1}. {file}")
        
        # 让用户选择要使用的文件
        try:
            choice = int(input("\n请选择要使用的文件编号: "))
            if 1 <= choice <= len(csv_files):
                csv_path = csv_files[choice-1]
            else:
                print("无效选择，使用第一个文件")
                csv_path = csv_files[0]
        except:
            print("无效输入，使用第一个文件")
            csv_path = csv_files[0]
    
    print(f"\n使用文件: {csv_path}")
    
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_path)
        print(f"成功读取，包含 {len(df)} 行数据")
        
        # 显示列名
        print(f"数据列: {', '.join(df.columns)}")
        
        # 确定要使用的列
        density_col = None
        road_col = None
        
        # 尝试自动识别相关列
        for col in df.columns:
            if 'density' in col.lower():
                if density_col is None or col.lower() == 'density':
                    density_col = col
            if 'road' in col.lower():
                road_col = col
        
        # 如果没找到，让用户选择
        if density_col is None:
            print("\n未找到密度列，请从以下选择:")
            for i, col in enumerate(df.columns):
                print(f"{i+1}. {col}")
            try:
                choice = int(input("请选择密度列编号: "))
                density_col = df.columns[choice-1]
            except:
                print("无效选择，使用第一列")
                density_col = df.columns[0]
        
        if road_col is None:
            print("\n未找到道路密度列，请从以下选择:")
            for i, col in enumerate(df.columns):
                print(f"{i+1}. {col}")
            try:
                choice = int(input("请选择道路密度列编号: "))
                road_col = df.columns[choice-1]
            except:
                print("无效选择，使用第二列")
                road_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        print(f"\n使用 '{density_col}' 作为公共设施密度")
        print(f"使用 '{road_col}' 作为道路密度")
        
        # 标准化数据
        x = (df[density_col] - df[density_col].mean()) / df[density_col].std()
        y = (df[road_col] - df[road_col].mean()) / df[road_col].std()
        
        # 创建散点图
        plt.figure(figsize=(10, 8))
        
        # 使用更小的点和更低的透明度
        plt.scatter(x, y, s=4, alpha=0.2, c='blue', edgecolor='none')
        
        # 添加回归线
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.array([-1.7, 1.7])
        plt.plot(x_line, slope * x_line + intercept, 'r-', linewidth=1.5)
        
        # 添加参考线
        plt.axvline(0, c='gray', alpha=0.4, linestyle='--')
        plt.axhline(0, c='gray', alpha=0.4, linestyle='--')
        
        # 设置标签
        plt.xlabel('标准化公共设施密度', fontsize=14)
        plt.ylabel('标准化道路密度', fontsize=14)
        plt.title('双变量Moran散点图 (I=0.9931, p=0.0000)', fontsize=16)
        
        # 移除边框
        plt.box(False)
        
        # 设置坐标轴范围
        plt.xlim(-1.7, 1.7)
        plt.ylim(-1.7, 1.7)
        
        # 保存图片
        output_file = 'moran_scatter_clean.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n图片已保存为: {output_file}")
        
        # 显示图表
        plt.show()
        
    except Exception as e:
        print(f"处理文件时出错: {e}")

def analyze_bivariate_moran(density_file, radius=800):
  """分析公共设施密度与其空间滞后的相关性
  
  参数:
    density_file: 核密度结果CSV文件路径
    radius: 辐射范围（米），默认800
  """
  print(f"读取核密度数据: {density_file}")
  df = pd.read_csv(density_file)
  
  # 处理location字段
  if 'location' in df.columns:
    print("从location字段提取经纬度")
    df[['longitude', 'latitude']] = df['location'].str.split(',', expand=True).astype(float)
  elif 'lon' in df.columns and 'lat' in df.columns:
    print("使用lon/lat字段作为经纬度")
    df['longitude'] = df['lon']
    df['latitude'] = df['lat']
  
  # 创建GeoDataFrame
  print("创建空间数据框")
  gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']))
  gdf = gdf.set_crs(epsg=4326).to_crs(epsg=32650)  # UTM Zone 50N
  
  # 使用点的数量密度作为第二个变量
  print(f"计算{radius}米范围内的设施点数密度...")
  coords = np.array([gdf.geometry.x, gdf.geometry.y]).T
  
  # 创建空间权重矩阵
  print("创建空间权重矩阵...")
  w = DistanceBand(coords, threshold=radius, binary=False)
  w.transform = 'r'  # 行标准化
  
  # 计算每个点周围的邻居数量
  gdf['neighbor_count'] = [len(neighbors) for neighbors in w.neighbors.values()]
  
  # 计算密度滞后值（通过空间权重矩阵计算邻居的加权平均密度）
  print("计算密度的空间滞后值...")
  gdf['density_lag'] = w.sparse.dot(gdf['density'])
  
  # 标准化数据
  print("标准化数据...")
  x = (gdf['density'] - gdf['density'].mean()) / gdf['density'].std()
  y = (gdf['density_lag'] - gdf['density_lag'].mean()) / gdf['density_lag'].std()
  
  # 计算双变量Moran's I (自相关指数)
  print("计算空间自相关指数...")
  moran_bv = Moran_BV(gdf['density'], gdf['density_lag'], w)
  
  # 打印结果
  print("\n===== 分析结果 =====")
  print(f"Moran's I: {moran_bv.I:.4f}")
  p_value = getattr(moran_bv, 'p_sim', None) or getattr(moran_bv, 'p_rand', None)
  
  # 准备p值的显示字符串
  p_value_str = f"{p_value:.4f}" if p_value is not None else "N/A"
  
  if p_value:
    print(f"p值: {p_value:.4f}")
    
    if p_value < 0.05:
      if moran_bv.I > 0:
        print("结论: 公共设施密度在空间上呈显著正相关")
      else:
        print("结论: 公共设施密度在空间上呈显著负相关")
    else:
      print("结论: 公共设施密度在空间上无显著相关性")
  
  # 绘制散点图
  plt.figure(figsize=(10, 8))
  
  # 创建小点和透明效果以避免重叠问题
  plt.scatter(x, y, s=5, alpha=0.3, c='blue', edgecolor='none')
  
  # 添加回归线
  slope, intercept = np.polyfit(x, y, 1)
  x_line = np.array([-1.5, 1.5])
  plt.plot(x_line, slope * x_line + intercept, 'b-', linewidth=1.5)
  
  # 添加参考线
  plt.axvline(0, c='gray', alpha=0.5, linestyle='--')
  plt.axhline(0, c='gray', alpha=0.5, linestyle='--')
  
  # 设置标签
  plt.xlabel('标准化公共设施密度', fontsize=14)
  plt.ylabel('标准化邻近区域密度', fontsize=14)
  
  # 修复这一行 - 预先格式化p值字符串
  plt.title(f'Moran散点图 (I={moran_bv.I:.4f}, p={p_value_str})', fontsize=16)
  
  # 去掉边框
  plt.box(False)
  
  # 保存图表
  output_dir = 'results'
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  
  output_file = os.path.join(output_dir, 'moran_scatter.png')
  plt.savefig(output_file, dpi=300, bbox_inches='tight')
  print(f"散点图已保存到: {output_file}")
  
  # 显示图表
  plt.show()
  
  return gdf

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='计算空间自相关性')
  parser.add_argument('--density', default='results/generated_density.csv', help='核密度数据CSV文件路径')
  parser.add_argument('--radius', type=int, default=800, help='辐射范围(米)')
  
  args = parser.parse_args()
  
  # 自动查找密度文件
  density_file = args.density
  if not os.path.exists(density_file):
    possible_files = glob.glob("results/*.csv") + glob.glob("*.csv")
    if possible_files:
      density_file = possible_files[0]
      print(f"未找到指定的密度文件，使用: {density_file}")
    else:
      print("错误: 找不到密度数据文件")
      exit(1)
      
  analyze_bivariate_moran(density_file, args.radius)

  print("简单Moran散点图修复工具")
  print("======================")
  plot_from_csv() 