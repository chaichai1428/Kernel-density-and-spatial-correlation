import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance_matrix
import geopandas as gpd
import os
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde

def load_poi_data(poi_file='data/元朗.csv'):
  """
  Load POI data and extract coordinates from location column.
  
  Args:
    poi_file: Path to the POI data file
    
  Returns:
    DataFrame with POI information including extracted coordinates
  """
  try:
    # Try to load data with appropriate encoding for Chinese characters
    df = pd.read_csv(poi_file, encoding='utf-8')
    print(f"成功读取文件: {poi_file}, 共{len(df)}行数据")
    
    # Check if location column exists
    if 'location' in df.columns:
      print("找到location列，正在提取坐标...")
      
      # Extract coordinates from location column
      # Assuming format is like "114.025511:[...]" where 114.025511 is longitude
      def extract_coordinates(loc_str):
        """
        Extract longitude and latitude from location string.
        
        Args:
          loc_str: String containing coordinates in 'longitude,latitude' format
          
        Returns:
          Series with longitude and latitude
        """
        try:
          # Convert to string and clean up
          loc_str = str(loc_str).strip()
          
          # Simple split by comma for 'longitude,latitude' format
          if ',' in loc_str:
            parts = loc_str.split(',')
            if len(parts) == 2:
              lon = float(parts[0])
              lat = float(parts[1])
              
              # Verify coordinates are within reasonable range
              if 113.0 <= lon <= 115.0 and 22.0 <= lat <= 23.0:
                return pd.Series([lon, lat])
        
          print(f"警告: 无法从 '{loc_str}' 提取坐标")
          return pd.Series([None, None])
        except Exception as e:
          print(f"坐标提取错误: {e}")
          return pd.Series([None, None])
      
      # Apply extraction to create longitude and latitude columns
      df[['longitude', 'latitude']] = df['location'].apply(extract_coordinates)
      
      # Print some examples for verification
      print("\n前5个坐标示例:")
      for i, row in df.head(5).iterrows():
          print(f"原始值: {row['location']} -> 提取坐标: ({row['longitude']}, {row['latitude']})")
      
      # Remove rows with missing coordinates
      original_count = len(df)
      df = df.dropna(subset=['longitude', 'latitude'])
      print(f"删除了{original_count - len(df)}行无效坐标数据")
      
      # Print coordinate ranges
      print(f"经度范围: {df['longitude'].min():.6f} 到 {df['longitude'].max():.6f}")
      print(f"纬度范围: {df['latitude'].min():.6f} 到 {df['latitude'].max():.6f}")
      
      print(f"成功提取坐标: {len(df)}个有效点")
      return df
    else:
      print(f"无法在数据中找到location列。可用的列有: {df.columns.tolist()}")
      raise ValueError("数据必须包含location列")
  
  except Exception as e:
    print(f"加载POI数据时出错: {e}")
    # Create sample data as fallback
    print("生成示例数据作为替代...")
    return load_sample_data()

def load_boundary_data(boundary_file='yuanlang.shp'):
  """
  Load boundary information from shapefile.
  
  Args:
    boundary_file: Path to the boundary shapefile
    
  Returns:
    tuple: (GeoDataFrame with boundary geometry, DataFrame with boundary points)
           Returns (None, None) if file not found
  """
  try:
    # Read shapefile using geopandas
    gdf = gpd.read_file(boundary_file)
    print(f"成功加载边界文件: {boundary_file}")
    print(f"边界数据CRS (坐标参考系统): {gdf.crs}")
    print(f"边界数据列: {gdf.columns.tolist()}")
    
    # Check that the file contains geometry
    if 'geometry' not in gdf.columns:
      raise ValueError("边界文件必须包含geometry列")
    
    # If CRS is not WGS84 (EPSG:4326), reproject
    if gdf.crs and gdf.crs != "EPSG:4326":
      print(f"将坐标从 {gdf.crs} 转换为 WGS84 (EPSG:4326)")
      gdf = gdf.to_crs(epsg=4326)
    
    # Extract boundary coordinates for plotting
    # This creates a dataframe with all boundary points
    boundary_points = []
    
    # Handle MultiPolygon or Polygon geometries
    for geom in gdf.geometry:
      if geom.geom_type == 'MultiPolygon':
        for polygon in geom.geoms:
          x, y = polygon.exterior.xy
          for i in range(len(x)):
            boundary_points.append((x[i], y[i]))
      elif geom.geom_type == 'Polygon':
        x, y = geom.exterior.xy
        for i in range(len(x)):
          boundary_points.append((x[i], y[i]))
    
    # Create a dataframe with the boundary points
    boundary_df = pd.DataFrame(boundary_points, columns=['longitude', 'latitude'])
    
    print(f"提取了 {len(boundary_df)} 个边界点")
    print(f"边界经度范围: {boundary_df['longitude'].min():.6f} 到 {boundary_df['longitude'].max():.6f}")
    print(f"边界纬度范围: {boundary_df['latitude'].min():.6f} 到 {boundary_df['latitude'].max():.6f}")
    
    return gdf, boundary_df
    
  except Exception as e:
    print(f"加载边界数据时出错: {e}")
    print("继续进行分析，但不显示边界...")
    return None, None  # 返回两个None而不是一个

def perform_kde_analysis(df, save_path=None):
  """
  Perform Kernel Density Estimation and generate heatmap.
  
  Args:
    df: DataFrame with longitude and latitude columns
    save_path: Optional path to save the figure
  """
  # Verify we have data
  if len(df) == 0:
    print("错误: 没有有效数据进行KDE分析")
    return
  
  # Print data stats before plotting
  print(f"进行KDE分析，使用{len(df)}个点")
  print(f"经度范围: {df['longitude'].min():.6f} 到 {df['longitude'].max():.6f}")
  print(f"纬度范围: {df['latitude'].min():.6f} 到 {df['latitude'].max():.6f}")
  
  plt.figure(figsize=(12, 10))
  
  # Create scatter plot first to confirm data is present
  plt.scatter(df['longitude'], df['latitude'], c='blue', s=10, alpha=0.3, label='POI Points')
  
  # Then add KDE plot with appropriate bandwidth
  try:
    sns.kdeplot(
      x=df['longitude'], 
      y=df['latitude'], 
      cmap="Reds", 
      shade=True, 
      alpha=0.7,
      bw_adjust=0.3  # lower value for more detail
    )
    plt.legend()
  except Exception as e:
    print(f"KDE计算错误: {e}")
  
  # Set appropriate axis limits based on data
  x_margin = (df['longitude'].max() - df['longitude'].min()) * 0.05
  y_margin = (df['latitude'].max() - df['latitude'].min()) * 0.05
  
  plt.xlim(df['longitude'].min() - x_margin, df['longitude'].max() + x_margin)
  plt.ylim(df['latitude'].min() - y_margin, df['latitude'].max() + y_margin)
  
  plt.title('Kernel Density Estimation Heatmap of Commercial Facilities')
  plt.xlabel('Longitude')
  plt.ylabel('Latitude')
  
  # Add grid for better readability
  plt.grid(True, linestyle='--', alpha=0.6)
  
  if save_path:
    plt.savefig(save_path, dpi=300)
  plt.show()

def nearest_neighbor_analysis(df):
  """
  Perform Nearest Neighbor Analysis to determine spatial distribution pattern.
  
  Args:
    df: DataFrame with longitude and latitude columns
    
  Returns:
    dict: Results including mean distance, expected distance, and R value
  """
  # Calculate distance matrix between all points
  coords = df[['longitude', 'latitude']].values
  dist_matrix = distance_matrix(coords, coords)
  
  # Set diagonal to infinity to avoid calculating distance to itself
  np.fill_diagonal(dist_matrix, np.inf)
  
  # Calculate distance to nearest neighbor for each point
  nearest_distances = np.min(dist_matrix, axis=1)
  
  # Calculate mean nearest neighbor distance
  mean_nearest_distance = np.mean(nearest_distances)
  
  # Calculate expected distance in theoretical random distribution
  area = (df['longitude'].max() - df['longitude'].min()) * (df['latitude'].max() - df['latitude'].min())
  n = len(df)
  expected_distance = 0.5 / np.sqrt(n / area)
  
  # Calculate R value to determine distribution pattern
  r = mean_nearest_distance / expected_distance
  
  # Determine pattern type
  if r < 1:
    pattern = "clustered"
  elif r > 1:
    pattern = "dispersed"
  else:
    pattern = "random"
    
  return {
    "mean_distance": mean_nearest_distance,
    "expected_distance": expected_distance,
    "r_value": r,
    "pattern": pattern
  }

def visualize_point_distribution(df, save_path=None):
  """
  Create scatter plot to visualize point distribution.
  
  Args:
    df: DataFrame with longitude and latitude columns
    save_path: Optional path to save the figure
  """
  plt.figure(figsize=(10, 8))
  plt.scatter(df['longitude'], df['latitude'], c='blue', s=50, alpha=0.6)
  plt.title('Scatter Plot of Commercial Facilities')
  plt.xlabel('Longitude')
  plt.ylabel('Latitude')
  
  if save_path:
    plt.savefig(save_path)
  plt.show()

def visualize_with_boundary(poi_df, boundary_gdf, boundary_df, save_path=None):
  """
  Visualize POI distribution with area boundary.
  
  Args:
    poi_df: DataFrame with POI data including longitude and latitude
    boundary_gdf: GeoDataFrame with boundary geometries
    boundary_df: DataFrame with boundary coordinates as longitude/latitude
    save_path: Optional path to save the figure
  """
  plt.figure(figsize=(12, 10))
  
  # Plot boundary if available
  if boundary_df is not None and len(boundary_df) > 0:
    # Plot boundary points as line
    plt.plot(
      boundary_df['longitude'], 
      boundary_df['latitude'], 
      'k-', 
      linewidth=1.5, 
      alpha=0.7,
      label='District Boundary'
    )
  
  # Plot POIs
  plt.scatter(
    poi_df['longitude'], 
    poi_df['latitude'], 
    c='blue', 
    s=10, 
    alpha=0.5,
    label='POI Points'
  )
  
  # Add KDE overlay
  if len(poi_df) > 10:  # Only add KDE if we have enough points
    try:
      sns.kdeplot(
        x=poi_df['longitude'], 
        y=poi_df['latitude'], 
        cmap="Reds", 
        alpha=0.4,
        levels=10,
        thresh=0.05
      )
    except Exception as e:
      print(f"KDE计算错误: {e}")
  
  plt.title('POI Distribution with Yuen Long District Boundary')
  plt.xlabel('Longitude')
  plt.ylabel('Latitude')
  plt.legend()
  plt.grid(True, linestyle='--', alpha=0.3)
  
  # Set limits based on combined data
  if boundary_df is not None and len(boundary_df) > 0:
    # Use boundary to set limits with some padding
    x_min = min(poi_df['longitude'].min(), boundary_df['longitude'].min())
    x_max = max(poi_df['longitude'].max(), boundary_df['longitude'].max())
    y_min = min(poi_df['latitude'].min(), boundary_df['latitude'].min())
    y_max = max(poi_df['latitude'].max(), boundary_df['latitude'].max())
    
    # Add some padding
    padding = 0.01
    plt.xlim(x_min - padding, x_max + padding)
    plt.ylim(y_min - padding, y_max + padding)
  
  if save_path:
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
  plt.show()

def ensure_directory_exists(path):
  """
  确保目录存在，如果不存在则创建
  """
  directory = os.path.dirname(path)
  if directory and not os.path.exists(directory):
    os.makedirs(directory)
    print(f"创建目录: {directory}")

def create_boundary_heatmap(poi_df, boundary_file='data/yuanlang.shp', save_path='yuenlong_with_boundary.png'):
  """
  创建带有正确元朗区边界的热力图
  使用完整的shapefile文件(包含.shp、.shx、.dbf、.prj)
  """
  print(f"创建带边界的热力图，使用边界文件: {boundary_file}")
  
  # 检查文件是否存在
  if not os.path.exists(boundary_file):
    raise FileNotFoundError(f"边界文件不存在: {boundary_file}")
  
  # 创建高质量的图像
  fig, ax = plt.subplots(figsize=(12, 10), dpi=300, facecolor='white')
  
  # 加载边界数据
  print(f"正在加载边界文件: {boundary_file}")
  boundary_gdf = gpd.read_file(boundary_file)
  print(f"成功加载边界数据，包含 {len(boundary_gdf)} 条记录")
  
  # 确保坐标系统为WGS84
  if boundary_gdf.crs and str(boundary_gdf.crs) != "EPSG:4326":
    print(f"转换坐标系，从 {boundary_gdf.crs} 到 EPSG:4326")
    boundary_gdf = boundary_gdf.to_crs("EPSG:4326")
  
  # 转换POI数据为GeoDataFrame
  poi_gdf = gpd.GeoDataFrame(
    poi_df, 
    geometry=gpd.points_from_xy(poi_df.longitude, poi_df.latitude),
    crs="EPSG:4326"
  )
  
  # 使用边界确定范围
  xmin, ymin, xmax, ymax = boundary_gdf.total_bounds
  # 小边距
  padding = 0.01
  
  # 创建紫色渐变色彩映射
  colors = [(1, 1, 1, 0),          # 透明白色（最低密度）
           (0.95, 0.9, 0.98, 0.3), # 超浅紫色（低密度） 
           (0.9, 0.8, 0.95, 0.5),  # 浅紫色（中低密度）
           (0.8, 0.5, 0.9, 0.7),   # 中紫色（中密度）
           (0.6, 0.3, 0.7, 0.8),   # 深紫色（高密度）
           (0.4, 0.1, 0.5, 0.9)]   # 暗紫色（最高密度）
  purple_cmap = LinearSegmentedColormap.from_list('custom_purple', colors)
  
  # 绘制边界
  boundary_gdf.boundary.plot(ax=ax, color='black', linewidth=1.5)
  
  # 创建网格并计算KDE
  print("计算核密度估计...")
  x = poi_gdf.geometry.x
  y = poi_gdf.geometry.y
  
  xx, yy = np.mgrid[xmin-padding:xmax+padding:200j, 
                    ymin-padding:ymax+padding:200j]
  positions = np.vstack([xx.ravel(), yy.ravel()])
  
  # 获取核密度估计
  kernel = gaussian_kde(np.vstack([x, y]), bw_method='scott')
  z = np.reshape(kernel(positions), xx.shape)
  
  # 裁剪边界外的数据
  try:
    from shapely.geometry import Point
    print("裁剪边界外的KDE值...")
    boundary_shape = boundary_gdf.geometry.unary_union
    
    # 使用掩码方法更有效地裁剪
    mask = np.zeros_like(z, dtype=bool)
    step = 5  # 采样步长以加快速度
    
    for i in range(0, xx.shape[0], step):
      for j in range(0, xx.shape[1], step):
        point = Point(xx[i, j], yy[i, j])
        in_boundary = boundary_shape.contains(point)
        
        # 设置该区域的掩码值
        i_max = min(xx.shape[0], i + step)
        j_max = min(xx.shape[1], j + step)
        mask[i:i_max, j:j_max] = in_boundary
    
    # 应用掩码
    z = np.where(mask, z, 0)
  except Exception as e:
    print(f"裁剪边界外的数据时出错: {e}")
    print("继续使用未裁剪的数据")
  
  # 绘制热力图
  contour_filled = ax.contourf(xx, yy, z, levels=20, cmap=purple_cmap, alpha=1.0)
  
  # 设置绘图范围
  ax.set_xlim(xmin-padding, xmax+padding)
  ax.set_ylim(ymin-padding, ymax+padding)
  
  # 移除坐标轴
  ax.set_xticks([])
  ax.set_yticks([])
  
  # 设置标题
  title = plt.title('The spatial distribution of cultural and recreational facilities in Yuen Long District', 
                   fontsize=14, pad=10)
  
  # 创建图例
  legend_ax = fig.add_axes([0.13, 0.12, 0.2, 0.03])
  legend_ax.set_title("Legend", fontsize=10, loc='left')
  legend_ax.axis('off')
  
  # 创建渐变色带
  gradient = np.linspace(0, 1, 100).reshape(1, -1)
  gradient = np.vstack((gradient, gradient))
  legend_ax.imshow(gradient, aspect='auto', cmap=purple_cmap)
  
  # 添加高低值标签
  z_max = z.max()
  legend_ax.text(-0.05, 0.5, f"低: 0", ha='right', va='center', fontsize=9)
  legend_ax.text(1.05, 0.5, f"高: {z_max:.4f}", ha='left', va='center', fontsize=9)
  
  # 添加北箭头
  north_ax = fig.add_axes([0.08, 0.2, 0.03, 0.07])
  north_ax.axis('off')
  north_ax.text(0.5, 0.1, 'N', ha='center', va='center', fontsize=12, fontweight='bold')
  north_ax.arrow(0.5, 0.3, 0, 0.6, head_width=0.3, head_length=0.2, fc='k', ec='k')
  
  # 添加比例尺
  scale_miles = 3  # 固定使用3英里
  miles_to_deg = 0.0145  # 在香港纬度下的近似值
  scale_deg = scale_miles * miles_to_deg
  
  # 创建比例尺
  scale_ax = fig.add_axes([0.7, 0.12, 0.2, 0.03])
  scale_ax.axis('off')
  
  # 绘制黑白刻度比例尺
  segments = 6
  segment_width = scale_deg / segments
  for i in range(segments):
    color = 'black' if i % 2 == 0 else 'white'
    scale_ax.add_patch(plt.Rectangle(
      (i * segment_width, 0), 
      segment_width, 
      0.2, 
      facecolor=color,
      edgecolor='black'
    ))
  
  # 添加比例尺标签
  scale_ax.text(0, -0.3, "0", ha='center', va='top', fontsize=8)
  scale_ax.text(scale_deg/2, -0.3, f"{scale_miles/2}", ha='center', va='top', fontsize=8)
  scale_ax.text(scale_deg, -0.3, f"{scale_miles}", ha='center', va='top', fontsize=8)
  scale_ax.text(scale_deg/2, -0.6, "Miles", ha='center', va='top', fontsize=8)
  
  # 保存图像
  try:
    # 确保results目录存在
    results_dir = os.path.dirname(save_path)
    if results_dir and not os.path.exists(results_dir):
      os.makedirs(results_dir)
      print(f"创建目录: {results_dir}")
    
    print(f"保存图像到: {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"图像已保存")
  except Exception as e:
    print(f"保存图像出错: {e}")
    # 尝试保存到当前目录
    alt_save_path = 'yuenlong_map.png'
    plt.savefig(alt_save_path, dpi=300)
    print(f"已保存到备用路径: {alt_save_path}")
  
  # 显示图像
  plt.show()
  return fig, ax

def main():
  """
  Main function to run the analysis pipeline.
  """
  # 加载POI数据
  poi_df = load_poi_data()
  print(f"加载了 {len(poi_df)} 个POI点")
  
  # 创建带边界的热力图，使用data/yuanlang.shp
  create_boundary_heatmap(poi_df, 
                         boundary_file='data/yuanlang.shp', 
                         save_path='results/yuenlong_with_boundary.png')

if __name__ == "__main__":
  main()