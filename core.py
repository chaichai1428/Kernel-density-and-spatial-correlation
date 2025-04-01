import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance_matrix
import geopandas as gpd

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

def load_boundary_data(boundary_file='data/yuanlang.shp'):
  """
  Load boundary information from shapefile.
  
  Args:
    boundary_file: Path to the boundary shapefile
    
  Returns:
    GeoDataFrame with boundary geometry
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
    return None, 

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

def main():
  # Load POI data
  poi_df = load_poi_data()
  
  # Load boundary data
  boundary_gdf, boundary_df = load_boundary_data()
  
  # Perform KDE analysis
  perform_kde_analysis(poi_df)
  
  # Perform nearest neighbor analysis
  results = nearest_neighbor_analysis(poi_df)
  print(f"平均最近邻距离: {results['mean_distance']:.4f}")
  print(f"理论随机分布的期望距离: {results['expected_distance']:.4f}")
  print(f"R = {results['r_value']:.4f}, 表示分布模式为{results['pattern']}")
  
  # Visualize with boundary
  if boundary_gdf is not None:
    visualize_with_boundary(poi_df, boundary_gdf, boundary_df)
  else:
    # Fallback to regular visualization
    visualize_point_distribution(poi_df)

if __name__ == "__main__":
  main()