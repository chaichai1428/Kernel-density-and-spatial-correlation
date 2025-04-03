import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import distance_matrix
import os
import glob
from matplotlib.patches import Patch

def nearest_neighbor_analysis(input_file=None, output_dir='results'):
  """
  进行最近邻分析并生成图表
  
  参数:
    input_file: 输入CSV文件，包含点位置数据
    output_dir: 输出目录
  """
  # 查找数据文件 - 更广泛地搜索
  if input_file is None:
    csv_files = glob.glob('./data/*.csv') + glob.glob('./results/*.csv') + glob.glob('./*.csv')
    if csv_files:
      input_file = csv_files[0]
      print(f"使用数据文件: {input_file}")
    else:
      print("错误: 找不到数据文件")
      return
      
  # 读取数据
  try:
    df = pd.read_csv(input_file)
    print(f"成功读取文件: {input_file}")
    print(f"数据列: {df.columns.tolist()}")
    
    # 特别处理location字段
    if 'location' in df.columns:
      print("发现location字段，提取经纬度...")
      df[['longitude', 'latitude']] = df['location'].str.split(',', expand=True).astype(float)
    
    if 'longitude' not in df.columns or 'latitude' not in df.columns:
      if 'x' in df.columns and 'y' in df.columns:
        df['longitude'] = df['x']
        df['latitude'] = df['y']
      else:
        print("错误: 找不到坐标列 (longitude/latitude 或 x/y)")
        return
        
    print(f"找到 {len(df)} 个点位")
    print(f"经度范围: {df['longitude'].min()} - {df['longitude'].max()}")
    print(f"纬度范围: {df['latitude'].min()} - {df['latitude'].max()}")
  except Exception as e:
    print(f"读取文件错误: {e}")
    return
    
  # 计算自定义最近邻分析
  coords = df[['longitude', 'latitude']].values
  
  # 计算点之间的距离矩阵
  print("计算距离矩阵...")
  dist_matrix = distance_matrix(coords, coords)
  
  # 将对角线设为无穷大（避免点与自身的距离）
  np.fill_diagonal(dist_matrix, np.inf)
  
  # 计算每个点到最近邻的距离
  nearest_distances = np.min(dist_matrix, axis=1)
  
  # 计算平均最近邻距离
  mean_nearest_distance = np.mean(nearest_distances)
  
  # 计算研究区域面积 (使用凸包面积更准确，这里简化为矩形面积)
  x_range = df['longitude'].max() - df['longitude'].min()
  y_range = df['latitude'].max() - df['latitude'].min()
  area = x_range * y_range
  
  # 点的数量
  n = len(coords)
  
  # 计算随机分布下的预期平均最近邻距离
  expected_distance = 0.5 * np.sqrt(area / n)
  
  # 计算最近邻指数 (NNR)
  nnr = mean_nearest_distance / expected_distance
  
  # 计算标准误差
  se = 0.26136 / np.sqrt(n * n / area)
  
  # 计算z分数
  z_score = (mean_nearest_distance - expected_distance) / se
  
  # 计算p值
  p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # 双尾检验
  if z_score < 0:  # 如果是聚集模式，只看左尾
    p_value = stats.norm.cdf(z_score)
  
  print(f"最近邻比率 (NNR): {nnr:.6f}")
  print(f"Z分数: {z_score:.6f}")
  print(f"p值: {p_value:.6f}")
  
  # 显示结论
  if p_value < 0.05:
    if z_score < 0:
      pattern = "聚集模式"
    else:
      pattern = "分散模式"
    significance = "显著"
  else:
    pattern = "随机模式"
    significance = "不显著"
    
  print(f"结论: 点的分布为{pattern} ({significance})")
  
  # 创建更大的图形以容纳不重叠的内容
  fig = plt.figure(figsize=(12, 12))
  
  # 给主图形分配更多的上部空间
  main_ax = fig.add_axes([0.1, 0.4, 0.8, 0.5])  # [left, bottom, width, height]
  
  # 创建正态分布曲线
  x = np.linspace(-4, 4, 1000)
  y = stats.norm.pdf(x)
  
  # 确保y是一个数组，并计算y_max用于垂直定位
  y_max = np.max(y)
  
  # 计算不同显著性水平的临界值
  z_critical_values = {
    0.01: -2.58,  # 显著聚集，p<0.01
    0.05: -1.96,  # 显著聚集，p<0.05
    0.10: -1.65,  # 显著聚集，p<0.10
    0.90: 1.65,   # 显著分散，p<0.10
    0.95: 1.96,   # 显著分散，p<0.05
    0.99: 2.58    # 显著分散，p<0.01
  }
  
  # 填充曲线下的不同区域
  main_ax.fill_between(x, y, where=(x <= z_critical_values[0.01]), color='#0b5394', alpha=0.6)
  main_ax.fill_between(x, y, where=((x > z_critical_values[0.01]) & (x <= z_critical_values[0.05])), color='#3d85c6', alpha=0.6)
  main_ax.fill_between(x, y, where=((x > z_critical_values[0.05]) & (x <= z_critical_values[0.10])), color='#9fc5e8', alpha=0.6)
  main_ax.fill_between(x, y, where=((x >= z_critical_values[0.90]) & (x < z_critical_values[0.95])), color='#f4cccc', alpha=0.6)
  main_ax.fill_between(x, y, where=((x >= z_critical_values[0.95]) & (x < z_critical_values[0.99])), color='#ea9999', alpha=0.6)
  main_ax.fill_between(x, y, where=(x >= z_critical_values[0.99]), color='#cc0000', alpha=0.6)
  main_ax.fill_between(x, y, where=((x > z_critical_values[0.10]) & (x < z_critical_values[0.90])), color='#ffe599', alpha=0.6)
  
  # 绘制中心线
  main_ax.plot([0, 0], [0, y_max*1.1], 'k--', alpha=0.5)
  
  # 绘制密度曲线
  main_ax.plot(x, y, 'k-', linewidth=1.5)
  
  # 添加Z分数标记
  main_ax.annotate('', xy=(z_score, 0), xytext=(z_score, y_max*0.2),
               arrowprops=dict(facecolor='blue', shrink=0.05))
  
  # 设置轴标签和标题
  main_ax.set_ylim(0, y_max*1.1)
  main_ax.set_xlim(-3, 3)
  main_ax.set_xlabel('Z-Score', fontsize=12)
  main_ax.set_yticks([])  # 移除Y轴刻度
  
  # 在左侧和右侧添加"Significant"标签
  main_ax.text(-2.8, y_max*0.1, 'Significant', fontsize=12, ha='left')
  main_ax.text(2.8, y_max*0.1, 'Significant', fontsize=12, ha='right')
  main_ax.text(0, y_max*0.3, '(Random)', fontsize=12, ha='center')
  
  # 添加结果文本框
  result_text = (
    f"NNR     {nnr:.6f}\n"
    f"z-分数  {z_score:.6f}\n"
    f"p 值    {p_value:.6f}"
  )
  main_ax.text(-2.8, y_max*0.85, result_text, fontsize=12, va='top', fontweight='bold',
          bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))
  
  # 添加图例和临界值说明
  legend_x = 1.15
  main_ax.text(legend_x, y_max*0.9, 'Significance Level\n(p-value)', fontsize=12, ha='left')
  main_ax.text(legend_x, y_max*0.75, '0.01\n0.05\n0.10\n0.10\n0.05\n0.01', fontsize=12, ha='left')
  
  main_ax.text(legend_x, y_max*0.55, 'Critical Value\n(z-score)', fontsize=12, ha='left')
  main_ax.text(legend_x, y_max*0.4, '< -2.58\n-2.58 - -1.96\n-1.96 - -1.65\n1.65 - 1.96\n1.96 - 2.58\n> 2.58', fontsize=12, ha='left')
  
  # 添加点模式示例图 - 移到底部
  # 聚集模式 (左)
  clustered_points = []
  
  # 创建5个聚集中心
  centers = [(0.2, 0.2), (0.8, 0.8), (0.2, 0.8), (0.8, 0.2), (0.5, 0.5)]
  for center_x, center_y in centers:
    # 在每个中心周围添加10个点
    for _ in range(10):
      x = center_x + np.random.normal(0, 0.05)
      y = center_y + np.random.normal(0, 0.05)
      clustered_points.append((x, y))
  
  clustered = np.array(clustered_points)
  
  # 随机模式 (中)
  random_points = np.random.uniform(0, 1, (50, 2))
  
  # 分散模式 (右)
  dispersed_points = []
  grid_size = int(np.sqrt(50))
  spacing = 1.0 / (grid_size + 1)
  
  for i in range(1, grid_size + 1):
    for j in range(1, grid_size + 1):
      if len(dispersed_points) < 50:
        x = i * spacing + np.random.normal(0, 0.01)
        y = j * spacing + np.random.normal(0, 0.01)
        dispersed_points.append((x, y))
  
  dispersed = np.array(dispersed_points)
  
  # 创建三个示例子图 - 放在底部
  ax_c = fig.add_axes([0.1, 0.1, 0.25, 0.25])  # 左下角
  ax_r = fig.add_axes([0.375, 0.1, 0.25, 0.25])  # 中下角
  ax_d = fig.add_axes([0.65, 0.1, 0.25, 0.25])  # 右下角
  
  # 绘制点
  ax_c.scatter(clustered[:, 0], clustered[:, 1], s=10, c='black')
  ax_r.scatter(random_points[:, 0], random_points[:, 1], s=10, c='black')
  ax_d.scatter(dispersed[:, 0], dispersed[:, 1], s=10, c='black')
  
  # 设置样式
  for ax, title in zip([ax_c, ax_r, ax_d], ['Clustered', 'Random', 'Dispersed']):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.text(0.5, -0.1, title, ha='center', fontsize=10, transform=ax.transAxes)
    
    # 为当前模式添加特殊边框
    if (title == 'Clustered' and pattern == "聚集模式") or \
       (title == 'Random' and pattern == "随机模式") or \
       (title == 'Dispersed' and pattern == "分散模式"):
      for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('#1F77B4')
    else:
      for spine in ax.spines.values():
        spine.set_linewidth(1)
        spine.set_color('#CCCCCC')
  
  # 创建输出目录
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
  # 保存图像
  output_file = os.path.join(output_dir, 'nearest_neighbor_analysis.png')
  plt.savefig(output_file, dpi=300, bbox_inches='tight')
  print(f"图像已保存为: {output_file}")
  
  # 显示图形
  plt.show()
  
  return {
    'nnr': nnr,
    'z_score': z_score,
    'p_value': p_value
  }

if __name__ == "__main__":
  import argparse
  
  parser = argparse.ArgumentParser(description='进行最近邻分析')
  parser.add_argument('--input', help='输入CSV文件路径')
  parser.add_argument('--output', default='results', help='输出目录')
  
  args = parser.parse_args()
  
  nearest_neighbor_analysis(args.input, args.output)