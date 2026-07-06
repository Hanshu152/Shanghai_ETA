import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import warnings
import json
from pathlib import Path

from pyproj import CRS, Transformer
from shapely.geometry import Point, LineString, shape
from shapely.ops import transform

warnings.filterwarnings('ignore')

# ==========================================
# 0. 核心配置区
# ==========================================
# 填入你最初始、未经特定清洗的原始数据
INPUT_FILE = "cleaned_mainstream_ais.csv" 
# vessel features 标记船舶静态特征
VESSEL_FILE = "target_vessels.csv"
# 导出的特征数据集文件名
OUTPUT_FEATURE_FILE = "voyages_features_pure_sailing_v5.csv"

# 中心线投影
CENTERLINE_FILE = "downstream_channel_centerline_relaxed_v3_manual_control.geojson"

# 中心线坐标顺序是否需要反转
REVERSE_CENTERLINE = False

# AIS点距离中心线超过该距离时，认为投影可信度较低
MAX_CROSS_TRACK_KM = 10.0

def haversine_vectorized(lon1, lat1, lon2, lat2):
    """向量化计算两点间球面距离 (km)"""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c

def load_centerline(centerline_file, reverse=False):
    """
    加载GeoJSON中心线并转换到以米为单位的局部等距投影。

    返回：
    centerline_m 米制坐标中心线
    transformer 经纬度到米制坐标的转换器
    """
    with open(centerline_file, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)

    if geojson_data['type'] == 'FeatureCollection':
        # 文件中同时包含一条中心线和大量中心线采样点，
        # 因此只提取其中的LineString要素
        line_features = [
            feature
            for feature in geojson_data['features']
            if (
                feature.get('geometry') is not None
                and feature['geometry'].get('type') == 'LineString'
            )
        ]

        if len(line_features) != 1:
            raise ValueError(
                f'中心线GeoJSON应当包含且只包含一条LineString，'
                f'实际找到 {len(line_features)} 条'
            )

        centerline_lonlat = shape(
            line_features[0]['geometry']
        )

    elif geojson_data['type'] == 'Feature':
        geometry = geojson_data.get('geometry')

        if geometry is None:
            raise ValueError(
                'GeoJSON Feature缺少geometry'
            )

        centerline_lonlat = shape(geometry)

    else:
        # 兼容文件本身直接就是LineString几何对象的情况
        centerline_lonlat = shape(geojson_data)

    if centerline_lonlat.geom_type != 'LineString':
        raise ValueError(
            f'中心线必须是LineString, 当前为'
            f'{centerline_lonlat.geom_type}'
        )

    if reverse:
        centerline_lonlat = LineString(
            list(centerline_lonlat.coords)[::-1]
        )

    # 使用中心线中心位置创建局部等距投影
    center = centerline_lonlat.centroid
    center_lon = center.x
    center_lat = center.y

    local_crs = CRS.from_proj4(
        f'+proj=aeqd '
        f'+lat_0={center_lat} '
        f'+lon_0={center_lon} '
        f'+datum=WGS84 '
        f'+units=m '
        f'+no_defs'
    )

    transformer = Transformer.from_crs(
        'EPSG:4326',
        local_crs,
        always_xy=True
    )

    centerline_m = transform(
        transformer.transform,
        centerline_lonlat
    )

    return centerline_m, transformer

def project_point_to_centerline(
    lon,
    lat,
    centerline_m,
    transformer
):
    """
    将AIS点投影到中心线。

    返回：
    route_s_km 从完整中心线起点到投影点的沿线里程
    cross_track_km AIS点到中心线的横向距离
    """
    if pd.isna(lon) or pd.isna(lat):
        return np.nan, np.nan

    x, y = transformer.transform(lon, lat)
    ais_point_m = Point(x, y)

    # 从中心线起点到最近投影点的累计沿线距离
    route_s_m = centerline_m.project(ais_point_m)

    # 得到中心线上的最近投影点
    projected_point_m = centerline_m.interpolate(route_s_m)

    # AIS原始位置与投影位置之间的横向距离
    cross_track_m = ais_point_m.distance(projected_point_m)

    return (
        route_s_m / 1000.0,
        cross_track_m / 1000.0
    )

print("🚀 启动 [清洗+特征+导出+训练+评估(含命中率)] 端到端 ETA 流水线...\n")

# ==========================================
# 1. 数据加载与 15 分钟重采样
# ==========================================

centerline_m, centerline_transformer = load_centerline(
    CENTERLINE_FILE,
    reverse=REVERSE_CENTERLINE
)

centerline_total_km = centerline_m.length / 1000.0

print(
    f'✔️ 中心航线加载完成，'
    f'沿线总长：{centerline_total_km:.2f} km'
)

print("▶️ Step 1: 加载原始数据与自动重采样")
df_raw = pd.read_csv(INPUT_FILE)
df_raw['postime'] = pd.to_datetime(df_raw['postime'])
df_raw = df_raw.sort_values(by=['voyage_id', 'postime'])
df_raw.set_index('postime', inplace=True)

# 聚合规则
agg_rules = {
    'mmsi': 'last',          
    'lon': 'last',           
    'lat': 'last',
    'sog': 'mean',           # 速度取均值平滑
    'cog': 'last',           
    'draught': 'last'
}
final_rules = {k: v for k, v in agg_rules.items() if k in df_raw.columns}

df_resampled = df_raw.groupby('voyage_id').resample('15min').agg(final_rules)
df_resampled = df_resampled.groupby(level=0).ffill().reset_index()

# 读取船舶静态特征表
vessel_static = pd.read_csv(VESSEL_FILE, encoding='utf-8-sig')
# 统一 mmsi 类型
df_resampled['mmsi'] = pd.to_numeric(df_resampled['mmsi'], errors='coerce').astype('Int64')
vessel_static['mmsi'] = pd.to_numeric(vessel_static['mmsi'], errors='coerce').astype('Int64')

vessel_static = vessel_static[['mmsi', '船舶类型', '船长', '船宽']].drop_duplicates(subset=['mmsi'])
vessel_static = vessel_static.rename(columns={
    '船舶类型': 'ship_type',
    '船长': 'ship_length',
    '船宽': 'ship_width'
})
# 数值字段转为数值类型
vessel_static['ship_length'] = pd.to_numeric(vessel_static['ship_length'], errors='coerce')
vessel_static['ship_width'] = pd.to_numeric(vessel_static['ship_width'], errors='coerce')
# 缺失填充
vessel_static['ship_type'] = vessel_static['ship_type'].fillna('未知')
vessel_static['ship_length'] = vessel_static['ship_length'].fillna(vessel_static['ship_length'].median())
vessel_static['ship_width'] = vessel_static['ship_width'].fillna(vessel_static['ship_width'].median())

# 船舶类型 one-hot 编码
ship_type_dummies = pd.get_dummies(
    vessel_static['ship_type'],
    prefix='ship_type',
    dtype=int
)

vessel_static = pd.concat(
    [vessel_static[['mmsi', 'ship_length', 'ship_width']], ship_type_dummies],
    axis=1
)
df_resampled = df_resampled.merge(vessel_static, on='mmsi', how='left')
# 合并后再次填充，防止 AIS 中有些 mmsi 不在静态表里
df_resampled['ship_length'] = df_resampled['ship_length'].fillna(df_resampled['ship_length'].median())
df_resampled['ship_width'] = df_resampled['ship_width'].fillna(df_resampled['ship_width'].median())

ship_type_features = [col for col in df_resampled.columns if col.startswith('ship_type_')]
df_resampled[ship_type_features] = df_resampled[ship_type_features].fillna(0).astype(int)

static_features = ['ship_length', 'ship_width'] + ship_type_features

print(f"✔️ 船舶静态特征合并完成，新增特征: {static_features}")

print(f"✔️ 重采样完成。数据量: {len(df_resampled)} 行\n")

# ==========================================
# 2. 纯航行时间标签清洗 (Reverse Cumsum)
# ==========================================
print("▶️ Step 2: 自动清洗锚泊时间，生成纯航行标签")
df = df_resampled.sort_values(by=['voyage_id', 'postime']).reset_index(drop=True)

# 定义：平均速度大于 0.5 节才算是在“有效航行”
df['is_sailing'] = df['sog'] > 0.5

# 如果在航行，这 15 分钟产生 0.25 小时的航行耗时；否则耗时为 0
df['step_hours'] = np.where(df['is_sailing'], 0.25, 0.0)

# [计算标签] 纯航行剩余时间 (逆向累加)
df['voyage_total_sailing_hours'] = df.groupby('voyage_id')['step_hours'].transform('sum')
df['pure_hours_elapsed'] = df.groupby('voyage_id')['step_hours'].cumsum() - df['step_hours']
df['pure_remaining_hours'] = df['voyage_total_sailing_hours'] - df['pure_hours_elapsed'] - df['step_hours']

# 剔除到达终点后以及无效的静止拖尾点
df = df[df['pure_remaining_hours'] > 0]
print(f"✔️ 纯航行标签挤水完成。有效航行样本: {len(df)} 行\n")

# ==========================================
# 3. V4 版本特征工程构建与导出
# ==========================================
print("▶️ Step 3: 构建 V4 核心特征空间并导出")

GATE_LON = 121.32963118277894
GATE_LAT = 31.544388261235497

voyage_context = df.groupby('voyage_id').agg(
    start_lon=('lon', 'first'),
    start_lat=('lat', 'first'),
    gate_lon=('lon', 'last'),
    gate_lat=('lat', 'last')
).reset_index()
df = df.merge(voyage_context, on='voyage_id', how='left')

df['hour_of_day'] = df['postime'].dt.hour
df['month'] = df['postime'].dt.month

df['prev_lon'] = df.groupby('voyage_id')['lon'].shift(1).fillna(df['lon'])
df['prev_lat'] = df.groupby('voyage_id')['lat'].shift(1).fillna(df['lat'])
df['step_dist_km'] = haversine_vectorized(df['prev_lon'], df['prev_lat'], df['lon'], df['lat'])
df['dist_traveled_km'] = df.groupby('voyage_id')['step_dist_km'].cumsum()

# 直接曲线距离
df['port_distance_to_gate'] = haversine_vectorized(df['start_lon'], df['start_lat'], GATE_LON, GATE_LAT)
df['dist_to_gate_km'] = haversine_vectorized(df['lon'], df['lat'],  GATE_LON, GATE_LAT)

# 沿线里程

projection_result = df.apply(
    lambda row: project_point_to_centerline(
        row['lon'],
        row['lat'],
        centerline_m,
        centerline_transformer
    ),
    axis=1,
    result_type='expand'
)

projection_result.columns = [
    'route_s_km',
    'cross_track_km'
]

df = pd.concat(
    [df, projection_result],
    axis=1
)

df = df.sort_values(
    ['voyage_id', 'postime']
).reset_index(drop=True)

# 每个航次第一个AIS点的中心线投影位置
df['start_route_s_km'] = (
    df.groupby('voyage_id')['route_s_km']
      .transform('first')
)

# 从航次实际起点投影位置到当前位置投影位置的距离
df['route_traveled_km'] = (
    df['route_s_km']
    - df['start_route_s_km']
)

# 负数通常来自AIS抖动或短暂回航
df['route_traveled_km_clipped'] = (
    df['route_traveled_km'].clip(lower=0)
)

gate_x, gate_y = centerline_transformer.transform(
    GATE_LON,
    GATE_LAT
)

gate_point_m = Point(gate_x, gate_y)

gate_route_s_km = (
    centerline_m.project(gate_point_m)
    / 1000.0
)

df['route_remaining_km'] = (
    gate_route_s_km
    - df['route_s_km']
).clip(lower=0)

# 检查异常比例 
df['route_projection_valid'] = (
    df['cross_track_km'] <= MAX_CROSS_TRACK_KM
).astype(int)
invalid_rate = (
    1 - df['route_projection_valid'].mean()
) * 100
print(
    f'中心线投影异常比例：{invalid_rate:.2f}%'
)

# 检查方向
df['route_s_delta_km'] = (
    df.groupby('voyage_id')['route_s_km'].diff()
)
print(
    df['route_s_delta_km'].describe()
)

df['avg_sog_all'] = df.groupby('voyage_id')['sog'].expanding().mean().reset_index(level=0, drop=True)
df['avg_sog_2h'] = df.groupby('voyage_id')['sog'].rolling(window=8, min_periods=1).mean().reset_index(level=0, drop=True)
df['avg_sog_4h'] = df.groupby('voyage_id')['sog'].rolling(window=16, min_periods=1).mean().reset_index(level=0, drop=True)
df['avg_speed_overall'] = df['dist_traveled_km'] / (df['pure_hours_elapsed'] + 1e-5)
df['avg_route_speed_knots'] = (
    df['route_traveled_km_clipped']
    / (df['pure_hours_elapsed'] + 1e-5)
    / 1.852
)

df = df.dropna(subset=['pure_remaining_hours', 'voyage_id'])

features = [
    'lon', 'lat', 'sog', 'cog', 'draught', 
    'hour_of_day', 'month', 
    'pure_hours_elapsed', 

    'dist_traveled_km', 'avg_speed_overall', 

    # 'route_s_km',
    # 'start_route_s_km',
    # 'route_traveled_km_clipped',
    'route_remaining_km',
    'cross_track_km',
    # 'route_projection_valid',
    'avg_route_speed_knots',

    'avg_sog_all', 'avg_sog_2h', 'avg_sog_4h',                                     
    'start_lon', 'start_lat', 'port_distance_to_gate', 
    # 'dist_to_gate_km',
    'ship_length', 'ship_width'
] + ship_type_features

features_distance_only = [
    'route_remaining_km'
]

# X = df[features]
# y = df['pure_remaining_hours']
# groups = df['voyage_id']

# ==========================================
# 按航次开始时间划分训练集、验证集、测试集
# ==========================================

# 每个 voyage_id 对应一个航次，只保留其开始时间
voyage_time = (
    df.groupby('voyage_id', as_index=False)
      .agg(voyage_start_time=('postime', 'min'))
      .sort_values('voyage_start_time')
      .reset_index(drop=True)
)
# 按航次数量划分：70%训练、15%验证、15%测试
train_ratio = 0.70
val_ratio = 0.15

n_voyages = len(voyage_time)
train_end = int(n_voyages * train_ratio)
val_end = int(n_voyages * (train_ratio + val_ratio))

train_voyage_ids = voyage_time.iloc[:train_end]['voyage_id']
val_voyage_ids = voyage_time.iloc[train_end:val_end]['voyage_id']
test_voyage_ids = voyage_time.iloc[val_end:]['voyage_id']

# 根据完整航次提取数据
train_df = df[df['voyage_id'].isin(train_voyage_ids)].copy()
val_df = df[df['voyage_id'].isin(val_voyage_ids)].copy()
test_df = df[df['voyage_id'].isin(test_voyage_ids)].copy()

# 各数据集按时间排序，便于检查
train_df = train_df.sort_values(['postime', 'voyage_id'])
val_df = val_df.sort_values(['postime', 'voyage_id'])
test_df = test_df.sort_values(['postime', 'voyage_id'])

# 构造模型输入和标签
X_train = train_df[features]
y_train = train_df['pure_remaining_hours']

X_val = val_df[features]
y_val = val_df['pure_remaining_hours']

X_test = test_df[features]
y_test = test_df['pure_remaining_hours']

print("\n时间顺序数据集划分完成：")
print(
    f"训练集：{train_df['voyage_id'].nunique()} 个航次，"
    f"{len(train_df)} 个样本"
)
print(
    f"验证集：{val_df['voyage_id'].nunique()} 个航次，"
    f"{len(val_df)} 个样本"
)
print(
    f"测试集：{test_df['voyage_id'].nunique()} 个航次，"
    f"{len(test_df)} 个样本"
)

print(
    f"训练时间：{train_df['postime'].min()} "
    f"至 {train_df['postime'].max()}"
)
print(
    f"验证时间：{val_df['postime'].min()} "
    f"至 {val_df['postime'].max()}"
)
print(
    f"测试时间：{test_df['postime'].min()} "
    f"至 {test_df['postime'].max()}"
)

# 导出干净的特征数据集
export_cols = ['voyage_id', 'mmsi', 'postime'] + features + ['pure_remaining_hours']
df[export_cols].to_csv(OUTPUT_FEATURE_FILE, index=False)
print(f"✔️ 特征构建完毕！已导出包含特征和标签的最终数据集至: {OUTPUT_FEATURE_FILE}\n")

# ==========================================
# 4. XGBoost 训练与交叉验证
# ==========================================
# print("▶️ Step 4: XGBoost GroupKFold(5折) 训练开始")
print("▶️ Step 4: XGBoost 时间顺序训练开始")
xgb_params = {
    'n_estimators': 1500,
    'learning_rate': 0.03,
    'max_depth': 15,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 3.0,
    'random_state': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 100  
}
xgb_model = xgb.XGBRegressor(**xgb_params)

xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
val_preds = xgb_model.predict(X_val)
test_preds = xgb_model.predict(X_test)
val_mae = mean_absolute_error(y_val, val_preds)
test_mae = mean_absolute_error(y_test, test_preds)

print(f"验证集 MAE: {val_mae:.4f} 小时")
print(f"测试集 MAE: {test_mae:.4f} 小时")
print(f"最佳迭代轮数: {xgb_model.best_iteration}")
feature_importances = xgb_model.feature_importances_

# ==========================================
# 5. 深度效果评估报告 (新增命中率计算)
# ==========================================
print("\n▶️ Step 5: 最终评估报告")
print("=" * 60)

# 合并所有验证集结果
# val_results_df = pd.concat(all_val_results, axis=0)
test_results_df = pd.DataFrame({
    'true_remaining_hours': y_test.to_numpy(),
    'pred_remaining_hours': test_preds,
    'route_remaining_km': X_test['route_remaining_km'].to_numpy()
})

# 计算绝对误差和命中标记 (误差 < 4 小时)
# val_results_df['abs_error'] = abs(val_results_df['true_remaining_hours'] - val_results_df['pred_remaining_hours'])
# val_results_df['is_hit'] = val_results_df['abs_error'] < 4.0
test_results_df['abs_error'] = abs(
    test_results_df['true_remaining_hours']
    - test_results_df['pred_remaining_hours']
)
test_results_df['is_hit'] = (
    test_results_df['abs_error'] < 4.0
)
global_mae = test_results_df['abs_error'].mean()
global_hit_rate = test_results_df['is_hit'].mean() * 100
# 1. 计算全局指标
# global_mae = np.mean(fold_maes)
# global_mae_std = np.std(fold_maes)
# global_hit_rate = val_results_df['is_hit'].mean() * 100

# print(f"🏆 全局纯航行 MAE: {global_mae:.4f} ± {global_mae_std:.4f} 小时")
print(f"🏆 测试集纯航行 MAE: {global_mae:.4f} 小时")
print(f"🎯 全局 4小时内命中率: {global_hit_rate:.2f}%")
print("-" * 60)

# 2. 计算按距离分段的指标 (漏斗评估)
bins = [0, 50, 200, 500, np.inf]
labels = ['<50km (临近)', '50-200km', '200-500km', '>500km (刚出发)']
test_results_df['distance_bucket'] = pd.cut(test_results_df['route_remaining_km'], bins=bins, labels=labels)

bucket_metrics = test_results_df.groupby('distance_bucket', observed=False).agg(
    mae=('abs_error', 'mean'),
    hit_rate=('is_hit', lambda x: x.mean() * 100),
    count=('abs_error', 'count')
).rename(columns={'mae': 'MAE(小时)', 'hit_rate': '命中率(%)', 'count': '样本量'})

print("\n🔍 按距离分段漏斗评估:")
# 格式化输出，让表格更好看
formatted_bucket = bucket_metrics.copy()
formatted_bucket['MAE(小时)'] = formatted_bucket['MAE(小时)'].map('{:.2f}'.format)
formatted_bucket['命中率(%)'] = formatted_bucket['命中率(%)'].map('{:.2f}%'.format)
print(formatted_bucket)
print("=" * 60)

# 3. 打印特征重要度
fi_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
print("\n🏅 Top 15 特征重要度:")
print(fi_df.to_string(index=False))