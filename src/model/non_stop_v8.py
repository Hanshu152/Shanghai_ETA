import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point, shape
from shapely.ops import transform
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')

# ==========================================
# 0. 核心配置区
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent
XGBOOST_DIR = next(
    path for path in [SCRIPT_DIR, *SCRIPT_DIR.parents]
    if path.name == "XGBoost"
)

TRAINING_DATA_DIR = XGBOOST_DIR / "data" / "training_data"
PIPELINE_DIR = XGBOOST_DIR / "src" / "pipeline"
MODEL_SOURCE_DIR = XGBOOST_DIR / "src" / "model"

# active_eta_v2 已经严格清洗并预先切好的三组点数据
ACTIVE_ETA_DIR = TRAINING_DATA_DIR
TRAIN_FILE = ACTIVE_ETA_DIR / "train_points_active_eta_v2.csv"
VAL_FILE = ACTIVE_ETA_DIR / "val_points_active_eta_v2.csv"
TEST_FILE = ACTIVE_ETA_DIR / "test_points_active_eta_v2.csv"

# 船舶静态特征表
VESSEL_FILE = MODEL_SOURCE_DIR / "target_vessels.csv"

# 导出的特征数据集文件名
OUTPUT_FEATURE_FILE = TRAINING_DATA_DIR / "voyages_features_active_eta_v8.csv"

# 训练完成后导出的模型和特征元数据
MODEL_FILE = PIPELINE_DIR / "xgboost_v8_model.json"
FEATURE_META_FILE = PIPELINE_DIR / "xgboost_v8_features.json"

# 中心航线
CENTERLINE_FILE = MODEL_SOURCE_DIR / "downstream_channel_centerline_relaxed_v3_manual_control.geojson"

# 中心线坐标顺序是否需要反转
REVERSE_CENTERLINE = False

# AIS 点距离中心线超过该距离时，认为投影可信度较低
MAX_CROSS_TRACK_KM = 10.0

# 终点坐标：取中心线最后一个点
GATE_LON = 121.5006300969005
GATE_LAT = 31.425158671076692


def haversine_vectorized(lon1, lat1, lon2, lat2):
    """向量化计算两点间球面距离，单位 km。"""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c


def load_centerline(centerline_file, reverse=False):
    """
    加载 GeoJSON 中心线，并转换到以米为单位的局部等距投影。

    返回：
    centerline_m：米制坐标中心线
    transformer：经纬度到米制坐标的转换器
    centerline_lonlat：原始经纬度中心线
    """
    with open(centerline_file, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)

    if geojson_data['type'] == 'FeatureCollection':
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
                '中心线 GeoJSON 应当包含且只包含一条 LineString，'
                f'实际找到 {len(line_features)} 条'
            )

        centerline_lonlat = shape(line_features[0]['geometry'])

    elif geojson_data['type'] == 'Feature':
        geometry = geojson_data.get('geometry')

        if geometry is None:
            raise ValueError('GeoJSON Feature 缺少 geometry')

        centerline_lonlat = shape(geometry)

    else:
        centerline_lonlat = shape(geojson_data)

    if centerline_lonlat.geom_type != 'LineString':
        raise ValueError(
            f'中心线必须是 LineString，当前为：{centerline_lonlat.geom_type}'
        )

    if reverse:
        centerline_lonlat = LineString(list(centerline_lonlat.coords)[::-1])

    center = centerline_lonlat.centroid
    local_crs = CRS.from_proj4(
        f'+proj=aeqd '
        f'+lat_0={center.y} '
        f'+lon_0={center.x} '
        f'+datum=WGS84 '
        f'+units=m '
        f'+no_defs'
    )

    transformer = Transformer.from_crs(
        'EPSG:4326',
        local_crs,
        always_xy=True
    )

    centerline_m = transform(transformer.transform, centerline_lonlat)

    return centerline_m, transformer, centerline_lonlat


def project_point_to_centerline(lon, lat, centerline_m, transformer):
    """
    将 AIS 点投影到中心线。

    返回：
    route_s_km：从完整中心线起点到投影点的沿线里程
    cross_track_km：AIS 点到中心线的横向距离
    """
    if pd.isna(lon) or pd.isna(lat):
        return np.nan, np.nan

    x, y = transformer.transform(lon, lat)
    ais_point_m = Point(x, y)

    route_s_m = centerline_m.project(ais_point_m)
    projected_point_m = centerline_m.interpolate(route_s_m)
    cross_track_m = ais_point_m.distance(projected_point_m)

    return route_s_m / 1000.0, cross_track_m / 1000.0


def load_active_eta_split(file_path, split_name):
    """读取 active_eta_v2 点数据，并映射成 non_stop 模型使用的统一字段。"""
    df = pd.read_csv(file_path)
    df['split'] = split_name

    required_cols = [
        'source_berth_uuid',
        'mmsi',
        'postime',
        'lon',
        'lat',
        'sog',
        'draught',
        'cog',
        'active_elapsed_h_v2',
        'active_total_h_v2',
        'active_remaining_h_v2',
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f'{file_path.name} 缺少必要字段：{missing_cols}'
        )

    df = df.rename(
        columns={
            'source_berth_uuid': 'voyage_id',
            'active_elapsed_h_v2': 'pure_hours_elapsed',
            'active_total_h_v2': 'voyage_total_sailing_hours',
            'active_remaining_h_v2': 'pure_remaining_hours',
        }
    )

    df['postime'] = pd.to_datetime(df['postime'], utc=True, errors='coerce')
    df['mmsi'] = pd.to_numeric(df['mmsi'], errors='coerce').astype('Int64')

    numeric_cols = [
        'lon',
        'lat',
        'sog',
        'draught',
        'cog',
        'pure_hours_elapsed',
        'voyage_total_sailing_hours',
        'pure_remaining_hours',
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(
        subset=[
            'voyage_id',
            'postime',
            'lon',
            'lat',
            'sog',
            'pure_hours_elapsed',
            'pure_remaining_hours',
        ]
    )
    df = df[df['pure_remaining_hours'] > 0].copy()

    return df


def load_vessel_static():
    """读取船舶静态特征，并对船舶类型做 one-hot 编码。"""
    vessel_static = pd.read_csv(VESSEL_FILE, encoding='utf-8-sig')
    vessel_static['mmsi'] = pd.to_numeric(
        vessel_static['mmsi'],
        errors='coerce'
    ).astype('Int64')

    vessel_static = vessel_static[
        ['mmsi', '船舶类型', '船长', '船宽']
    ].drop_duplicates(subset=['mmsi'])
    vessel_static = vessel_static.rename(
        columns={
            '船舶类型': 'ship_type',
            '船长': 'ship_length',
            '船宽': 'ship_width',
        }
    )

    vessel_static['ship_length'] = pd.to_numeric(
        vessel_static['ship_length'],
        errors='coerce'
    )
    vessel_static['ship_width'] = pd.to_numeric(
        vessel_static['ship_width'],
        errors='coerce'
    )

    vessel_static['ship_type'] = vessel_static['ship_type'].fillna('未知')
    vessel_static['ship_length'] = vessel_static['ship_length'].fillna(
        vessel_static['ship_length'].median()
    )
    vessel_static['ship_width'] = vessel_static['ship_width'].fillna(
        vessel_static['ship_width'].median()
    )

    ship_type_dummies = pd.get_dummies(
        vessel_static['ship_type'],
        prefix='ship_type',
        dtype=int
    )

    vessel_static = pd.concat(
        [
            vessel_static[['mmsi', 'ship_length', 'ship_width']],
            ship_type_dummies,
        ],
        axis=1
    )

    return vessel_static


def add_time_rolling_sog(df):
    """按航次计算历史平均速度和 2h/4h 时间窗口平均速度。"""
    df = df.sort_values(['voyage_id', 'postime']).copy()
    df['avg_sog_all'] = (
        df.groupby('voyage_id')['sog']
          .expanding()
          .mean()
          .reset_index(level=0, drop=True)
    )

    rolling_parts = []
    for _, group in df.groupby('voyage_id', sort=False):
        group = group.sort_values('postime').copy()
        group_indexed = group.set_index('postime')
        group['avg_sog_2h'] = (
            group_indexed['sog']
            .rolling('2h', min_periods=1)
            .mean()
            .to_numpy()
        )
        group['avg_sog_4h'] = (
            group_indexed['sog']
            .rolling('4h', min_periods=1)
            .mean()
            .to_numpy()
        )
        rolling_parts.append(group)

    return pd.concat(rolling_parts, axis=0).sort_index()


print("启动 non_stop_v8：读取 active_eta_v2 三组清洗数据并训练 ETA 模型...\n")

# ==========================================
# 1. 加载中心线和 active_eta_v2 三组数据
# ==========================================
centerline_m, centerline_transformer, centerline_lonlat = load_centerline(
    CENTERLINE_FILE,
    reverse=REVERSE_CENTERLINE
)
centerline_total_km = centerline_m.length / 1000.0
print(f"中心航线加载完成，沿线总长：{centerline_total_km:.2f} km")

train_df = load_active_eta_split(TRAIN_FILE, 'train')
val_df = load_active_eta_split(VAL_FILE, 'val')
test_df = load_active_eta_split(TEST_FILE, 'test')

df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
df = df.sort_values(['voyage_id', 'postime']).reset_index(drop=True)

print(
    "active_eta_v2 数据加载完成："
    f"train={len(train_df)} 行，"
    f"val={len(val_df)} 行，"
    f"test={len(test_df)} 行"
)

# ==========================================
# 2. 合并船舶静态特征
# ==========================================
vessel_static = load_vessel_static()
df = df.merge(vessel_static, on='mmsi', how='left')

df['ship_length'] = df['ship_length'].fillna(df['ship_length'].median())
df['ship_width'] = df['ship_width'].fillna(df['ship_width'].median())

ship_type_features = sorted(
    [col for col in df.columns if col.startswith('ship_type_')]
)
df[ship_type_features] = df[ship_type_features].fillna(0).astype(int)

static_features = ['ship_length', 'ship_width'] + ship_type_features
print(f"船舶静态特征合并完成，新增特征：{static_features}")

# ==========================================
# 3. 特征工程
# ==========================================
print("开始构建 v8 特征空间...")

voyage_context = df.groupby('voyage_id').agg(
    start_lon=('lon', 'first'),
    start_lat=('lat', 'first'),
).reset_index()
df = df.merge(voyage_context, on='voyage_id', how='left')

df['hour_of_day'] = df['postime'].dt.hour
df['month'] = df['postime'].dt.month

df['prev_lon'] = df.groupby('voyage_id')['lon'].shift(1).fillna(df['lon'])
df['prev_lat'] = df.groupby('voyage_id')['lat'].shift(1).fillna(df['lat'])
df['step_dist_km'] = haversine_vectorized(
    df['prev_lon'],
    df['prev_lat'],
    df['lon'],
    df['lat']
)
df['dist_traveled_km'] = df.groupby('voyage_id')['step_dist_km'].cumsum()

df['port_distance_to_gate'] = haversine_vectorized(
    df['start_lon'],
    df['start_lat'],
    GATE_LON,
    GATE_LAT
)
df['dist_to_gate_km'] = haversine_vectorized(
    df['lon'],
    df['lat'],
    GATE_LON,
    GATE_LAT
)

projection_result = [
    project_point_to_centerline(
        lon,
        lat,
        centerline_m,
        centerline_transformer
    )
    for lon, lat in zip(df['lon'], df['lat'])
]
projection_result = pd.DataFrame(
    projection_result,
    columns=['route_s_km', 'cross_track_km'],
    index=df.index
)
df = pd.concat([df, projection_result], axis=1)

df = df.sort_values(['voyage_id', 'postime']).reset_index(drop=True)
df['start_route_s_km'] = (
    df.groupby('voyage_id')['route_s_km']
      .transform('first')
)
df['route_traveled_km'] = df['route_s_km'] - df['start_route_s_km']
df['route_traveled_km_clipped'] = df['route_traveled_km'].clip(lower=0)

gate_x, gate_y = centerline_transformer.transform(GATE_LON, GATE_LAT)
gate_point_m = Point(gate_x, gate_y)
gate_route_s_km = centerline_m.project(gate_point_m) / 1000.0

df['route_remaining_km'] = (
    gate_route_s_km - df['route_s_km']
).clip(lower=0)

df['route_projection_valid'] = (
    df['cross_track_km'] <= MAX_CROSS_TRACK_KM
).astype(int)
invalid_rate = (1 - df['route_projection_valid'].mean()) * 100
print(f"中心线投影异常比例：{invalid_rate:.2f}%")

df = add_time_rolling_sog(df)

df['avg_speed_overall'] = (
    df['dist_traveled_km'] / (df['pure_hours_elapsed'] + 1e-5)
)
df['avg_route_speed_knots'] = (
    df['route_traveled_km_clipped']
    / (df['pure_hours_elapsed'] + 1e-5)
    / 1.852
)

df = df.dropna(subset=['pure_remaining_hours', 'voyage_id'])

features = [
    'lon',
    'lat',
    'sog',
    'cog',
    'draught',
    'hour_of_day',
    'month',
    'pure_hours_elapsed',
    'dist_traveled_km',
    'avg_speed_overall',
    'route_remaining_km',
    'cross_track_km',
    'avg_route_speed_knots',
    'avg_sog_all',
    'avg_sog_2h',
    'avg_sog_4h',
    'start_lon',
    'start_lat',
    'port_distance_to_gate',
    'ship_length',
    'ship_width',
] + ship_type_features

df = df.dropna(subset=features + ['pure_remaining_hours'])

# ==========================================
# 4. 使用 active_eta_v2 原始三组划分训练、验证、测试
# ==========================================
train_df = df[df['split'] == 'train'].copy()
val_df = df[df['split'] == 'val'].copy()
test_df = df[df['split'] == 'test'].copy()

train_df = train_df.sort_values(['postime', 'voyage_id'])
val_df = val_df.sort_values(['postime', 'voyage_id'])
test_df = test_df.sort_values(['postime', 'voyage_id'])

X_train = train_df[features]
y_train = train_df['pure_remaining_hours']

X_val = val_df[features]
y_val = val_df['pure_remaining_hours']

X_test = test_df[features]
y_test = test_df['pure_remaining_hours']

print("\nactive_eta_v2 数据集划分：")
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
print(f"训练时间：{train_df['postime'].min()} 至 {train_df['postime'].max()}")
print(f"验证时间：{val_df['postime'].min()} 至 {val_df['postime'].max()}")
print(f"测试时间：{test_df['postime'].min()} 至 {test_df['postime'].max()}")

export_cols = [
    'split',
    'voyage_id',
    'mmsi',
    'postime',
] + features + ['pure_remaining_hours']
TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
df[export_cols].to_csv(OUTPUT_FEATURE_FILE, index=False)
print(f"特征数据已导出至：{OUTPUT_FEATURE_FILE}\n")

# ==========================================
# 5. XGBoost 训练
# ==========================================
print("Step 5: XGBoost 训练开始")
xgb_params = {
    'n_estimators': 1500,
    'learning_rate': 0.03,
    'max_depth': 15,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 3.0,
    'random_state': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 100,
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

PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
xgb_model.save_model(MODEL_FILE)
feature_meta = {
    'features': features,
    'ship_type_features': ship_type_features,
}
FEATURE_META_FILE.write_text(
    json.dumps(feature_meta, ensure_ascii=False, indent=2),
    encoding='utf-8'
)
print(f"模型已保存至: {MODEL_FILE}")
print(f"特征元数据已保存至: {FEATURE_META_FILE}")

# ==========================================
# 6. 评估报告
# ==========================================
print("\nStep 6: 最终评估报告")
print("=" * 60)

test_results_df = pd.DataFrame({
    'true_remaining_hours': y_test.to_numpy(),
    'pred_remaining_hours': test_preds,
    'route_remaining_km': X_test['route_remaining_km'].to_numpy(),
})

test_results_df['abs_error'] = abs(
    test_results_df['true_remaining_hours']
    - test_results_df['pred_remaining_hours']
)
test_results_df['is_hit_2h'] = test_results_df['abs_error'] < 2.0
test_results_df['is_hit_4h'] = test_results_df['abs_error'] < 4.0

global_mae = test_results_df['abs_error'].mean()
global_hit_rate_2h = test_results_df['is_hit_2h'].mean() * 100
global_hit_rate_4h = test_results_df['is_hit_4h'].mean() * 100

business_window_df = test_results_df[
    test_results_df['true_remaining_hours'].between(10.0, 34.0, inclusive='both')
]
business_window_mae = business_window_df['abs_error'].mean()
business_window_hit_rate_2h = business_window_df['is_hit_2h'].mean() * 100
business_window_hit_rate_4h = business_window_df['is_hit_4h'].mean() * 100

print(f"测试集 active ETA MAE: {global_mae:.4f} 小时")
print(f"全局 2 小时内命中率: {global_hit_rate_2h:.2f}%")
print(f"全局 4 小时内命中率: {global_hit_rate_4h:.2f}%")
print(f"业务需求窗口(真实剩余 10-34 小时)样本量: {len(business_window_df)}")
print(f"业务需求窗口 MAE: {business_window_mae:.4f} 小时")
print(f"业务需求窗口 2 小时内命中率: {business_window_hit_rate_2h:.2f}%")
print(f"业务需求窗口 4 小时内命中率: {business_window_hit_rate_4h:.2f}%")
print("-" * 60)

bins = [0, 50, 200, 500, np.inf]
labels = ['<50km (临近)', '50-200km', '200-500km', '>500km (刚出发)']
test_results_df['distance_bucket'] = pd.cut(
    test_results_df['route_remaining_km'],
    bins=bins,
    labels=labels
)

bucket_metrics = test_results_df.groupby(
    'distance_bucket',
    observed=False
).agg(
    mae=('abs_error', 'mean'),
    hit_rate_2h=('is_hit_2h', lambda x: x.mean() * 100),
    hit_rate_4h=('is_hit_4h', lambda x: x.mean() * 100),
    count=('abs_error', 'count')
).rename(
    columns={
        'mae': 'MAE(小时)',
        'hit_rate_2h': '2小时命中率(%)',
        'hit_rate_4h': '4小时命中率(%)',
        'count': '样本量',
    }
)

print("\n按沿线剩余距离分段漏斗评估:")
formatted_bucket = bucket_metrics.copy()
formatted_bucket['MAE(小时)'] = formatted_bucket['MAE(小时)'].map('{:.2f}'.format)
formatted_bucket['2小时命中率(%)'] = formatted_bucket['2小时命中率(%)'].map('{:.2f}%'.format)
formatted_bucket['4小时命中率(%)'] = formatted_bucket['4小时命中率(%)'].map('{:.2f}%'.format)
print(formatted_bucket)
print("=" * 60)

fi_df = pd.DataFrame(
    {
        'Feature': features,
        'Importance': feature_importances,
    }
).sort_values(by='Importance', ascending=False)
print("\nTop 15 特征重要度:")
print(fi_df.head(15).to_string(index=False))
