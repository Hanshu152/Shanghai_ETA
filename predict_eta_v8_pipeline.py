import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point, shape
from shapely.ops import transform


BASE_DIR = Path(__file__).resolve().parent

INPUT_AIS_FILE = BASE_DIR / "ais_6his_6h.csv"
INPUT_ETA_FILE = BASE_DIR / "eta_6his_6h.csv"
RESEARCH_VESSEL_FILE = (
    BASE_DIR / "predicted_crossing_20260630_20260701_bj_6his6h_simple(1).csv"
)
VESSEL_FILE = BASE_DIR / "target_vessels.csv"
CENTERLINE_FILE = BASE_DIR / "downstream_channel_centerline_relaxed_v3_manual_control.geojson"

MODEL_FILE = BASE_DIR / "xgboost_v8_model.json"
FEATURE_META_FILE = BASE_DIR / "xgboost_v8_features.json"
OUTPUT_FILE = BASE_DIR / "eta_predictions_v8.csv"
RESEARCH_OUTPUT_FILE = (
    BASE_DIR
    / "predicted_crossing_20260630_20260701_bj_6his6h_simple_with_v8_eta.csv"
)
ARRIVAL_JUN30_FILE = BASE_DIR / "eta_predictions_arrive_2026-06-30.csv"
ARRIVAL_JUL01_FILE = BASE_DIR / "eta_predictions_arrive_2026-07-01.csv"

LOCAL_TZ = "Asia/Shanghai"
USE_RESEARCH_VESSEL_LIST = True

REVERSE_CENTERLINE = False
MAX_CROSS_TRACK_KM = 10.0

# 外高桥港终点坐标：中心线最后一个点
GATE_LON = 121.5006300969005
GATE_LAT = 31.425158671076692

SHANGHAI_DEST_PREFIXES = {
    'SHANGHA',
    'SH',
    'SHH',
    'SHAIHAI',
    'SHAMGHAI',
    'SHANGGAI',
    'SHANGHAII',
}

SHANGHAI_DEST_EXACT = {
    'SH',
    'SHA',
    'LUODONG',
    'BAOSHAN',
    'BAO',
    'BS',
    'ZHB',
    'JGL',
    'WGQ',
    'WGQ1',
    'WGQ2',
    'WGQ3',
    'WGQ4',
    'WGQ5',
    'W1',
    'W2',
    'W3',
    'W4',
    'W5',
}

SHANGHAI_DEST_CONTAINS = {
    'YANGSHAN',
    'YANG0SHAN',
    'YANGSHANG',
    'WAIGAOQIAO',
    'WGQ',
    'WAIER',
    'WAI2',
    'W2',
    'WAISI',
    'WAI4',
    'W4',
    'WAISIQI',
    'WAIWU',
    'WAI5',
    'W5',
    'WAIYI',
    'WAI1',
    'W1',
    'ZHANGHUABANG',
    'JUNGONGLU',
    'JUNLONGLU',
    'LUODONG',
    'TONGHAIMATOU',
}

SHANGHAI_TERMINAL_CODES = {
    'LD',
    'JGL',
    'WGQ1',
    'WGQ2',
    'WGQ4',
    'WGQ5',
    'YS01',
    'YS03',
    'YS04',
}


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


def normalize_dest_text(value):
    """归一化目的港文本，用于上海港规则匹配。"""
    if pd.isna(value):
        return ''

    return ''.join(str(value).upper().split())


def read_csv_with_encoding_fallback(file_path):
    """按常见中文 CSV 编码顺序读取文件。"""
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb18030']
    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise last_error


def load_research_vessel_list():
    """读取研究对象船舶清单，返回原表和 mmsi 白名单。"""
    if not USE_RESEARCH_VESSEL_LIST or not RESEARCH_VESSEL_FILE.exists():
        return pd.DataFrame(), set()

    research_df = read_csv_with_encoding_fallback(RESEARCH_VESSEL_FILE)
    if 'mmsi' not in research_df.columns:
        raise ValueError(
            f'{RESEARCH_VESSEL_FILE.name} 缺少必要字段：mmsi'
        )

    research_df['mmsi'] = pd.to_numeric(
        research_df['mmsi'],
        errors='coerce'
    ).astype('Int64')
    research_df = research_df.dropna(subset=['mmsi']).copy()

    research_mmsi_set = set(research_df['mmsi'].astype('int64'))
    print(
        f"研究对象清单读取完成：{len(research_mmsi_set)} 条唯一 mmsi"
    )

    return research_df, research_mmsi_set


def is_possible_shanghai_dest(value):
    """根据 AIS/ETA 目的地文本判断是否可能到达上海港。"""
    dest = normalize_dest_text(value)

    if not dest:
        return False

    if any(dest.startswith(prefix) for prefix in SHANGHAI_DEST_PREFIXES):
        return True

    if dest in SHANGHAI_DEST_EXACT:
        return True

    if any(keyword in dest for keyword in SHANGHAI_DEST_CONTAINS):
        return True

    if dest in SHANGHAI_TERMINAL_CODES:
        return True

    return False


def load_centerline(centerline_file, reverse=False):
    """加载中心线并转换到米制局部等距投影。"""
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
        centerline_lonlat = shape(geojson_data['geometry'])
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
    return centerline_m, transformer


def project_point_to_centerline(lon, lat, centerline_m, transformer):
    """返回 AIS 点的沿线里程和横向偏离距离，单位 km。"""
    if pd.isna(lon) or pd.isna(lat):
        return np.nan, np.nan

    x, y = transformer.transform(lon, lat)
    ais_point_m = Point(x, y)
    route_s_m = centerline_m.project(ais_point_m)
    projected_point_m = centerline_m.interpolate(route_s_m)
    cross_track_m = ais_point_m.distance(projected_point_m)

    return route_s_m / 1000.0, cross_track_m / 1000.0


def load_vessel_static(ship_type_features):
    """读取船舶静态特征，并对齐训练阶段保存的船舶类型 one-hot 列。"""
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

    length_median = vessel_static['ship_length'].median()
    width_median = vessel_static['ship_width'].median()
    vessel_static['ship_length'] = vessel_static['ship_length'].fillna(length_median)
    vessel_static['ship_width'] = vessel_static['ship_width'].fillna(width_median)

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

    for col in ship_type_features:
        if col not in vessel_static.columns:
            vessel_static[col] = 0

    keep_cols = ['mmsi', 'ship_length', 'ship_width'] + ship_type_features
    return vessel_static[keep_cols], length_median, width_median


def add_time_rolling_sog(df):
    """按船舶六小时历史计算历史均速和 2h/4h 时间窗口均速。"""
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


def load_eta_context():
    """读取 ETA 侧信息，按 mmsi 保留最近一条用于补充出发港和到达预测。"""
    if not INPUT_ETA_FILE.exists():
        return pd.DataFrame(columns=['mmsi'])

    eta_df = pd.read_csv(INPUT_ETA_FILE)
    if 'mmsi' not in eta_df.columns:
        return pd.DataFrame(columns=['mmsi'])

    eta_df['mmsi'] = pd.to_numeric(eta_df['mmsi'], errors='coerce').astype('Int64')

    time_candidates = [
        'postime',
        'sync_update_time',
        'create_time',
    ]
    eta_time_col = next(
        (col for col in time_candidates if col in eta_df.columns),
        None
    )
    if eta_time_col is not None:
        eta_df['_eta_context_time'] = pd.to_datetime(
            eta_df[eta_time_col],
            utc=True,
            errors='coerce'
        )
    else:
        eta_df['_eta_context_time'] = pd.NaT

    for col in [
        'start_postime',
        'eta',
        'local_eta',
        'cta',
        'local_cta',
        'real_arrival_time',
        'ais_eta',
    ]:
        if col in eta_df.columns:
            eta_df[col] = pd.to_datetime(eta_df[col], utc=True, errors='coerce')

    for col in ['rest_hour', 'rest_distance', 'past_hour', 'past_distance']:
        if col in eta_df.columns:
            eta_df[col] = pd.to_numeric(eta_df[col], errors='coerce')

    keep_cols = [
        'mmsi',
        'start_port_code',
        'start_postime',
        'end_port_code',
        'eta',
        'local_eta',
        'cta',
        'local_cta',
        'rest_hour',
        'rest_distance',
        'past_hour',
        'past_distance',
        'ais_dest',
        'ais_eta',
        '_eta_context_time',
    ]
    keep_cols = [col for col in keep_cols if col in eta_df.columns]

    eta_df = (
        eta_df[keep_cols]
        .dropna(subset=['mmsi'])
        .sort_values(['mmsi', '_eta_context_time'])
        .drop_duplicates(subset=['mmsi'], keep='last')
        .reset_index(drop=True)
    )

    return eta_df


def load_live_ais():
    """读取六小时 AIS 历史，并做基础字段标准化。"""
    df = pd.read_csv(INPUT_AIS_FILE)
    _, research_mmsi_set = load_research_vessel_list()

    required_cols = [
        'mmsi',
        'postime',
        'lon',
        'lat',
        'sog',
        'draught',
        'cog',
        'dest',
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f'{INPUT_AIS_FILE.name} 缺少必要字段：{missing_cols}')

    df['mmsi'] = pd.to_numeric(df['mmsi'], errors='coerce').astype('Int64')
    df['voyage_id'] = df['mmsi'].astype(str)
    df['postime'] = pd.to_datetime(df['postime'], utc=True, errors='coerce')
    df['dest_norm'] = df['dest'].map(normalize_dest_text)
    df['is_possible_shanghai_dest'] = df['dest'].map(is_possible_shanghai_dest)

    for col in ['lon', 'lat', 'sog', 'draught', 'cog']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['mmsi', 'postime', 'lon', 'lat']).copy()
    df = df.sort_values(['voyage_id', 'postime']).reset_index(drop=True)

    raw_vessel_count = df['voyage_id'].nunique()

    if research_mmsi_set:
        df = df[df['mmsi'].astype('int64').isin(research_mmsi_set)].copy()
        print(
            f"研究对象 mmsi 筛选：{raw_vessel_count} 条船中保留 "
            f"{df['voyage_id'].nunique()} 条"
        )
    else:
        print("未启用研究对象 mmsi 清单，使用上海港目的地规则筛选。")

    latest_dest_df = (
        df.sort_values(['voyage_id', 'postime'])
        .groupby('voyage_id', as_index=False)
        .tail(1)
        [['voyage_id', 'mmsi', 'dest', 'dest_norm', 'is_possible_shanghai_dest']]
        .rename(
            columns={
                'dest': 'latest_dest',
                'dest_norm': 'latest_dest_norm',
                'is_possible_shanghai_dest': 'latest_dest_match_shanghai',
            }
        )
    )

    eta_context = load_eta_context()
    if not eta_context.empty:
        eta_match_cols = ['mmsi']
        if 'ais_dest' in eta_context.columns:
            eta_context['eta_ais_dest_match_shanghai'] = (
                eta_context['ais_dest'].map(is_possible_shanghai_dest)
            )
            eta_match_cols.append('eta_ais_dest_match_shanghai')
        if 'end_port_code' in eta_context.columns:
            eta_context['eta_end_port_match_shanghai'] = (
                eta_context['end_port_code'].map(is_possible_shanghai_dest)
            )
            eta_match_cols.append('eta_end_port_match_shanghai')

        latest_dest_df = latest_dest_df.merge(
            eta_context[eta_match_cols],
            on='mmsi',
            how='left'
        )
    else:
        latest_dest_df['eta_ais_dest_match_shanghai'] = False
        latest_dest_df['eta_end_port_match_shanghai'] = False

    for col in ['eta_ais_dest_match_shanghai', 'eta_end_port_match_shanghai']:
        if col not in latest_dest_df.columns:
            latest_dest_df[col] = False
        latest_dest_df[col] = latest_dest_df[col].fillna(False).astype(bool)

    latest_dest_df['target_match_shanghai'] = (
        latest_dest_df['latest_dest_match_shanghai']
        | latest_dest_df['eta_ais_dest_match_shanghai']
        | latest_dest_df['eta_end_port_match_shanghai']
    )

    if research_mmsi_set:
        matched_voyage_ids = set(latest_dest_df['voyage_id'])
    else:
        matched_voyage_ids = set(
            latest_dest_df.loc[
                latest_dest_df['target_match_shanghai'],
                'voyage_id'
            ]
        )
        df = df[df['voyage_id'].isin(matched_voyage_ids)].copy()

    latest_dest_context_cols = [
        col for col in latest_dest_df.columns if col != 'mmsi'
    ]
    df = df.merge(
        latest_dest_df[latest_dest_context_cols],
        on='voyage_id',
        how='left'
    )
    print(
        f"最终预测对象：{df['voyage_id'].nunique()} 条船"
    )

    for col in ['sog', 'draught', 'cog']:
        df[col] = (
            df.groupby('voyage_id')[col]
              .transform(lambda s: s.ffill().bfill())
        )
        fallback = df[col].median()
        if pd.isna(fallback):
            fallback = 0.0
        df[col] = df[col].fillna(fallback)

    return df


def build_live_features(df, features, ship_type_features):
    """把六小时 AIS 历史构造成 xgboost_v8 需要的特征。"""
    centerline_m, centerline_transformer = load_centerline(
        CENTERLINE_FILE,
        reverse=REVERSE_CENTERLINE
    )

    vessel_static, length_median, width_median = load_vessel_static(
        ship_type_features
    )
    df = df.merge(vessel_static, on='mmsi', how='left')
    df['ship_length'] = df['ship_length'].fillna(length_median)
    df['ship_width'] = df['ship_width'].fillna(width_median)
    df[ship_type_features] = df[ship_type_features].fillna(0).astype(int)

    df = df.sort_values(['voyage_id', 'postime']).reset_index(drop=True)

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

    df['prev_postime'] = df.groupby('voyage_id')['postime'].shift(1)
    df['dt_h'] = (
        df['postime'] - df['prev_postime']
    ).dt.total_seconds() / 3600.0
    df['dt_h'] = df['dt_h'].fillna(0).clip(lower=0)
    df['active_step_h'] = np.where(
        df['sog'].between(2.0, 12.0)
        & df['dt_h'].between(0.0, 0.5, inclusive='both'),
        df['dt_h'],
        0.0
    )
    df['pure_hours_elapsed'] = (
        df.groupby('voyage_id')['active_step_h'].cumsum()
    )

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

    df['start_route_s_km'] = (
        df.groupby('voyage_id')['route_s_km']
          .transform('first')
    )
    df['route_traveled_km'] = df['route_s_km'] - df['start_route_s_km']
    df['route_traveled_km_clipped'] = df['route_traveled_km'].clip(lower=0)

    gate_x, gate_y = centerline_transformer.transform(GATE_LON, GATE_LAT)
    gate_route_s_km = centerline_m.project(Point(gate_x, gate_y)) / 1000.0
    df['route_remaining_km'] = (
        gate_route_s_km - df['route_s_km']
    ).clip(lower=0)

    df['route_projection_valid'] = (
        df['cross_track_km'] <= MAX_CROSS_TRACK_KM
    ).astype(int)

    df = add_time_rolling_sog(df)

    df['avg_speed_overall'] = (
        df['dist_traveled_km'] / (df['pure_hours_elapsed'] + 1e-5)
    )
    df['avg_route_speed_knots'] = (
        df['route_traveled_km_clipped']
        / (df['pure_hours_elapsed'] + 1e-5)
        / 1.852
    )

    for col in features:
        if col not in df.columns:
            df[col] = 0

    return df


def export_daily_arrival_files(result):
    """按预测到达日期导出 2026-06-30 和 2026-07-01 两个业务文件。"""
    output_specs = [
        ('2026-06-30', ARRIVAL_JUN30_FILE),
        ('2026-07-01', ARRIVAL_JUL01_FILE),
    ]
    output_cols = ['mmsi', 'upload_time', 'pred_arrival_time']

    if result.empty:
        for _, output_file in output_specs:
            pd.DataFrame(columns=output_cols).to_csv(output_file, index=False)
            print(f"已导出空到港日文件: {output_file}")
        return

    upload_time_local = (
        pd.to_datetime(result['postime'], utc=True, errors='coerce')
        .dt.tz_convert(LOCAL_TZ)
    )
    arrival_time_local = (
        pd.to_datetime(result['pred_arrival_time'], utc=True, errors='coerce')
        .dt.tz_convert(LOCAL_TZ)
    )

    for date_text, output_file in output_specs:
        day_start = pd.Timestamp(date_text, tz=LOCAL_TZ)
        day_end = day_start + pd.Timedelta(days=1)
        day_mask = (
            (arrival_time_local >= day_start)
            & (arrival_time_local < day_end)
        )

        day_df = pd.DataFrame({
            'mmsi': result.loc[day_mask, 'mmsi'].to_numpy(),
            'upload_time': (
                upload_time_local.loc[day_mask]
                .dt.strftime('%Y-%m-%d %H:%M:%S%z')
                .to_numpy()
            ),
            'pred_arrival_time': (
                arrival_time_local.loc[day_mask]
                .dt.strftime('%Y-%m-%d %H:%M:%S%z')
                .to_numpy()
            ),
        })
        day_df.to_csv(output_file, index=False)
        print(
            f"{date_text} 当天预计到达船舶: {len(day_df)} 条，"
            f"已导出至: {output_file}"
        )


def export_research_vessel_predictions(result):
    """把 v8 预测到达时间按 mmsi 合并回研究对象清单。"""
    research_df, _ = load_research_vessel_list()
    if research_df.empty:
        return

    pred_col = 'xgboost_v8预测到达时间_北京时间'

    if result.empty:
        research_df[pred_col] = pd.NA
        research_df.to_csv(
            RESEARCH_OUTPUT_FILE,
            index=False,
            encoding='utf-8-sig'
        )
        research_df.to_csv(
            RESEARCH_VESSEL_FILE,
            index=False,
            encoding='utf-8-sig'
        )
        print(f"研究对象预测结果已导出至: {RESEARCH_OUTPUT_FILE}")
        print(f"研究对象原文件已追加预测列: {RESEARCH_VESSEL_FILE}")
        return

    pred_df = result[
        ['mmsi', 'pred_arrival_time']
    ].copy()
    pred_df['mmsi'] = pd.to_numeric(
        pred_df['mmsi'],
        errors='coerce'
    ).astype('Int64')
    pred_df[pred_col] = (
        pd.to_datetime(
            pred_df['pred_arrival_time'],
            utc=True,
            errors='coerce'
        )
        .dt.tz_convert(LOCAL_TZ)
        .dt.strftime('%Y-%m-%d %H:%M:%S')
    )
    pred_df = pred_df[['mmsi', pred_col]]

    output_df = research_df.merge(pred_df, on='mmsi', how='left')
    output_df.to_csv(
        RESEARCH_OUTPUT_FILE,
        index=False,
        encoding='utf-8-sig'
    )
    output_df.to_csv(
        RESEARCH_VESSEL_FILE,
        index=False,
        encoding='utf-8-sig'
    )
    print(f"研究对象预测结果已导出至: {RESEARCH_OUTPUT_FILE}")
    print(f"研究对象原文件已追加预测列: {RESEARCH_VESSEL_FILE}")


def main():
    if not MODEL_FILE.exists() or not FEATURE_META_FILE.exists():
        raise FileNotFoundError(
            '未找到 xgboost_v8 模型文件。请先运行 non_stop_v8.py，'
            '生成 xgboost_v8_model.json 和 xgboost_v8_features.json。'
        )

    feature_meta = json.loads(
        FEATURE_META_FILE.read_text(encoding='utf-8')
    )
    features = feature_meta['features']
    ship_type_features = feature_meta.get('ship_type_features', [])

    model = xgb.XGBRegressor()
    model.load_model(MODEL_FILE)

    ais_df = load_live_ais()
    if ais_df.empty:
        empty_cols = [
            'mmsi',
            'postime',
            'lon',
            'lat',
            'sog',
            'draught',
            'cog',
            'pred_remaining_hours',
            'pred_arrival_time',
        ]
        empty_result = pd.DataFrame(columns=empty_cols)
        empty_result.to_csv(OUTPUT_FILE, index=False)
        export_daily_arrival_files(empty_result)
        export_research_vessel_predictions(empty_result)
        print("没有筛选到可能到达上海港的船舶，已导出空预测结果。")
        print(f"预测结果已导出至: {OUTPUT_FILE}")
        return

    feature_df = build_live_features(
        ais_df,
        features,
        ship_type_features
    )
    eta_context = load_eta_context()

    latest_df = (
        feature_df
        .sort_values(['voyage_id', 'postime'])
        .groupby('voyage_id', as_index=False)
        .tail(1)
        .copy()
    )

    if not eta_context.empty:
        latest_df = latest_df.merge(
            eta_context,
            on='mmsi',
            how='left',
            suffixes=('', '_eta_context')
        )

    X_live = latest_df[features]
    pred_remaining_hours = model.predict(X_live)
    pred_remaining_hours = np.clip(pred_remaining_hours, 0, None)

    result = latest_df[
        [
            'mmsi',
            'postime',
            'lon',
            'lat',
            'sog',
            'draught',
            'cog',
            'route_remaining_km',
            'cross_track_km',
            'route_projection_valid',
            'latest_dest',
            'latest_dest_norm',
            'latest_dest_match_shanghai',
            'eta_ais_dest_match_shanghai',
            'eta_end_port_match_shanghai',
            'target_match_shanghai',
        ]
    ].copy()

    eta_output_cols = [
        'start_port_code',
        'start_postime',
        'end_port_code',
        'eta',
        'local_eta',
        'cta',
        'local_cta',
        'rest_hour',
        'rest_distance',
        'past_hour',
        'past_distance',
        'ais_dest',
        'ais_eta',
    ]
    for col in eta_output_cols:
        if col in latest_df.columns:
            result[col] = latest_df[col].to_numpy()

    result['pred_remaining_hours'] = pred_remaining_hours
    result['pred_arrival_time'] = (
        result['postime']
        + pd.to_timedelta(result['pred_remaining_hours'], unit='h')
    )

    result = result.sort_values('pred_arrival_time').reset_index(drop=True)
    result.to_csv(OUTPUT_FILE, index=False)
    export_daily_arrival_files(result)
    export_research_vessel_predictions(result)

    print(f"已完成 {len(result)} 条船舶 ETA 预测")
    print(f"预测结果已导出至: {OUTPUT_FILE}")
    print(result.head(30).to_string(index=False))


if __name__ == '__main__':
    main()
