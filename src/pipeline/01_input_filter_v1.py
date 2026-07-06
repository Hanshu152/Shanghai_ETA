import json
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point, shape
from shapely.ops import transform


SCRIPT_DIR = Path(__file__).resolve().parent
XGBOOST_DIR = next(
    path for path in [SCRIPT_DIR, *SCRIPT_DIR.parents]
    if path.name == "XGBoost"
)

# PRED_DATA_DIR = XGBOOST_DIR / "data" / "pred_data" / "705"
MODEL_SOURCE_DIR = XGBOOST_DIR / "src" / "model"

# INPUT_AIS_FILE = PRED_DATA_DIR / "ais_6his_6h.csv"
# INPUT_ETA_FILE = PRED_DATA_DIR / "eta_6his_6h.csv"
CENTERLINE_FILE = MODEL_SOURCE_DIR / "downstream_channel_centerline_relaxed_v3_manual_control.geojson"

# OUTPUT_FILE = PRED_DATA_DIR / "input_filtered_v1.csv"
# REPORT_FILE = PRED_DATA_DIR / "input_filter_report_v1.csv"

REVERSE_CENTERLINE = False

# 最终点距离中心线过远，认为不在主干航道附近
MAX_FINAL_CROSS_TRACK_KM = 10.0

# 仅将同时满足净位移、趋势斜率和方向占比的船舶判定为明确上游航行。
ROUTE_SMOOTHING_WINDOW = 5
MOVEMENT_NOISE_KM = 0.2
UPSTREAM_NET_THRESHOLD_KM = 2.0
UPSTREAM_SLOPE_THRESHOLD_KMH = 0.3
UPSTREAM_DISTANCE_RATIO_THRESHOLD = 0.65
STATIONARY_NET_THRESHOLD_KM = 2.0
STATIONARY_ROUTE_SPAN_KM = 3.0
MIN_DIRECTION_POINTS = 3
MIN_DIRECTION_TIMESPAN_HOURS = 0.5
STATIONARY_STATUS_CODES = {1, 5}
MAX_STATIONARY_SOG_KNOTS = 0.5

# 外高桥门线。预测点必须位于 A->B 门线的西北侧，即尚未越过门线。
GATE_A_LON = 121.3766833333
GATE_A_LAT = 31.6277333333
GATE_B_LON = 121.31625
GATE_B_LAT = 31.5172333333

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

def parse_args():
    parser = argparse.ArgumentParser(
        description="ETA预测流水线"
    )
    parser.add_argument(
        "data_batch",
        help="数据批次目录，例如705"
    )
    return parser.parse_args()

def configure_paths(data_batch):
    pred_data_dir = (
        XGBOOST_DIR
        / "data"
        / "pred_data"
        / data_batch
    )

    return {
        "pred_data_dir": pred_data_dir,
        "input_ais_file": pred_data_dir / "ais_6his_6h.csv",
        "input_eta_file": pred_data_dir / "eta_6his_6h.csv",
        "input_filtered_file": pred_data_dir / "input_filtered.csv",
        "filter_report_file": pred_data_dir / "input_filter_report.csv",
        "prediction_all_file": pred_data_dir / "eta_predictions_all.csv",
    }

def normalize_dest_text(value):
    """归一化目的港文本，用于上海港规则匹配。"""
    if pd.isna(value):
        return ''

    return ''.join(str(value).upper().split())


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


def gate_side_cross_value(lon, lat):
    """计算点相对 A->B 门线的二维叉积；非正值表示位于西北侧。"""
    return (
        (GATE_B_LON - GATE_A_LON) * (lat - GATE_A_LAT)
        - (GATE_B_LAT - GATE_A_LAT) * (lon - GATE_A_LON)
    )


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
    """返回 AIS 点沿中心线里程和横向偏离距离，单位 km。"""
    if pd.isna(lon) or pd.isna(lat):
        return np.nan, np.nan

    x, y = transformer.transform(lon, lat)
    ais_point_m = Point(x, y)
    route_s_m = centerline_m.project(ais_point_m)
    projected_point_m = centerline_m.interpolate(route_s_m)
    cross_track_m = ais_point_m.distance(projected_point_m)

    return route_s_m / 1000.0, cross_track_m / 1000.0


def load_ais_data():
    """读取六小时 AIS 历史，并做基础字段标准化。"""
    df = pd.read_csv(INPUT_AIS_FILE)

    required_cols = [
        'mmsi',
        'postime',
        'lon',
        'lat',
        'sog',
        'draught',
        'cog',
        'dest',
        'status',
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f'{INPUT_AIS_FILE.name} 缺少必要字段：{missing_cols}')

    df['mmsi'] = pd.to_numeric(df['mmsi'], errors='coerce').astype('Int64')
    df['postime'] = pd.to_datetime(df['postime'], utc=True, errors='coerce')

    for col in ['lon', 'lat', 'sog', 'draught', 'cog', 'status']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['dest_norm'] = df['dest'].map(normalize_dest_text)
    df['ais_dest_match_shanghai'] = df['dest'].map(is_possible_shanghai_dest)

    df = df.dropna(subset=['mmsi', 'postime', 'lon', 'lat']).copy()
    df = df.sort_values(['mmsi', 'postime']).reset_index(drop=True)

    return df


def load_eta_context():
    """读取 ETA 侧信息，按 mmsi 保留最近一条作为目的地补充。"""
    if not INPUT_ETA_FILE.exists():
        return pd.DataFrame(columns=['mmsi'])

    eta_df = pd.read_csv(INPUT_ETA_FILE)
    if 'mmsi' not in eta_df.columns:
        return pd.DataFrame(columns=['mmsi'])

    eta_df['mmsi'] = pd.to_numeric(eta_df['mmsi'], errors='coerce').astype('Int64')

    time_candidates = ['postime', 'sync_update_time', 'create_time']
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

    keep_cols = [
        'mmsi',
        'ais_dest',
        'end_port_code',
        'start_port_code',
        'start_postime',
        'eta',
        'rest_hour',
        'rest_distance',
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

    if 'ais_dest' in eta_df.columns:
        eta_df['eta_ais_dest_match_shanghai'] = (
            eta_df['ais_dest'].map(is_possible_shanghai_dest)
        )
    else:
        eta_df['eta_ais_dest_match_shanghai'] = False

    if 'end_port_code' in eta_df.columns:
        eta_df['eta_end_port_match_shanghai'] = (
            eta_df['end_port_code'].map(is_possible_shanghai_dest)
        )
    else:
        eta_df['eta_end_port_match_shanghai'] = False

    return eta_df


def summarize_direction(group):
    """汇总单船六小时沿线运动，并仅识别证据充分的上游航行。"""
    group = group.sort_values('postime')
    valid = group.dropna(subset=['postime', 'route_s_smoothed_km'])

    point_count = len(valid)
    if point_count == 0:
        return pd.Series({
            'first_route_s_km': np.nan,
            'last_route_s_km': np.nan,
            'route_delta_km': np.nan,
            'route_span_km': np.nan,
            'route_trend_kmh': np.nan,
            'upstream_distance_km': 0.0,
            'downstream_distance_km': 0.0,
            'upstream_distance_ratio': 0.0,
            'downstream_positive_ratio': 0.0,
            'point_count': 0,
            'time_span_hours': 0.0,
            'latest_status': np.nan,
            'latest_sog': np.nan,
            'stationary_status_ratio': 0.0,
            'stationary_by_status': False,
            'direction_quality_ok': False,
            'confirmed_upstream': False,
            'movement_class': 'insufficient_data',
        })

    start_time = valid['postime'].iloc[0]
    end_time = valid['postime'].iloc[-1]
    time_span_hours = (end_time - start_time).total_seconds() / 3600.0

    start_window = valid.loc[
        valid['postime'] <= start_time + pd.Timedelta(minutes=30),
        'route_s_smoothed_km'
    ]
    end_window = valid.loc[
        valid['postime'] >= end_time - pd.Timedelta(minutes=30),
        'route_s_smoothed_km'
    ]
    first_route_s_km = start_window.median()
    last_route_s_km = end_window.median()
    route_delta_km = last_route_s_km - first_route_s_km
    route_span_km = (
        valid['route_s_smoothed_km'].max()
        - valid['route_s_smoothed_km'].min()
    )

    elapsed_hours = (
        valid['postime'] - start_time
    ).dt.total_seconds().to_numpy() / 3600.0
    if point_count >= 2 and time_span_hours > 0:
        route_trend_kmh = np.polyfit(
            elapsed_hours,
            valid['route_s_smoothed_km'].to_numpy(),
            1
        )[0]
    else:
        route_trend_kmh = np.nan

    deltas = valid['route_s_smoothed_km'].diff().dropna()
    meaningful_deltas = deltas[deltas.abs() >= MOVEMENT_NOISE_KM]
    upstream_distance_km = -meaningful_deltas[meaningful_deltas < 0].sum()
    downstream_distance_km = meaningful_deltas[meaningful_deltas > 0].sum()
    directional_distance_km = upstream_distance_km + downstream_distance_km
    if directional_distance_km > 0:
        upstream_distance_ratio = upstream_distance_km / directional_distance_km
    else:
        upstream_distance_ratio = 0.0
    if len(meaningful_deltas) > 0:
        downstream_positive_ratio = (meaningful_deltas > 0).mean()
    else:
        downstream_positive_ratio = 0.0

    status_values = pd.to_numeric(group['status'], errors='coerce')
    sog_values = pd.to_numeric(group['sog'], errors='coerce')
    latest_status = status_values.iloc[-1]
    latest_sog = sog_values.iloc[-1]
    stationary_status_flags = (
        status_values.isin(STATIONARY_STATUS_CODES)
        & sog_values.le(MAX_STATIONARY_SOG_KNOTS)
    )
    stationary_status_ratio = stationary_status_flags.mean()
    stationary_by_status = (
        not pd.isna(latest_status)
        and not pd.isna(latest_sog)
        and int(latest_status) in STATIONARY_STATUS_CODES
        and latest_sog <= MAX_STATIONARY_SOG_KNOTS
    )
    stationary_by_motion = (
        abs(route_delta_km) <= STATIONARY_NET_THRESHOLD_KM
        and route_span_km <= STATIONARY_ROUTE_SPAN_KM
    )
    direction_quality_ok = (
        point_count >= MIN_DIRECTION_POINTS
        and time_span_hours >= MIN_DIRECTION_TIMESPAN_HOURS
    )
    confirmed_upstream = (
        direction_quality_ok
        and not stationary_by_status
        and route_delta_km <= -UPSTREAM_NET_THRESHOLD_KM
        and route_trend_kmh <= -UPSTREAM_SLOPE_THRESHOLD_KMH
        and upstream_distance_ratio >= UPSTREAM_DISTANCE_RATIO_THRESHOLD
    )

    if confirmed_upstream:
        movement_class = 'upstream'
    elif stationary_by_status or stationary_by_motion:
        movement_class = 'stationary'
    elif direction_quality_ok and route_delta_km > STATIONARY_NET_THRESHOLD_KM:
        movement_class = 'downstream'
    else:
        movement_class = 'uncertain_but_not_upstream'

    return pd.Series({
        'first_route_s_km': first_route_s_km,
        'last_route_s_km': last_route_s_km,
        'route_delta_km': route_delta_km,
        'route_span_km': route_span_km,
        'route_trend_kmh': route_trend_kmh,
        'upstream_distance_km': upstream_distance_km,
        'downstream_distance_km': downstream_distance_km,
        'upstream_distance_ratio': upstream_distance_ratio,
        'downstream_positive_ratio': downstream_positive_ratio,
        'point_count': point_count,
        'time_span_hours': time_span_hours,
        'latest_status': latest_status,
        'latest_sog': latest_sog,
        'stationary_status_ratio': stationary_status_ratio,
        'stationary_by_status': stationary_by_status,
        'direction_quality_ok': direction_quality_ok,
        'confirmed_upstream': confirmed_upstream,
        'movement_class': movement_class,
    })


def build_vessel_filter_report(ais_df):
    """按船舶生成中心线距离、目的地和下行趋势筛选报告。"""
    centerline_m, transformer = load_centerline(
        CENTERLINE_FILE,
        reverse=REVERSE_CENTERLINE
    )

    projection_result = [
        project_point_to_centerline(lon, lat, centerline_m, transformer)
        for lon, lat in zip(ais_df['lon'], ais_df['lat'])
    ]
    projection_result = pd.DataFrame(
        projection_result,
        columns=['route_s_km', 'cross_track_km'],
        index=ais_df.index
    )
    ais_df = pd.concat([ais_df, projection_result], axis=1)

    ais_df = ais_df.sort_values(['mmsi', 'postime']).reset_index(drop=True)
    ais_df['route_s_smoothed_km'] = (
        ais_df.groupby('mmsi')['route_s_km']
        .transform(
            lambda values: values.rolling(
                ROUTE_SMOOTHING_WINDOW,
                center=True,
                min_periods=1
            ).median()
        )
    )
    ais_df['route_s_delta_km'] = (
        ais_df.groupby('mmsi')['route_s_smoothed_km'].diff()
    )

    latest_df = (
        ais_df
        .groupby('mmsi', as_index=False)
        .tail(1)
        [[
            'mmsi',
            'postime',
            'lon',
            'lat',
            'dest',
            'dest_norm',
            'ais_dest_match_shanghai',
            'route_s_km',
            'cross_track_km',
            'status',
        ]]
        .rename(
            columns={
                'postime': 'latest_postime',
                'lon': 'latest_lon',
                'lat': 'latest_lat',
                'dest': 'latest_dest',
                'dest_norm': 'latest_dest_norm',
                'route_s_km': 'last_route_s_km',
                'cross_track_km': 'latest_cross_track_km',
                'status': 'latest_ais_status',
            }
        )
    )

    trend_df = (
        ais_df.groupby('mmsi', sort=False)
        .apply(summarize_direction, include_groups=False)
        .reset_index()
    )

    report_df = latest_df.merge(trend_df, on='mmsi', how='left')

    eta_context = load_eta_context()
    if not eta_context.empty:
        report_df = report_df.merge(eta_context, on='mmsi', how='left')
    else:
        report_df['eta_ais_dest_match_shanghai'] = False
        report_df['eta_end_port_match_shanghai'] = False

    for col in ['eta_ais_dest_match_shanghai', 'eta_end_port_match_shanghai']:
        if col not in report_df.columns:
            report_df[col] = False
        report_df[col] = report_df[col].fillna(False).astype(bool)

    report_df['destination_match_shanghai'] = (
        report_df['ais_dest_match_shanghai'].fillna(False)
        | report_df['eta_ais_dest_match_shanghai']
        | report_df['eta_end_port_match_shanghai']
    )

    report_df['final_point_near_centerline'] = (
        report_df['latest_cross_track_km'] <= MAX_FINAL_CROSS_TRACK_KM
    )
    report_df['gate_side_cross_value'] = gate_side_cross_value(
        report_df['latest_lon'],
        report_df['latest_lat']
    )
    report_df['latest_point_before_gate'] = (
        report_df['gate_side_cross_value'] <= 0
    )
    report_df['not_upstream_navigation'] = ~report_df['confirmed_upstream']

    report_df['keep_for_prediction'] = (
        report_df['final_point_near_centerline']
        & report_df['latest_point_before_gate']
        & report_df['destination_match_shanghai']
        & report_df['not_upstream_navigation']
    )

    report_df['filter_reason'] = 'kept'
    report_df.loc[
        ~report_df['final_point_near_centerline'],
        'filter_reason'
    ] = 'latest_point_far_from_centerline'
    report_df.loc[
        report_df['final_point_near_centerline']
        & ~report_df['latest_point_before_gate'],
        'filter_reason'
    ] = 'latest_point_has_crossed_gate'
    report_df.loc[
        report_df['final_point_near_centerline']
        & report_df['latest_point_before_gate']
        & ~report_df['destination_match_shanghai'],
        'filter_reason'
    ] = 'destination_not_shanghai'
    report_df.loc[
        report_df['final_point_near_centerline']
        & report_df['latest_point_before_gate']
        & report_df['destination_match_shanghai']
        & ~report_df['not_upstream_navigation'],
        'filter_reason'
    ] = 'confirmed_upstream_navigation'

    return ais_df, report_df


def main():
    global PRED_DATA_DIR
    global INPUT_AIS_FILE, INPUT_ETA_FILE
    global OUTPUT_FILE, REPORT_FILE
    args = parse_args()
    paths = configure_paths(args.data_batch)

    PRED_DATA_DIR = paths["pred_data_dir"]
    INPUT_AIS_FILE = paths["input_ais_file"]
    INPUT_ETA_FILE = paths["input_eta_file"]
    OUTPUT_FILE = paths["input_filtered_file"]
    REPORT_FILE = paths["filter_report_file"]
    PRED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    ais_df = load_ais_data()
    ais_with_projection_df, report_df = build_vessel_filter_report(ais_df)

    kept_mmsi = set(
        report_df.loc[report_df['keep_for_prediction'], 'mmsi']
    )
    filtered_df = ais_with_projection_df[
        ais_with_projection_df['mmsi'].isin(kept_mmsi)
    ].copy()

    filtered_df.to_csv(OUTPUT_FILE, index=False)
    report_df.to_csv(REPORT_FILE, index=False)

    print(f"原始 AIS 船舶数: {ais_df['mmsi'].nunique()}")
    print(f"保留船舶数: {len(kept_mmsi)}")
    print(f"输出模型输入数据: {OUTPUT_FILE}")
    print(f"输出筛选报告: {REPORT_FILE}")
    print("\n筛选原因分布:")
    print(report_df['filter_reason'].value_counts(dropna=False).to_string())


if __name__ == '__main__':
    main()
